from __future__ import annotations

"""OCR utilities built on OpenCV and Tesseract."""

import logging
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

# --- Image helpers ---------------------------------------------------------

def bytes_to_cv2(img_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into an OpenCV BGR ndarray."""
    image_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")
    return img


def rotate_exif(pil_img: Image.Image) -> Image.Image:
    """Auto-rotate a PIL image according to its EXIF Orientation tag, if present."""
    return ImageOps.exif_transpose(pil_img)


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Deskew an image using OpenCV moments-based approach.

    Parameters
    ----------
    gray: np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Rotated (deskewed) image in grayscale.
    """
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# --- Card detection --------------------------------------------------------

# Helper utilities for perspective transform


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL has smallest sum
    rect[2] = pts[np.argmax(s)]  # BR has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR has smallest diff (x−y)
    rect[3] = pts[np.argmax(diff)]  # BL has largest diff
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective transform so the 4-pt *pts* becomes a rect image."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


def detect_card_contours(img: np.ndarray) -> List[np.ndarray]:
    """Detect rectangular contours likely to contain individual business cards.

    For simplicity, we assume cards are the largest rectangles in the image.
    Returns a list of cropped card BGR images. If no contours, returns the original image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold isolates light cards from darker table/background
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15,
    )

    # Closing to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape
    total_area = h_img * w_img

    card_images: List[np.ndarray] = []

    for cnt in contours:
        area_ratio = cv2.contourArea(cnt) / total_area
        if not 0.02 < area_ratio < 0.7:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Perspective-warp the card to a flat rectangle
        warped = _four_point_transform(img, approx.reshape(4, 2))

        # Discard extremely small warped images (noise)
        if warped.shape[0] < 60 or warped.shape[1] < 60:
            continue

        card_images.append(warped)

    # Fallback: if detection failed, process entire image as single card
    if not card_images:
        LOGGER.debug("Fallback: no card contours met criteria; using full image.")
        card_images.append(img)

    LOGGER.info("Detected %d card(s) in photo", len(card_images))
    return card_images


# --- OCR -------------------------------------------------------------------

def ocr_image(img: np.ndarray, lang: str = "eng") -> str:
    """Run Tesseract OCR on a BGR OpenCV image and return recognised UTF-8 text."""
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_image = rotate_exif(pil_image)
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    gray = deskew_image(gray)

    # Recreate BGR for pytesseract API (expects RGB or PIL)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    text = pytesseract.image_to_string(processed, lang=lang)
    LOGGER.debug("Tesseract output (first 100 chars): %s", text[:100])
    return text


# --- Public pipeline -------------------------------------------------------

def process_image_bytes(img_bytes: bytes, lang: str = "eng") -> List[Tuple[np.ndarray, str]]:
    """Full pipeline: bytes → card crops → OCR → list of (thumbnail, raw_text)."""
    img = bytes_to_cv2(img_bytes)
    cards = detect_card_contours(img)

    results: List[Tuple[np.ndarray, str]] = []
    for card in cards:
        text = ocr_image(card, lang=lang)
        results.append((card, text))
    return results 