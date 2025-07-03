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


def _is_business_card_aspect_ratio(width: int, height: int, tolerance: float = 0.25) -> bool:
    """Check if dimensions match standard business card aspect ratios.
    
    Standard business card ratios:
    - US: 3.5" x 2" (1.75:1)
    - ISO 216: 85mm x 55mm (1.55:1)
    - Credit card: 85.6mm x 53.98mm (1.59:1)
    """
    if width == 0 or height == 0:
        return False
    
    ratio = max(width, height) / min(width, height)
    target_ratios = [1.75, 1.55, 1.59]  # Standard aspect ratios
    
    return any(abs(ratio - target) / target <= tolerance for target in target_ratios)


def _filter_contours_by_geometry(contours: List[np.ndarray], img_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Filter contours based on business card geometric properties."""
    h_img, w_img = img_shape
    total_area = h_img * w_img
    
    valid_contours = []
    
    for cnt in contours:
        # Area filtering
        area = cv2.contourArea(cnt)
        area_ratio = area / total_area
        if not 0.015 < area_ratio < 0.6:  # Adjusted for better range
            continue
        
        # Approximate to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)  # More precise approximation
        
        # Must be roughly rectangular (3-6 vertices after approximation)
        if not 3 <= len(approx) <= 6:
            continue
        
        # Check bounding rectangle aspect ratio
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        
        if not _is_business_card_aspect_ratio(int(width), int(height)):
            continue
        
        # Check if contour is reasonably convex (business cards are rectangular)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.85:  # Should be quite solid/rectangular
                continue
        
        valid_contours.append(cnt)
    
    return valid_contours


def detect_card_contours(img: np.ndarray) -> List[np.ndarray]:
    """Detect rectangular contours likely to contain individual business cards.

    Uses enhanced filtering based on standard business card dimensions and geometry.
    Returns a list of cropped card BGR images. If no contours, returns the original image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for better edge detection
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use multiple threshold techniques and combine results
    thresh_methods = []
    
    # Method 1: Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    thresh_methods.append(thresh1)
    
    # Method 2: OTSU threshold
    _, thresh2 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_methods.append(thresh2)
    
    # Method 3: Adaptive threshold with mean
    thresh3 = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 8
    )
    thresh_methods.append(thresh3)
    
    # Combine thresholds for better edge detection
    combined_thresh = np.zeros_like(thresh1)
    for thresh in thresh_methods:
        combined_thresh = cv2.bitwise_or(combined_thresh, thresh)
    
    # Morphological operations to clean up and connect edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours using geometric properties
    valid_contours = _filter_contours_by_geometry(contours, gray.shape)
    
    # Sort by area (largest first) and process, but limit to reasonable number
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    
    # Process up to 5 largest contours to avoid processing too many false positives
    max_cards_to_process = min(5, len(valid_contours))
    valid_contours = valid_contours[:max_cards_to_process]
    
    card_images: List[np.ndarray] = []
    
    for cnt in valid_contours:
        # Get minimum area rectangle for better perspective correction
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Perspective-warp the card to a flat rectangle
        try:
            warped = _four_point_transform(img, box.astype(np.float32))
            
            # Final size validation
            h, w = warped.shape[:2]
            if h < 100 or w < 100:  # Minimum reasonable size
                continue
            
            # Verify aspect ratio of final warped image
            if not _is_business_card_aspect_ratio(w, h, tolerance=0.35):
                continue
            
            card_images.append(warped)
            
        except Exception as e:
            LOGGER.debug("Failed to warp contour: %s", e)
            continue
    
    # Fallback: if detection failed, try with more lenient criteria
    if not card_images:
        LOGGER.debug("Primary detection failed, trying fallback with relaxed criteria...")
        
        # Relaxed filtering for fallback with improved multiple card detection
        fallback_cards = []
        for cnt in contours:
            area_ratio = cv2.contourArea(cnt) / (gray.shape[0] * gray.shape[1])
            if 0.05 < area_ratio < 0.8:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) >= 4:
                    try:
                        warped = _four_point_transform(img, approx.reshape(4, 2))
                        if warped.shape[0] > 60 and warped.shape[1] > 60:
                            fallback_cards.append(warped)
                            if len(fallback_cards) >= 3:  # Limit fallback attempts
                                break
                    except:
                        continue
        
        # Add fallback cards if they seem reasonable
        if fallback_cards:
            card_images.extend(fallback_cards)
    
    # Final fallback: use entire image
    if not card_images:
        LOGGER.debug("All detection methods failed; using full image as single card.")
        card_images.append(img)
    
    # Quality check: remove cards that are too similar (likely duplicates)
    if len(card_images) > 1:
        filtered_cards = [card_images[0]]  # Keep first card
        for card in card_images[1:]:
            # Simple duplicate check based on size similarity
            is_duplicate = False
            for existing_card in filtered_cards:
                size_ratio = (card.shape[0] * card.shape[1]) / (existing_card.shape[0] * existing_card.shape[1])
                if 0.8 < size_ratio < 1.2:  # Similar size, likely duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_cards.append(card)
        
        card_images = filtered_cards[:3]  # Limit to max 3 cards for processing efficiency

    LOGGER.info("Detected %d card(s) in photo using enhanced detection", len(card_images))
    return card_images


# --- OCR -------------------------------------------------------------------

def _enhance_for_ocr(gray: np.ndarray) -> List[np.ndarray]:
    """Generate multiple enhanced versions of grayscale image for better OCR results."""
    enhanced_versions = []
    
    # Version 1: CLAHE + sharpening (original method)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    denoised = cv2.medianBlur(sharpened, 3)
    enhanced_versions.append(denoised)
    
    # Version 2: Stronger CLAHE for low contrast images
    clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_strong = clahe_strong.apply(gray)
    enhanced_versions.append(enhanced_strong)
    
    # Version 3: Gaussian blur then sharpen (for noisy images)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    unsharp_kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]])
    unsharp = cv2.filter2D(blurred, -1, unsharp_kernel)
    enhanced_versions.append(unsharp)
    
    # Version 4: Morphological operations for text clarity
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph_enhanced = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)
    enhanced_versions.append(morph_enhanced)
    
    # Version 5: Bilateral filter + adaptive threshold preprocessing
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced_versions.append(bilateral)
    
    return enhanced_versions


def _calculate_ocr_confidence(text: str, img_shape: tuple) -> float:
    """Calculate confidence score for OCR result."""
    if not text.strip():
        return 0.0
    
    text_clean = text.strip()
    
    # Basic length check
    if len(text_clean) < 5:
        return 0.2
    
    # Check for reasonable character density
    total_pixels = img_shape[0] * img_shape[1]
    char_density = len(text_clean) / (total_pixels / 10000)  # chars per 10k pixels
    
    # Calculate confidence based on various factors
    confidence = 0.5  # base confidence
    
    # Length factor
    if 10 <= len(text_clean) <= 200:
        confidence += 0.2
    
    # Character variety (letters, numbers, symbols)
    has_letters = any(c.isalpha() for c in text_clean)
    has_numbers = any(c.isdigit() for c in text_clean)
    has_symbols = any(c in '@.-()' for c in text_clean)
    
    if has_letters:
        confidence += 0.1
    if has_numbers:
        confidence += 0.1
    if has_symbols:
        confidence += 0.1
    
    # Penalize excessive special characters or repeated characters
    special_ratio = sum(1 for c in text_clean if not c.isalnum() and c != ' ') / len(text_clean)
    if special_ratio > 0.3:
        confidence -= 0.2
    
    return min(1.0, max(0.0, confidence))


def ocr_image(img: np.ndarray, lang: str = "eng") -> tuple[str, float]:
    """Run enhanced Tesseract OCR on a BGR OpenCV image and return recognised UTF-8 text with confidence."""
    # Convert to PIL and handle EXIF rotation
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_image = rotate_exif(pil_image)
    
    # Convert back to numpy for processing
    rgb_array = np.array(pil_image)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # Deskew the image
    gray = deskew_image(gray)
    
    # Generate multiple enhanced versions
    enhanced_versions = _enhance_for_ocr(gray)
    
    # Multiple OCR strategies with different PSM modes
    ocr_configs = [
        ('--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-+()#&, ', 'PSM 6 (uniform block)'),
        ('--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-+()#&, ', 'PSM 11 (sparse text)'),
        ('--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-+()#&, ', 'PSM 8 (single word)'),
        ('--psm 3', 'PSM 3 (fully automatic)'),
        ('--psm 13', 'PSM 13 (raw line)'),
    ]
    
    best_text = ""
    best_confidence = 0.0
    best_method = "none"
    
    # Try each enhancement version with each OCR configuration
    for i, enhanced in enumerate(enhanced_versions):
        # Convert to RGB for Tesseract
        processed_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        for config, method_name in ocr_configs:
            try:
                text = pytesseract.image_to_string(processed_rgb, lang=lang, config=config)
                confidence = _calculate_ocr_confidence(text, enhanced.shape)
                
                LOGGER.debug(f"OCR attempt {i+1}/{len(enhanced_versions)} with {method_name}: {len(text.strip())} chars, conf: {confidence:.3f}")
                
                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence
                    best_method = f"enhancement_{i+1}_{method_name}"
                    
            except Exception as e:
                LOGGER.debug(f"OCR failed with {method_name}: {e}")
                continue
    
    # Final fallback with basic settings if nothing worked well
    if best_confidence < 0.3:
        try:
            processed_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            fallback_text = pytesseract.image_to_string(processed_rgb, lang=lang)
            fallback_confidence = _calculate_ocr_confidence(fallback_text, gray.shape)
            
            if fallback_confidence > best_confidence:
                best_text = fallback_text
                best_confidence = fallback_confidence
                best_method = "basic_fallback"
                
        except Exception as e:
            LOGGER.warning("Even basic OCR failed: %s", e)
    
    LOGGER.info(f"Best OCR result: {len(best_text.strip())} chars, confidence: {best_confidence:.3f}, method: {best_method}")
    LOGGER.debug("Tesseract output (first 150 chars): %s", best_text[:150])
    
    return best_text, best_confidence


# --- Public pipeline -------------------------------------------------------

def process_image_bytes(img_bytes: bytes, lang: str = "eng") -> List[Tuple[np.ndarray, str, float]]:
    """Full pipeline: bytes → card crops → OCR → list of (thumbnail, raw_text, confidence)."""
    img = bytes_to_cv2(img_bytes)
    cards = detect_card_contours(img)

    results: List[Tuple[np.ndarray, str, float]] = []
    for card in cards:
        text, confidence = ocr_image(card, lang=lang)
        results.append((card, text, confidence))
    return results 