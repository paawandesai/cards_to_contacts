from __future__ import annotations

"""Parsing heuristics that convert raw OCR output into structured :class:`Contact`."""

import logging
import re
from typing import List

from .models import Contact

LOGGER = logging.getLogger(__name__)

# Pre-compiled regex patterns
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", flags=re.I)
PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3})?[\s.-]?)?\(?(?:\d{3})\)?[\s.-]?\d{3}[\s.-]?\d{4}",
)
WEBSITE_RE = re.compile(r"(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}", flags=re.I)


LINE_SPLIT_RE = re.compile(r"[\r\n]+")


# --- Helper functions ------------------------------------------------------

def _first_non_empty(lines: List[str]) -> str | None:
    for line in lines:
        if line.strip():
            return line.strip()
    return None


def parse_contact(raw_text: str) -> Contact:
    """Parse raw OCR text into a :class:`Contact` dataclass instance."""
    lines = [l.strip() for l in LINE_SPLIT_RE.split(raw_text) if l.strip()]

    emails = EMAIL_RE.findall(raw_text)
    phones = PHONE_RE.findall(raw_text)
    website_match = WEBSITE_RE.search(raw_text)
    website = website_match.group(0) if website_match else None

    # Remove extracted artefacts from lines for cleaner name/title inference
    cleaned_lines = [
        l
        for l in lines
        if l not in emails and l not in phones and (website is None or l != website)
    ]

    full_name = _first_non_empty(cleaned_lines)
    title = None
    company = None

    if full_name and len(cleaned_lines) >= 2:
        title = cleaned_lines[1]
    if full_name and len(cleaned_lines) >= 3:
        company = cleaned_lines[2]

    # Address: join remaining unclassified lines
    known_lines = {full_name, title, company}
    address_parts = [l for l in cleaned_lines if l not in known_lines]
    address = "\n".join(address_parts) if address_parts else None

    LOGGER.debug(
        "Parsed contact: name=%s, title=%s, company=%s, email=%s, phone=%s, website=%s",
        full_name,
        title,
        company,
        emails,
        phones,
        website,
    )

    return Contact(
        full_name=full_name,
        title=title,
        company=company,
        emails=emails,
        phones=phones,
        website=website,
        address=address,
    ) 