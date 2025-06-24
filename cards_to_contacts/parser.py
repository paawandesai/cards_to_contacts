from __future__ import annotations

"""Parsing heuristics that convert raw OCR output into structured :class:`Contact`."""

import logging
import re
from typing import List, Optional

from .models import Contact

LOGGER = logging.getLogger(__name__)

# Enhanced regex patterns
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", flags=re.I)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})",
)
WEBSITE_RE = re.compile(r"(?:https?://)?(?:www\.)?[\w.-]+\.[a-z]{2,}(?:/\S*)?", flags=re.I)

# Patterns for better field classification
TITLE_KEYWORDS = re.compile(
    r'\b(?:CEO|CTO|CFO|COO|VP|Vice President|President|Director|Manager|Senior|Lead|Principal|'
    r'Engineer|Developer|Designer|Analyst|Consultant|Specialist|Coordinator|Assistant|'
    r'Executive|Founder|Owner|Partner|Sales|Marketing|HR|Operations|Finance|Legal|Admin)\b',
    flags=re.I
)

COMPANY_INDICATORS = re.compile(
    r'\b(?:Inc|LLC|Corp|Corporation|Company|Co\.|Ltd|Limited|Group|Associates|'
    r'Partners|Solutions|Services|Systems|Technologies|Tech|Consulting|'
    r'International|Global|Enterprises|Holdings)\b',
    flags=re.I
)

# Address patterns
ADDRESS_PATTERNS = re.compile(
    r'\b(?:\d+\s+\w+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|'
    r'Ct|Court|Pl|Place|Way|Circle|Pkwy|Parkway)|'
    r'P\.?O\.?\s+Box\s+\d+|'
    r'\d{5}(?:-\d{4})?)\b',
    flags=re.I
)


LINE_SPLIT_RE = re.compile(r"[\r\n]+")


# --- Helper functions ------------------------------------------------------

def _first_non_empty(lines: List[str]) -> str | None:
    for line in lines:
        if line.strip():
            return line.strip()
    return None


def _classify_line(line: str) -> str:
    """Classify a line as 'name', 'title', 'company', 'address', or 'other'."""
    line_lower = line.lower()
    
    # Check for title keywords
    if TITLE_KEYWORDS.search(line):
        return 'title'
    
    # Check for company indicators
    if COMPANY_INDICATORS.search(line):
        return 'company'
    
    # Check for address patterns
    if ADDRESS_PATTERNS.search(line):
        return 'address'
    
    # Heuristics for names (typically shorter, contain common name patterns)
    if len(line.split()) <= 3 and not any(char.isdigit() for char in line):
        # Likely a name if it's short and has no numbers
        return 'name'
    
    return 'other'


def _smart_parse_fields(cleaned_lines: List[str]) -> dict:
    """Use content-based classification to identify fields."""
    classified = {}
    line_classifications = [(line, _classify_line(line)) for line in cleaned_lines]
    
    # Find the best candidates for each field type
    names = [line for line, cls in line_classifications if cls == 'name']
    titles = [line for line, cls in line_classifications if cls == 'title']
    companies = [line for line, cls in line_classifications if cls == 'company']
    addresses = [line for line, cls in line_classifications if cls == 'address']
    others = [line for line, cls in line_classifications if cls == 'other']
    
    # Select best candidates (first occurrence of each type)
    classified['full_name'] = names[0] if names else cleaned_lines[0] if cleaned_lines else None
    classified['title'] = titles[0] if titles else None
    classified['company'] = companies[0] if companies else None
    
    # Handle address - combine all address-like lines
    all_address_lines = addresses + [line for line in others if ADDRESS_PATTERNS.search(line)]
    classified['address'] = '\n'.join(all_address_lines) if all_address_lines else None
    
    # Fallback logic if smart classification didn't find enough
    if not classified['title'] and len(cleaned_lines) >= 2:
        # If no title found and we have unused lines, use positional fallback
        remaining_lines = [line for line in cleaned_lines 
                          if line not in [classified['full_name'], classified['company']] 
                          and line not in all_address_lines]
        if remaining_lines:
            classified['title'] = remaining_lines[0]
    
    if not classified['company'] and len(cleaned_lines) >= 3:
        # Similar fallback for company
        remaining_lines = [line for line in cleaned_lines 
                          if line not in [classified['full_name'], classified['title']] 
                          and line not in all_address_lines]
        if remaining_lines:
            classified['company'] = remaining_lines[0]
    
    return classified


def parse_contact(raw_text: str) -> Contact:
    """Parse raw OCR text into a :class:`Contact` dataclass instance."""
    lines = [l.strip() for l in LINE_SPLIT_RE.split(raw_text) if l.strip()]

    # Extract structured data first
    emails = EMAIL_RE.findall(raw_text)
    phones = PHONE_RE.findall(raw_text)
    
    # Find websites, but exclude email domains
    website = None
    website_matches = WEBSITE_RE.findall(raw_text)
    for match in website_matches:
        # Check if this match is not part of an email address
        if not any(match in email for email in emails):
            website = match
            break

    # Remove extracted artifacts from lines for cleaner text processing
    cleaned_lines = [
        l for l in lines
        if l not in emails and l not in phones and (website is None or l != website)
    ]

    # Use smart field classification
    classified_fields = _smart_parse_fields(cleaned_lines)
    
    full_name = classified_fields.get('full_name')
    title = classified_fields.get('title')
    company = classified_fields.get('company')
    address = classified_fields.get('address')

    LOGGER.debug(
        "Parsed contact: name=%s, title=%s, company=%s, email=%s, phone=%s, website=%s, address=%s",
        full_name,
        title,
        company,
        emails,
        phones,
        website,
        address,
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