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
    line_clean = line.strip()
    
    # Skip very short or empty lines
    if len(line_clean) < 2:
        return 'other'
    
    # Skip lines that are clearly contact info (already extracted)
    if EMAIL_RE.search(line_clean) or PHONE_RE.search(line_clean) or WEBSITE_RE.search(line_clean):
        return 'other'
    
    # Check for title keywords first (higher priority)
    # But be more careful about names that might contain job-like words
    if TITLE_KEYWORDS.search(line):
        # If it's a short line with typical name structure, prefer name classification
        words = line_clean.split()
        if len(words) == 2 and all(word[0].isupper() and word[1:].islower() for word in words):
            # Looks like "FirstName LastName" pattern, check if it's really a title
            title_matches = TITLE_KEYWORDS.findall(line)
            if len(title_matches) == 1 and title_matches[0].lower() in ['developer', 'engineer', 'manager', 'designer']:
                # Could be a surname, prioritize name classification
                return 'name'
        return 'title'
    
    # Check for company indicators
    if COMPANY_INDICATORS.search(line):
        return 'company'
    
    # Check for address patterns
    if ADDRESS_PATTERNS.search(line):
        return 'address'
    
    # Enhanced name detection heuristics
    words = line_clean.split()
    if len(words) <= 4 and not any(char.isdigit() for char in line_clean):
        # Additional name indicators
        name_indicators = [
            all(word[0].isupper() for word in words if word),  # All words capitalized
            len(words) == 2,  # Common first + last name pattern
            any(word.lower() in ['jr', 'sr', 'iii', 'ii', 'phd', 'md', 'esq'] for word in words),  # Name suffixes
            # Check for common name patterns
            len(words) >= 2 and all(word.isalpha() for word in words),  # All alphabetic multi-word
        ]
        
        if any(name_indicators):
            return 'name'
        
        # If it's short and has proper case, likely a name
        if len(words) <= 3 and line_clean[0].isupper():
            return 'name'
    
    # Check for department/division names (could be part of title or company)
    dept_keywords = ['department', 'dept', 'division', 'team', 'group', 'unit']
    if any(keyword in line_lower for keyword in dept_keywords):
        return 'company'  # Treat as company info
    
    # Check for location indicators (city, state, country)
    location_keywords = ['usa', 'united states', 'canada', 'uk', 'california', 'texas', 'new york', 'florida']
    if any(keyword in line_lower for keyword in location_keywords):
        return 'address'
    
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
    
    # Select best candidates with improved logic
    classified['full_name'] = names[0] if names else None
    classified['title'] = titles[0] if titles else None
    classified['company'] = companies[0] if companies else None
    
    # If we don't have a name yet, use the first line as fallback
    if not classified['full_name'] and cleaned_lines:
        classified['full_name'] = cleaned_lines[0]
    
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


def _validate_field_assignments(classified_fields: dict, emails: List[str], phones: List[str]) -> dict:
    """Validate and improve field assignments for key data quality."""
    validated = classified_fields.copy()
    
    # Ensure we have critical contact information
    has_contact_info = bool(emails or phones)
    
    # If we have contact info but no name, try harder to find one
    if has_contact_info and not validated.get('full_name'):
        # Look for the first reasonable text that could be a name
        for key, value in validated.items():
            if value and key != 'address' and len(value.split()) <= 3:
                # Enhanced name validation
                if (value[0].isupper() and 
                    not any(char.isdigit() for char in value) and
                    len(value) > 3 and
                    not COMPANY_INDICATORS.search(value) and
                    not TITLE_KEYWORDS.search(value)):
                    validated['full_name'] = value
                    validated[key] = None  # Remove from original field
                    break
    
    # Validate title makes sense
    if validated.get('title'):
        title = validated['title']
        # If "title" looks more like a company name, swap them
        if (COMPANY_INDICATORS.search(title) and 
            validated.get('company') and 
            not COMPANY_INDICATORS.search(validated['company'])):
            validated['title'], validated['company'] = validated['company'], validated['title']
        
        # If title contains both title and company info, try to split
        if ',' in title and not validated.get('company'):
            parts = [p.strip() for p in title.split(',')]
            if len(parts) == 2:
                if COMPANY_INDICATORS.search(parts[1]):
                    validated['title'] = parts[0]
                    validated['company'] = parts[1]
                elif TITLE_KEYWORDS.search(parts[0]):
                    validated['title'] = parts[0]
                    validated['company'] = parts[1]
    
    # Enhanced company validation
    if validated.get('company'):
        company = validated['company']
        # Remove common prefixes/suffixes that might be misclassified
        company_clean = re.sub(r'^(at|@)\s+', '', company, flags=re.I)
        if company_clean != company:
            validated['company'] = company_clean
    
    # Clean up address field with better validation
    if validated.get('address'):
        addr_lines = [line.strip() for line in validated['address'].split('\n') if line.strip()]
        # Remove any lines that look like they should be other fields
        clean_addr_lines = []
        for line in addr_lines:
            line_type = _classify_line(line)
            if line_type in ['address', 'other']:
                # Keep address lines more liberally to avoid losing address info
                clean_addr_lines.append(line)
            elif not validated.get('company') and line_type == 'company':
                validated['company'] = line
            elif not validated.get('title') and line_type == 'title':
                validated['title'] = line
                
        validated['address'] = '\n'.join(clean_addr_lines) if clean_addr_lines else None
    
    return validated


def _extract_structured_data(raw_text: str) -> dict:
    """Extract emails, phones, and websites from text."""
    # Enhanced email extraction
    emails = EMAIL_RE.findall(raw_text)
    # Remove duplicates and normalize
    emails = list(set(email.lower().strip() for email in emails))
    
    # Enhanced phone extraction with better cleaning
    phone_matches = PHONE_RE.findall(raw_text)
    phones = []
    for phone in phone_matches:
        # Clean and validate phone numbers
        cleaned = re.sub(r'[^0-9+]', '', phone)
        if len(cleaned) >= 10:  # Minimum valid phone length
            phones.append(phone)  # Keep original formatting for display
    
    # Enhanced website extraction
    website = None
    website_matches = WEBSITE_RE.findall(raw_text)
    for match in website_matches:
        # Check if this match is not part of an email address
        if not any(match in email for email in emails):
            website = match  # Keep original format for now
            break
    
    return {
        'emails': emails,
        'phones': phones,
        'website': website
    }


def parse_contact(raw_text: str, ocr_confidence: float = None) -> Contact:
    """Parse raw OCR text into a :class:`Contact` dataclass instance."""
    lines = [l.strip() for l in LINE_SPLIT_RE.split(raw_text) if l.strip()]

    # Extract structured data with enhanced processing
    structured_data = _extract_structured_data(raw_text)
    emails = structured_data['emails']
    phones = structured_data['phones']
    website = structured_data['website']

    # Remove extracted artifacts from lines for cleaner text processing
    cleaned_lines = [
        l for l in lines
        if l not in emails and l not in phones and (website is None or l != website)
    ]

    # Use smart field classification
    classified_fields = _smart_parse_fields(cleaned_lines)
    
    # Validate and improve field assignments
    validated_fields = _validate_field_assignments(classified_fields, emails, phones)
    
    # Ensure we have some address if address-like lines exist
    if not validated_fields.get('address'):
        potential_address_lines = []
        for line in cleaned_lines:
            if (ADDRESS_PATTERNS.search(line) or 
                any(keyword in line.lower() for keyword in ['street', 'ave', 'road', 'blvd', 'drive', 'suite', 'floor']) or
                re.search(r'\d{5}', line) or  # ZIP code
                re.search(r'\d+\s+\w+', line)):  # Street number + name
                potential_address_lines.append(line)
        
        if potential_address_lines:
            validated_fields['address'] = '\n'.join(potential_address_lines)
    
    full_name = validated_fields.get('full_name')
    title = validated_fields.get('title')
    company = validated_fields.get('company')
    address = validated_fields.get('address')

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
        ocr_confidence=ocr_confidence,
    ) 