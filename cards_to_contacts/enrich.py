from __future__ import annotations

"""Contact enrichment & validation utilities.

Optional online lookups (Clearbit autocomplete) are used when an environment
variable ``CLEARBIT_API_KEY`` is present. Otherwise, enrichment falls back to
local heuristics.
"""

from collections import Counter
import logging
import os
import re
from typing import Optional

import phonenumbers
import requests
import tldextract
from email_validator import validate_email, EmailNotValidError
from nameparser import HumanName
from rapidfuzz import process as fuzz_process, fuzz

from .models import Contact

LOGGER = logging.getLogger(__name__)

CLEARBIT_SUGGEST_URL = "https://autocomplete.clearbit.com/v1/companies/suggest?query={q}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise_phone(raw: str, default_region: str = "US") -> Optional[str]:
    """Return E.164-formatted phone number or ``None`` if parsing fails."""
    try:
        number = phonenumbers.parse(raw, default_region)
        if not phonenumbers.is_possible_number(number):
            return None
        return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return None


def _dedupe_preserve_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _suggest_company(domain_or_name: str) -> Optional[str]:
    """Hit Clearbit autocomplete and return best-matched company name, if plausible."""
    api_key = os.getenv("CLEARBIT_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.get(CLEARBIT_SUGGEST_URL.format(q=domain_or_name), timeout=4)
        resp.raise_for_status()
        suggestions = resp.json()
        if not suggestions:
            return None
        # Take the top suggestion if similarity is high enough
        best = suggestions[0]
        guess = best.get("name")
        if guess and fuzz.token_set_ratio(domain_or_name.lower(), guess.lower()) > 70:
            return guess
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Clearbit lookup failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_contact(c: Contact) -> Contact:
    """Return an enriched **copy** of *c* with validated & normalised fields.

    1. Deduplicate and validate e-mails (RFC compliant)
    2. Parse and format phone numbers to E.164
    3. Derive company from domain (e-mail or website) if missing
    4. Optionally confirm/correct company via Clearbit autocomplete
    5. Title-case the person's name where safe
    """
    # Validate/dedupe emails
    valid_emails = []
    for e in c.emails:
        try:
            clean = validate_email(e).email
            valid_emails.append(clean)
        except EmailNotValidError:
            LOGGER.debug("Dropping invalid email: %s", e)
    c.emails = _dedupe_preserve_order(valid_emails)

    # Phone numbers -> E.164
    formatted_phones = []
    for p in c.phones:
        fmt = _normalise_phone(p)
        if fmt:
            formatted_phones.append(fmt)
    c.phones = _dedupe_preserve_order(formatted_phones)

    # Name cleanup – title-case unless ALL CAPS etc.
    if c.full_name:
        hn = HumanName(c.full_name)
        formatted_name = str(hn).strip()
        if formatted_name:
            c.full_name = formatted_name

    # Company inference
    if not c.company:
        # Try from email domain / website
        domain_source = None
        if c.emails:
            domain_source = c.emails[0].split("@", 1)[-1]
        elif c.website:
            ext = tldextract.extract(c.website)
            domain_source = f"{ext.domain}.{ext.suffix}" if ext.domain else None
        if domain_source:
            inferred = re.sub(r"\.(co|com|net|org)$", "", domain_source, flags=re.I)
            c.company = inferred.title()

    # Online confirmation via Clearbit
    if c.company:
        corrected = _suggest_company(c.company)
        if corrected:
            c.company = corrected

    return c 