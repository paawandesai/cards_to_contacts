from __future__ import annotations

"""Data models for contacts extracted from business cards."""

import re
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

PHONE_CLEAN_RE = re.compile(r"[^0-9+]")


class Contact(BaseModel):
    """A structured representation of a business-card contact."""

    full_name: Optional[str] = Field(None, description="Full name of the person")
    title: Optional[str] = Field(None, description="Job title / role")
    company: Optional[str] = Field(None, description="Company name")
    emails: List[EmailStr] = Field(default_factory=list, description="List of e-mails")
    phones: List[str] = Field(default_factory=list, description="List of phone numbers")
    website: Optional[str] = Field(None, description="Website / URL")
    address: Optional[str] = Field(None, description="Postal address, best-effort")
    ocr_confidence: Optional[float] = Field(None, description="OCR confidence score (0.0 to 1.0)")

    @field_validator("phones", mode="before")
    @classmethod
    def _clean_phones(cls, v: List[str]) -> List[str]:
        """Strip non-numeric artifacts from phone numbers, keep leading ``+``."""
        if not isinstance(v, list):
            return v
        cleaned = []
        for phone in v:
            if isinstance(phone, str):
                cleaned_phone = PHONE_CLEAN_RE.sub("", phone)
                if len(cleaned_phone) >= 10:  # Minimum valid phone length
                    cleaned.append(cleaned_phone)
        return cleaned

    @field_validator("full_name", mode="before")
    @classmethod
    def _validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Clean and validate full name."""
        if not v:
            return None
        
        # Remove extra whitespace and ensure proper capitalization
        cleaned = " ".join(v.strip().split())
        
        # Basic validation - should contain letters and reasonable length
        if len(cleaned) < 2 or len(cleaned) > 100:
            return None
        
        # Should contain at least some letters
        if not any(c.isalpha() for c in cleaned):
            return None
            
        return cleaned

    @field_validator("emails", mode="before") 
    @classmethod
    def _validate_emails(cls, v: List[str]) -> List[EmailStr]:
        """Validate email addresses."""
        if not isinstance(v, list):
            return v
        
        validated = []
        for email in v:
            if isinstance(email, str) and "@" in email:
                # Basic email validation will be handled by EmailStr
                try:
                    validated.append(email.lower().strip())
                except:
                    continue
        return validated

    def as_dict(self) -> dict[str, str | list[str] | None | float]:
        """Return a JSON-serialisable ``dict`` with snake-case keys."""
        return self.model_dump()

    # For DataFrame display coherence
    def as_flat_dict(self) -> dict[str, str]:
        """Return a flat ``dict[str, str]`` with lists serialised as ``;``-joined strings."""
        data: dict[str, str | list[str] | None | float] = self.as_dict()
        flattened: dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                flattened[key] = ""
            elif isinstance(value, list):
                flattened[key] = "; ".join(map(str, value))
            elif isinstance(value, float):
                flattened[key] = f"{value:.3f}" if value is not None else ""
            else:
                flattened[key] = str(value)
        return flattened 