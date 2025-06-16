from __future__ import annotations

"""Data models for contacts extracted from business cards."""

import re
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator

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

    @validator("phones", each_item=True)
    def _clean_phone(cls, v: str) -> str:  # noqa: D401, N804
        """Strip non-numeric artifacts from phone numbers, keep leading ``+``."""
        return PHONE_CLEAN_RE.sub("", v)

    def as_dict(self) -> dict[str, str | list[str] | None]:
        """Return a JSON-serialisable ``dict`` with snake-case keys."""
        return self.model_dump()

    # For DataFrame display coherence
    def as_flat_dict(self) -> dict[str, str]:
        """Return a flat ``dict[str, str]`` with lists serialised as ``;``-joined strings."""
        data: dict[str, str | list[str] | None] = self.as_dict()
        flattened: dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                flattened[key] = ""
            elif isinstance(value, list):
                flattened[key] = "; ".join(map(str, value))
            else:
                flattened[key] = str(value)
        return flattened 