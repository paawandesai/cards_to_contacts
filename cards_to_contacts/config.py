from __future__ import annotations

"""Application-wide configuration constants and helper utilities."""

from datetime import datetime, timezone

TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def timestamp_now_tz() -> str:
    """Return a UTC timestamp string following :pydata:`TIMESTAMP_FMT`."""
    return datetime.now(tz=timezone.utc).strftime(TIMESTAMP_FMT) 