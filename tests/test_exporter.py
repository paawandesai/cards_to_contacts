from __future__ import annotations

"""Unit tests for DataFrame export helpers."""

import pandas as pd

from cards_to_contacts import exporter
from cards_to_contacts.models import Contact


def _sample_contacts() -> list[Contact]:
    return [
        Contact(
            full_name="Alice Smith",
            title="CEO",
            company="Wonder Works",
            emails=["alice@wonder.com"],
            phones=["+1 555 987 6543"],
            website="wonder.com",
        ),
        Contact(
            full_name="Bob Jones",
            title="CTO",
            company="Tech Corp",
            emails=["bob@techcorp.io"],
            phones=["+1 555 555 0000"],
        ),
    ]


def test_contacts_to_dataframe() -> None:
    df = exporter.contacts_to_dataframe(_sample_contacts())
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "full_name",
        "title",
        "company",
        "emails",
        "phones",
        "website",
        "address",
    }
    assert len(df) == 2


def test_export_bytes_non_empty() -> None:
    df = exporter.contacts_to_dataframe(_sample_contacts())
    csv_bytes = exporter.get_csv_bytes(df)
    xlsx_bytes = exporter.get_excel_bytes(df)

    assert csv_bytes.startswith(b"full_name,")
    # XLSX starts with PK header for ZIP files
    assert xlsx_bytes[:2] == b"PK"

    assert len(csv_bytes) > 50
    assert len(xlsx_bytes) > 100 