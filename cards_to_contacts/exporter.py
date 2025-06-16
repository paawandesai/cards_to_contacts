from __future__ import annotations

"""CSV/Excel export helpers."""

import io
import logging
from typing import List

import pandas as pd
from pandas import DataFrame

from .config import timestamp_now_tz
from .models import Contact

LOGGER = logging.getLogger(__name__)


# --- core helpers ----------------------------------------------------------

def contacts_to_dataframe(contacts: List[Contact]) -> DataFrame:
    """Convert a list of :class:`Contact` objects to a :class:`pandas.DataFrame`."""
    return pd.DataFrame([c.as_flat_dict() for c in contacts])


def get_csv_bytes(df: DataFrame) -> bytes:
    """Return UTF-8 CSV bytes for download via Streamlit."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    LOGGER.debug("Generated CSV size: %d bytes", len(csv_bytes))
    return csv_bytes


def get_excel_bytes(df: DataFrame) -> bytes:
    """Return Excel (xlsx) bytes for download via Streamlit."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Contacts")
    buffer.seek(0)
    excel_bytes = buffer.read()
    LOGGER.debug("Generated Excel size: %d bytes", len(excel_bytes))
    return excel_bytes


def make_filename(ext: str) -> str:
    """Return a timestamped filename with given *ext* (no leading dot)."""
    return f"contacts_{timestamp_now_tz()}.{ext}"


def export_to_notion(df: DataFrame, notion_api_key: str, database_id: str) -> int:
    """Export *df* rows to the given Notion *database_id*.

    Each row is converted to a Notion page where columns map to database
    properties. The database **must already contain** properties with the
    following names (case-sensitive) and suitable types:

    • Name – *title*
    • Company – *rich_text* (or *text*)
    • Job Title – *rich_text*
    • Email – *email*
    • Phone – *phone_number*
    • Website – *url*
    • Address – *rich_text*

    Parameters
    ----------
    df:
        DataFrame returned by :pyfunc:`contacts_to_dataframe` (or the edited
        version). List columns are expected to be serialised with ``;``
        separators – exactly what :pyfunc:`contacts_to_dataframe` produces.
    notion_api_key:
        A Notion integration token (starts with ``secret_``).
    database_id:
        The ID of the target database (can be the 32-character or dashed
        variant).

    Returns
    -------
    int
        Number of contacts successfully created.
    """

    try:
        from notion_client import Client  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Package 'notion_client' is required for Notion export. Install it\n"
            "via `pip install notion-client`."
        ) from exc

    import pandas as pd  # Local import to avoid hard dependency otherwise

    client = Client(auth=notion_api_key)

    # Fetch database metadata to know available properties & their types
    try:
        db_meta = client.databases.retrieve(database_id=database_id)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to retrieve Notion database %s: %s", database_id, exc)
        raise

    db_props: dict = db_meta["properties"]

    def _build_prop(prop_name: str, value: str):
        """Return a Notion property payload matching existing *prop_name*."""
        if not value or prop_name not in db_props:
            return None

        prop_type = db_props[prop_name]["type"]
        if prop_type == "title":
            return {"title": [{"text": {"content": value}}]}
        if prop_type == "rich_text":
            return {"rich_text": [{"text": {"content": value}}]}
        if prop_type == "email":
            # Notion supports only a single email value – take the first if multiple
            first_email = value.split(";")[0].strip()
            return {"email": first_email}
        if prop_type == "phone_number":
            first_phone = value.split(";")[0].strip()
            return {"phone_number": first_phone}
        if prop_type == "url":
            return {"url": value}
        if prop_type == "multi_select":
            options = [v.strip() for v in value.split(";") if v.strip()]
            return {"multi_select": [{"name": opt} for opt in options]}

        # Fallback – best-effort as rich text
        return {"rich_text": [{"text": {"content": value}}]}

    # Mapping between DataFrame columns and Notion property names
    column_map = {
        "full_name": "Name",
        "company": "Company",
        "title": "Job Title",
        "emails": "Email",
        "phones": "Phone",
        "website": "Website",
        "address": "Address",
    }

    created = 0
    for _, row in df.iterrows():
        row = row.fillna("")
        props: dict[str, dict] = {}

        for col_name, notion_prop in column_map.items():
            if col_name not in row:
                continue
            cell_val = str(row[col_name]).strip()
            # Remove surrounding brackets for lists (e.g. "['a', 'b']") if user edited
            cell_val = cell_val.strip("[]")
            payload = _build_prop(notion_prop, cell_val)
            if payload:
                props[notion_prop] = payload

        if not props:
            continue  # Skip if nothing to push

        client.pages.create(parent={"database_id": database_id}, properties=props)
        created += 1

    LOGGER.info("Exported %d contacts to Notion database %s", created, database_id)
    return created 