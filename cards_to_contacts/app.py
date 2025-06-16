from __future__ import annotations

"""Streamlit application entry-point."""

import logging
from typing import List

import sys
from pathlib import Path

import streamlit as st

# Must be first Streamlit command
# Attempt to set page configuration (may already be set by Streamlit)
try:
    st.set_page_config(
        page_title="Business-card ➜ Contact CSV/Excel Exporter",
        layout="wide",
    )
except RuntimeError:
    # Streamlit raises RuntimeError if experimental flag or API changed
    pass
except Exception as exc:  # noqa: BLE001
    # Ignore if already configured
    import streamlit.errors as stre

    if isinstance(exc, stre.StreamlitAPIException):
        pass
    else:
        raise

import numpy as np  # pylint: disable=import-error,import-outside-toplevel
import pandas as pd

import sys, subprocess, textwrap, os
print("▶︎ Running with:", sys.executable)
try:
    import phonenumbers
    print("▶︎ phonenumbers imported OK, version:", phonenumbers.__version__)
except ModuleNotFoundError as e:
    print("▶︎ phonenumbers NOT importable:", e)

# also show what pip thinks is installed in **this** interpreter
out = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=columns"])
print(textwrap.indent(out.decode(), "    "))

# Ensure project root is on PYTHONPATH for `streamlit run cards_to_contacts/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from cards_to_contacts import exporter, ocr, parser
from cards_to_contacts.models import Contact
from cards_to_contacts.enrich import enrich_contact

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.title("📇 Business-card ➜ Contact CSV/Excel Exporter")

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    ocr_lang = st.text_input("Tesseract language(s)", value="eng", help="Space-separated list, e.g. 'eng deu' for English + German")
    show_raw = st.checkbox("Show raw OCR text", value=False)
    st.markdown("---")
    st.subheader("📤 Notion export")
    notion_token = st.text_input(
        "Notion integration token",
        type="password",
        help="Generate an internal integration in Notion and paste the secret_... token",
    )
    notion_db_id = st.text_input(
        "Database ID",
        help="The target database where contacts will be appended",
    )
    st.markdown("---")
    st.markdown(
        "### ℹ️ How it works\n1. Upload card photo(s) → OCR → parse\n2. Edit any fields in the table\n3. Download as CSV or Excel ✨",
    )

# --- Main uploader ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or many business-card images", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)

if uploaded_files:
    contacts: List[Contact] = []
    for file in uploaded_files:
        st.subheader(f"🖼 {file.name}")
        img_bytes = file.read()

        try:
            results = ocr.process_image_bytes(img_bytes, lang=ocr_lang)
        except Exception as exc:  # noqa: BLE001
            st.error(f"OCR failed for {file.name}: {exc}")
            LOGGER.exception("OCR failure")
            continue

        for idx, (card_img, raw_text) in enumerate(results, start=1):
            cols = st.columns([1, 3])
            with cols[0]:
                # Convert numpy BGR to RGB for display
                rgb = np.ascontiguousarray(card_img[:, :, ::-1])
                st.image(rgb, caption=f"Card #{idx}")
            with cols[1]:
                contact = enrich_contact(parser.parse_contact(raw_text))
                contacts.append(contact)

                if show_raw:
                    with st.expander("Raw OCR text"):
                        st.code(raw_text)

    # Deduplicate contacts by email if duplicates produced
    unique_contacts: dict[str, Contact] = {}
    for c in contacts:
        key = c.emails[0] if c.emails else c.full_name or id(c)
        unique_contacts[key] = c
    contacts = list(unique_contacts.values())

    if not contacts:
        st.warning("No contacts could be parsed.")
        st.stop()

    st.subheader("📝 Preview & edit")
    df = exporter.contacts_to_dataframe(contacts)
    edited_df: pd.DataFrame = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    st.markdown("---")
    st.subheader("⬇️ Export")

    csv_bytes = exporter.get_csv_bytes(edited_df)
    excel_bytes = exporter.get_excel_bytes(edited_df)

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=exporter.make_filename("csv"),
            mime="text/csv",
        )
    with col_xlsx:
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name=exporter.make_filename("xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # --- Notion export ----------------------------------------------------
    if notion_token and notion_db_id:
        if st.button("Export to Notion"):
            with st.spinner("Exporting contacts to Notion..."):
                try:
                    created = exporter.export_to_notion(edited_df, notion_token, notion_db_id)
                    st.success(f"✅ Exported {created} contacts to Notion!")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Failed to export to Notion: {exc}")

    # Log panel
    st.markdown("---")
    st.subheader("Logs")
    with st.expander("Show debug log"):
        st.code("Processed %d contacts" % len(contacts))
else:
    st.info("Upload one or more images to begin.") 