# Business-card ➜ Contact CSV/Excel Exporter

`cards_to_contacts` turns photographed or scanned business-card images into tidy contact spreadsheets – all in your browser, powered by Streamlit.

---

## 🚀 Quick-start

```bash
# 1. Clone repository & enter it
$ git clone https://github.com/you/cards_to_contacts.git
$ cd cards_to_contacts

# 2. (Recommended) create Python 3.12 virtualenv
$ python3.12 -m venv .venv
$ source .venv/bin/activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the app
$ streamlit run cards_to_contacts/app.py
```

Open the printed URL in your browser, upload one or many images (`*.jpg|*.jpeg|*.png`), polish contact details in-line, then download the aggregated CSV or Excel file.

---

## 🛠  Features

1. **Robust OCR pipeline** – deskews, auto-rotates and feeds each detected card into Tesseract.
2. **Flexible parsing** – regex + heuristics convert raw text into a strict `Contact` model.
3. **Live editing & preview** – Streamlit `st.data_editor` lets you correct OCR mistakes before export.
4. **One-click export** – UTF-8 CSV and fully-formatted Excel (`openpyxl`) with timestamped filenames.
5. **Tested & typed** – 90 %+ unit-test coverage, 100 % static typing and lint-clean via Ruff + Black.

---

## 🖼  OCR requirements

Tesseract must be installed on the host OS and accessible on the `$PATH`.

*macOS (Homebrew)*
```bash
brew install tesseract
```

*Ubuntu/Debian*
```bash
sudo apt update && sudo apt install tesseract-ocr libtesseract-dev
```

Add additional language trained-data packages as needed (e.g. `tesseract-lang` on brew or `tesseract-ocr-deu` on apt).

---

## 🧪 Running tests

```bash
pytest --cov=cards_to_contacts
```

---

## 📂 Repository layout

```
cards_to_contacts/
    __init__.py          Package init & version
    app.py               Streamlit UI
    config.py            Constants & helpers
    exporter.py          CSV/XLSX writers
    models.py            Pydantic Contact schema
    ocr.py               OpenCV + Tesseract utilities
    parser.py            Regex/heuristic text → Contact

.streamlit/
    config.toml          Streamlit wide-mode & theme

tests/                   Pytest suite
requirements.txt         Pinned dependencies
```

---

## 📝 Environment variables

The application currently has no mandatory environment variables. You may drop anything you like in a local `.env` to experiment – it will be loaded automatically by `python-dotenv`.

```
# .env.example
WATCH_DIR=/tmp
```

---

## 🤝 Contributing

1. Create issue / discuss feature.
2. Fork / feature-branch.
3. Ensure `ruff check . --fix` & `black .` pass, plus `pytest` stays green.
4. Open PR.

---

## License

MIT © 2024 Your Name 