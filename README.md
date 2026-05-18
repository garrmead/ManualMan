# ManualMan — Pump Manual RAG Chatbot

Ask questions about your pump manuals (Goulds, Aurora, Gorman-Rupp, etc.) and get cited answers with page references and diagrams.

---

## Prerequisites

- **Python 3.11 or newer** — check with `python --version`
- A **Voyage AI** API key → [dash.voyageai.com](https://dash.voyageai.com/)
- An **Anthropic** API key → [console.anthropic.com](https://console.anthropic.com/)

---

## Setup (one-time)

Open a terminal (PowerShell or Command Prompt on Windows) in the `ManualMan` folder.

### 1 — Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

You should see `(venv)` at the start of your prompt when it's active.

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

This takes a few minutes the first time. Everything installs locally — no Docker, no cloud services.

### 3 — Add your API keys

Copy the example file and fill it in:

```powershell
copy .env.example .env      # Windows
# cp .env.example .env      # Mac/Linux
```

Then open `.env` in Notepad (or any text editor) and replace the placeholder values:

```
VOYAGE_API_KEY=your_actual_voyage_key
ANTHROPIC_API_KEY=your_actual_anthropic_key
```

Save and close. The app reads this file at startup — never share or commit it.

---

## Running the app

```powershell
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser. The app stays running until you press `Ctrl+C` in the terminal.

---

## Phase status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Project scaffold — sidebar, tabs, API key check | ✅ Done |
| 2 | PDF upload, parse, chunk preview & edit table | ⏳ Coming |
| 3 | Embed chunks via Voyage AI → store in ChromaDB | ⏳ Coming |
| 4 | Chat interface with citations and diagram display | ⏳ Coming |
| 5 | Manual management — view, delete, re-process | ⏳ Coming |

---

## Project layout

```
ManualMan/
├── app.py                  Main Streamlit app — run this
├── config.py               Tuneable settings (chunk size, model names, etc.)
├── requirements.txt        All Python dependencies
├── .env                    Your API keys (never commit this)
├── .env.example            Key template — safe to commit
├── data/
│   ├── uploads/            PDFs you upload (not committed to git)
│   ├── extracted_images/   Images pulled from PDFs (not committed)
│   └── chroma_db/          Local vector database (not committed)
└── utils/
    ├── __init__.py
    ├── pdf_parser.py       (Phase 2) PDF text + image extraction
    ├── embedder.py         (Phase 3) Voyage AI embedding + ChromaDB writes
    └── retriever.py        (Phase 4) RAG query pipeline
```

---

## Phase 1 test checklist

After running `streamlit run app.py`:

1. **Browser opens** at `http://localhost:8501` — you see a two-column layout.
2. **Sidebar** shows "ManualMan" title, API key status (✅ or ❌), a PDF uploader, and a Phase Status list.
3. **API keys**: if you filled in `.env`, both rows show ✅. If not, both show ❌ and a warning.
4. **Three tabs** appear in the main area: `📋 Parse & Edit`, `💬 Chat`, `📚 Manuals`.
5. **Upload test**: click the uploader, select any PDF. The sidebar shows the filename and size; the Parse & Edit tab shows "X PDF(s) ready."
6. No errors appear in the terminal.

If anything looks wrong, check the terminal for error messages and compare against the setup steps above.
