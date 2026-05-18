# app.py — Main entry point. Run with: streamlit run app.py
#
# This file wires together all tabs of the application. Each phase adds
# content to one of the tabs below. Phase 1 is just the scaffold — you
# should see a working sidebar and three empty tabs.

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os

# Load API keys from .env file (must exist before importing config)
load_dotenv()

import config

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ManualMan — Pump Manual RAG",
    page_icon="pump_icon",   # text fallback; swap to a .png path if you have one
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure all data directories exist at startup ──────────────────────────────
for directory in [config.UPLOADS_DIR, config.IMAGES_DIR, config.CHROMA_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# ── Session state defaults ─────────────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction, so we store anything
# that needs to persist across interactions in st.session_state.
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []   # list of (filename, bytes) tuples
if "parsed_chunks" not in st.session_state:
    st.session_state.parsed_chunks = []   # list of chunk dicts (Phase 2+)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {role, content} dicts (Phase 4+)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ManualMan")
    st.caption("Pump Manual RAG Assistant")
    st.divider()

    # API key status — shows green/red so you know your .env is loaded
    st.subheader("API Key Status")
    voyage_ok    = bool(config.VOYAGE_API_KEY)
    anthropic_ok = bool(config.ANTHROPIC_API_KEY)
    st.markdown(
        f"{'✅' if voyage_ok    else '❌'} Voyage AI  \n"
        f"{'✅' if anthropic_ok else '❌'} Anthropic"
    )
    if not voyage_ok or not anthropic_ok:
        st.warning("Add your keys to the `.env` file and restart the app.")
    st.divider()

    # PDF uploader — available across all tabs
    st.subheader("Upload Manuals")
    uploaded_files = st.file_uploader(
        "Drop PDF manuals here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more pump manual PDFs (Goulds, Aurora, Gorman-Rupp, etc.)",
        key="sidebar_uploader",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) queued")
        for f in uploaded_files:
            st.caption(f"  {f.name}  ({f.size / 1024:.0f} KB)")

    st.divider()

    # Phase progress tracker — update this as each phase is completed
    st.subheader("Phase Status")
    st.markdown(
        "- Phase 1 — Scaffold ✅\n"
        "- Phase 2 — Parse & Edit ⏳\n"
        "- Phase 3 — Embed & Index ⏳\n"
        "- Phase 4 — Chat ⏳\n"
        "- Phase 5 — Manage Manuals ⏳"
    )

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_parse, tab_chat, tab_manuals = st.tabs(
    ["📋 Parse & Edit", "💬 Chat", "📚 Manuals"]
)

# ── Tab 1: Parse & Edit ───────────────────────────────────────────────────────
with tab_parse:
    st.header("Parse & Edit Chunks")
    st.caption(
        "Upload a PDF in the sidebar, parse it into text chunks, "
        "edit/delete/tag them, then commit to the knowledge base."
    )

    if not uploaded_files:
        st.info(
            "**Step 1:** Drop one or more PDFs in the sidebar uploader.  \n"
            "The parsed chunks will appear here so you can review and edit them "
            "before anything is embedded."
        )
    else:
        # Phase 2 will replace this placeholder with the real parsing UI
        st.success(
            f"{len(uploaded_files)} PDF(s) ready.  "
            f"Parsing UI coming in Phase 2."
        )
        for f in uploaded_files:
            st.write(f"  {f.name}")

# ── Tab 2: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Chat with Your Manuals")
    st.caption(
        "Ask questions about indexed manuals. Answers will cite the source "
        "PDF and page number, and show any relevant diagrams."
    )
    st.info(
        "Index at least one manual first (Phase 3), then come back here to chat."
    )

# ── Tab 3: Manuals ────────────────────────────────────────────────────────────
with tab_manuals:
    st.header("Indexed Manuals")
    st.caption(
        "View all manuals currently in the knowledge base. "
        "Delete or re-process them here."
    )
    st.info("No manuals indexed yet. Use the Parse & Edit tab to get started.")
