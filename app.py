# app.py — Main entry point. Run with: streamlit run app.py
#
# Each phase adds content to one of the three tabs below. Phases 3-5 are
# still placeholders — the code will be filled in as we progress.

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load API keys from .env before importing config (config reads os.getenv)
load_dotenv()

import config
from utils.pdf_parser import extract_chunks_from_pdf, get_pdf_page_count


def _build_chunks_df(chunks: list[dict]) -> pd.DataFrame:
    """
    Convert the list of chunk dicts from pdf_parser into a pandas DataFrame
    shaped for st.data_editor. Only the columns users should see/edit are included;
    chunk_id and image_paths stay in st.session_state.parsed_chunks for Phase 3.
    """
    rows = []
    for c in chunks:
        rows.append({
            "keep":       c["keep"],
            "source_pdf": c["source_pdf"],
            "page":       c["page_number"],
            "text":       c["text"],
            "tags":       c["tags"],
        })
    df = pd.DataFrame(rows)
    df["keep"]       = df["keep"].astype(bool)
    df["page"]       = df["page"].astype(int)
    df["source_pdf"] = df["source_pdf"].astype(str)
    df["text"]       = df["text"].astype(str)
    df["tags"]       = df["tags"].astype(str)
    return df

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ManualMan — Pump Manual RAG",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure data directories exist ─────────────────────────────────────────────
for d in [config.UPLOADS_DIR, config.IMAGES_DIR, config.CHROMA_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
# Streamlit reruns the whole script on every user interaction, so anything
# that needs to persist between reruns lives in st.session_state.
if "parsed_chunks" not in st.session_state:
    st.session_state.parsed_chunks = []     # raw output from pdf_parser

if "chunks_df" not in st.session_state:
    st.session_state.chunks_df = None       # pandas DataFrame shown in the editor

if "parse_version" not in st.session_state:
    st.session_state.parse_version = 0      # incremented on each re-parse to
                                            # force the data_editor to reset

if "edited_df" not in st.session_state:
    st.session_state.edited_df = None       # latest user-edited DataFrame
                                            # (Phase 3 reads this for embedding)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # {role, content} dicts (Phase 4)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ManualMan")
    st.caption("Pump Manual RAG Assistant")
    st.divider()

    # Quick visual check that API keys loaded from .env
    st.subheader("API Key Status")
    voyage_ok    = bool(config.VOYAGE_API_KEY)
    anthropic_ok = bool(config.ANTHROPIC_API_KEY)
    st.markdown(
        f"{'✅' if voyage_ok    else '❌'} Voyage AI  \n"
        f"{'✅' if anthropic_ok else '❌'} Anthropic"
    )
    if not voyage_ok or not anthropic_ok:
        st.warning("Add your keys to `.env` and restart the app.")
    st.divider()

    st.subheader("Upload Manuals")
    uploaded_files = st.file_uploader(
        "Drop PDF manuals here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more pump manual PDFs",
        key="sidebar_uploader",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) queued")
        for f in uploaded_files:
            st.caption(f"  {f.name}  ({f.size / 1024:.0f} KB)")

    st.divider()
    st.subheader("Phase Status")
    st.markdown(
        "- Phase 1 — Scaffold ✅\n"
        "- Phase 2 — Parse & Edit ✅\n"
        "- Phase 3 — Embed & Index ⏳\n"
        "- Phase 4 — Chat ⏳\n"
        "- Phase 5 — Manage Manuals ⏳"
    )

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_parse, tab_chat, tab_manuals = st.tabs(
    ["📋 Parse & Edit", "💬 Chat", "📚 Manuals"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Parse & Edit
# ══════════════════════════════════════════════════════════════════════════════
with tab_parse:
    st.header("Parse & Edit Chunks")
    st.caption(
        "Upload PDFs → parse them into text chunks → "
        "edit/delete/tag the chunks → commit to the knowledge base."
    )

    if not uploaded_files:
        st.info(
            "**Step 1:** Drop one or more pump manual PDFs in the sidebar uploader, "
            "then click **Parse PDFs** to extract text and images."
        )
    else:
        # ── Parse button ───────────────────────────────────────────────────────
        col_btn, col_info = st.columns([2, 5])
        with col_btn:
            parse_clicked = st.button("🔍 Parse PDFs", type="primary", use_container_width=True)
        with col_info:
            total_pages = 0
            for f in uploaded_files:
                total_pages += get_pdf_page_count(f.getvalue())
            st.caption(
                f"{len(uploaded_files)} PDF(s) · {total_pages} pages total  \n"
                f"Chunk size: ~{config.CHUNK_SIZE} tokens · Overlap: {config.CHUNK_OVERLAP} tokens"
            )

        if parse_clicked:
            all_chunks = []
            progress = st.progress(0, text="Starting…")
            for i, f in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Parsing {f.name}…"
                )
                chunks = extract_chunks_from_pdf(f.getvalue(), f.name)
                all_chunks.extend(chunks)

            progress.progress(1.0, text="Done!")
            progress.empty()

            # Store raw chunks and build the editable DataFrame
            st.session_state.parsed_chunks = all_chunks
            st.session_state.chunks_df = _build_chunks_df(all_chunks)
            st.session_state.parse_version += 1   # forces data_editor to reset
            st.session_state.edited_df = None

            img_count = sum(len(c["image_paths"]) for c in all_chunks)
            st.success(
                f"Extracted **{len(all_chunks)} chunks** from "
                f"**{len(uploaded_files)} PDF(s)** — "
                f"**{img_count} image(s)** saved to `data/extracted_images/`"
            )

        # ── Editable chunk table ───────────────────────────────────────────────
        if st.session_state.chunks_df is not None:
            st.subheader("Chunk Editor")
            st.caption(
                "✏️ **Edit** the text directly in the table (double-click a cell).  \n"
                "🏷️ **Tags** — add comma-separated labels, e.g. `model-number, curve-data`.  \n"
                "☑️ **Keep** — uncheck a row to exclude it from indexing (won't be deleted here).  \n"
                "Scroll right to see all columns."
            )

            edited_df = st.data_editor(
                st.session_state.chunks_df,
                key=f"chunk_editor_{st.session_state.parse_version}",
                column_config={
                    "keep": st.column_config.CheckboxColumn(
                        "Keep?",
                        help="Uncheck to exclude this chunk from the knowledge base",
                        width="small",
                    ),
                    "source_pdf": st.column_config.TextColumn(
                        "Source PDF",
                        disabled=True,
                        width="medium",
                    ),
                    "page": st.column_config.NumberColumn(
                        "Page",
                        disabled=True,
                        width="small",
                        format="%d",
                    ),
                    "text": st.column_config.TextColumn(
                        "Chunk Text",
                        width="large",
                        help="Double-click to edit. Fix OCR errors, remove junk headers, etc.",
                    ),
                    "tags": st.column_config.TextColumn(
                        "Tags",
                        width="medium",
                        help="Comma-separated metadata tags, e.g. model-number, performance-curve",
                    ),
                },
                hide_index=True,
                use_container_width=True,
                height=450,
                num_rows="fixed",   # DECISION: fixed rows — use Keep checkbox to
                                    # exclude chunks rather than deleting rows.
                                    # This prevents accidental data loss.
            )

            # Save the latest edited state so Phase 3's Commit button can read it
            st.session_state.edited_df = edited_df

            # Summary line below the table
            keep_count = int(edited_df["keep"].sum())
            total_count = len(edited_df)
            skipped = total_count - keep_count
            st.caption(
                f"**{keep_count}** of **{total_count}** chunks will be indexed  "
                f"{'· ' + str(skipped) + ' excluded' if skipped else ''}"
            )

            # ── Image gallery ──────────────────────────────────────────────────
            all_image_paths: list[str] = []
            for chunk in st.session_state.parsed_chunks:
                for p in chunk["image_paths"]:
                    if p not in all_image_paths:
                        all_image_paths.append(p)

            if all_image_paths:
                with st.expander(f"📷 Extracted Images ({len(all_image_paths)})", expanded=False):
                    st.caption(
                        "These images were extracted from the PDFs and saved locally. "
                        "They'll be linked to their source chunks when you commit."
                    )
                    # Display in a 4-column grid
                    grid_cols = st.columns(4)
                    for i, img_path in enumerate(all_image_paths):
                        try:
                            p = Path(img_path)
                            grid_cols[i % 4].image(
                                img_path,
                                use_container_width=True,
                                caption=f"{p.parent.name} · {p.name}",
                            )
                        except Exception:
                            grid_cols[i % 4].caption(f"⚠️ Could not display {img_path}")
            else:
                st.caption("No images extracted (PDF may be text-only or images are too small).")

            # ── Commit button (Phase 3 placeholder) ────────────────────────────
            st.divider()
            col_commit, col_commit_info = st.columns([2, 5])
            with col_commit:
                st.button(
                    "💾 Commit to Knowledge Base",
                    type="primary",
                    disabled=True,
                    use_container_width=True,
                    help="Coming in Phase 3 — will embed all 'Keep' chunks via Voyage AI and store them in ChromaDB.",
                )
            with col_commit_info:
                st.caption(f"Will embed **{keep_count} chunk(s)** via Voyage AI → ChromaDB  _(Phase 3)_")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat (Phase 4 placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 Chat with Your Manuals")
    st.caption(
        "Ask questions about indexed manuals. Answers will cite the source "
        "PDF and page number, and show any relevant diagrams."
    )
    st.info("Index at least one manual first (Phase 3), then come back here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Manuals (Phase 5 placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_manuals:
    st.header("📚 Indexed Manuals")
    st.caption("View, delete, or re-process manuals in the knowledge base.")
    st.info("No manuals indexed yet — use Parse & Edit to get started.")


