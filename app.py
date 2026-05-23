# app.py — Main entry point. Run with: streamlit run app.py

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load API keys from .env before importing config (config reads os.getenv)
load_dotenv()

import anthropic

import config
from utils.pdf_parser import extract_chunks_from_pdf, get_pdf_page_count
from utils.embedder import commit_chunks, get_indexed_pdfs, delete_pdf_from_index, get_total_chunk_count
from utils.retriever import retrieve, build_context_prompt

# System prompt sent to Claude on every chat turn.
# DECISION: "ONLY the provided context" prevents hallucination from Claude's
# training data. If you want Claude to supplement with general pump knowledge
# when the manuals don't cover something, remove that constraint here.
_SYSTEM_PROMPT = """\
You are ManualMan, a technical assistant specializing in pump equipment manuals \
(Goulds, Aurora, Gorman-Rupp, and similar manufacturers).

Answer the user's question using ONLY the context chunks provided below from \
indexed pump manuals. Cite every fact with its source number using [Source N] \
inline. If multiple sources support a point, cite all of them.

If the context does not contain enough information to answer the question \
confidently, say so clearly — do not guess or draw on outside knowledge.

Be precise and technical. The user is a sales engineer who needs accurate \
specifications, model numbers, and performance data.\
"""


def _render_sources(chunks: list[dict]) -> None:
    """
    Render a collapsible 'Sources' section below an assistant message.
    Shows each retrieved chunk with its PDF name, page number, similarity
    score, a text preview, and any images extracted from that page.
    """
    if not chunks:
        return

    with st.expander(f"📎 Sources ({len(chunks)} chunks retrieved)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**Source {i}** — `{chunk['source_pdf']}` · "
                f"Page **{chunk['page_number']}** · "
                f"relevance: {chunk['score']:.2f}"
                + (f" · tags: `{chunk['tags']}`" if chunk["tags"] else "")
            )
            # Show a short preview of the chunk text (not the whole thing)
            preview = chunk["text"][:400]
            if len(chunk["text"]) > 400:
                preview += "…"
            st.caption(preview)

            # Show images extracted from the same page, if any exist on disk
            valid_images = [p for p in chunk["image_paths"] if Path(p).exists()]
            if valid_images:
                img_cols = st.columns(min(len(valid_images), 3))
                for j, img_path in enumerate(valid_images):
                    img_cols[j % 3].image(
                        img_path,
                        use_container_width=True,
                        caption=Path(img_path).name,
                    )

            if i < len(chunks):
                st.divider()


def _build_chunks_df(chunks: list[dict]) -> pd.DataFrame:
    """
    Convert chunk dicts from pdf_parser into a DataFrame for st.data_editor.
    chunk_id and image_paths stay in session_state.parsed_chunks — not shown
    in the table but needed when committing.
    """
    rows = [
        {
            "keep":       c["keep"],
            "source_pdf": c["source_pdf"],
            "page":       c["page_number"],
            "text":       c["text"],
            "tags":       c["tags"],
        }
        for c in chunks
    ]
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
if "parsed_chunks"  not in st.session_state: st.session_state.parsed_chunks  = []
if "chunks_df"      not in st.session_state: st.session_state.chunks_df      = None
if "parse_version"  not in st.session_state: st.session_state.parse_version  = 0
if "edited_df"      not in st.session_state: st.session_state.edited_df      = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "last_committed" not in st.session_state: st.session_state.last_committed = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ManualMan")
    st.caption("Pump Manual RAG Assistant")
    st.divider()

    voyage_ok    = bool(config.VOYAGE_API_KEY)
    anthropic_ok = bool(config.ANTHROPIC_API_KEY)
    st.subheader("API Key Status")
    st.markdown(
        f"{'✅' if voyage_ok    else '❌'} Voyage AI  \n"
        f"{'✅' if anthropic_ok else '❌'} Anthropic"
    )
    if not voyage_ok or not anthropic_ok:
        st.warning("Add your keys to `.env` and restart the app.")

    st.divider()
    st.subheader("Knowledge Base")
    total = get_total_chunk_count()
    indexed = get_indexed_pdfs()
    st.metric("Indexed chunks", total)
    if indexed:
        st.caption(f"{len(indexed)} manual(s) in the index")

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
        "- Phase 3 — Embed & Index ✅\n"
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
        "Upload PDFs → parse into chunks → edit/tag → commit to knowledge base."
    )

    if not uploaded_files:
        st.info(
            "**Step 1:** Drop one or more PDFs in the sidebar uploader, "
            "then click **Parse PDFs**."
        )
    else:
        # ── Parse button ───────────────────────────────────────────────────────
        col_btn, col_info = st.columns([2, 5])
        with col_btn:
            parse_clicked = st.button("🔍 Parse PDFs", type="primary", use_container_width=True)
        with col_info:
            total_pages = sum(get_pdf_page_count(f.getvalue()) for f in uploaded_files)
            st.caption(
                f"{len(uploaded_files)} PDF(s) · {total_pages} pages  \n"
                f"Chunk size: ~{config.CHUNK_SIZE} tokens · Overlap: {config.CHUNK_OVERLAP} tokens"
            )

        if parse_clicked:
            all_chunks = []
            progress = st.progress(0, text="Starting…")
            for i, f in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Parsing {f.name}…")
                all_chunks.extend(extract_chunks_from_pdf(f.getvalue(), f.name))
            progress.progress(1.0, text="Done!")
            progress.empty()

            st.session_state.parsed_chunks = all_chunks
            st.session_state.chunks_df     = _build_chunks_df(all_chunks)
            st.session_state.parse_version += 1
            st.session_state.edited_df     = None

            img_count = sum(len(c["image_paths"]) for c in all_chunks)
            st.success(
                f"Extracted **{len(all_chunks)} chunks** from "
                f"**{len(uploaded_files)} PDF(s)** — "
                f"**{img_count} image(s)** saved."
            )

        # ── Editable chunk table ───────────────────────────────────────────────
        if st.session_state.chunks_df is not None:
            st.subheader("Chunk Editor")
            st.caption(
                "✏️ Double-click **Chunk Text** to edit (fix OCR errors, trim junk).  \n"
                "🏷️ **Tags** — comma-separated labels, e.g. `model-number, curve-data`.  \n"
                "☑️ Uncheck **Keep?** to exclude a chunk from indexing."
            )

            edited_df = st.data_editor(
                st.session_state.chunks_df,
                key=f"chunk_editor_{st.session_state.parse_version}",
                column_config={
                    "keep":       st.column_config.CheckboxColumn("Keep?", width="small"),
                    "source_pdf": st.column_config.TextColumn("Source PDF", disabled=True, width="medium"),
                    "page":       st.column_config.NumberColumn("Page", disabled=True, width="small", format="%d"),
                    "text":       st.column_config.TextColumn("Chunk Text", width="large"),
                    "tags":       st.column_config.TextColumn("Tags", width="medium"),
                },
                hide_index=True,
                use_container_width=True,
                height=450,
                num_rows="fixed",
            )

            # Always keep edited_df in session state so the Commit button can
            # access it regardless of which widget triggered the rerun.
            st.session_state.edited_df = edited_df

            keep_count  = int(edited_df["keep"].sum())
            total_count = len(edited_df)
            skipped     = total_count - keep_count
            st.caption(
                f"**{keep_count}** of **{total_count}** chunks will be indexed"
                + (f" · {skipped} excluded" if skipped else "")
            )

            # ── Image gallery ──────────────────────────────────────────────────
            all_image_paths: list[str] = []
            for chunk in st.session_state.parsed_chunks:
                for p in chunk["image_paths"]:
                    if p not in all_image_paths:
                        all_image_paths.append(p)

            if all_image_paths:
                with st.expander(f"📷 Extracted Images ({len(all_image_paths)})", expanded=False):
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
                st.caption("No images extracted (text-only PDF, or all images were below the size threshold).")

            # ── Commit to Knowledge Base ───────────────────────────────────────
            st.divider()
            col_commit, col_info = st.columns([2, 5])

            with col_info:
                if not voyage_ok:
                    st.error("Voyage AI key missing — add it to `.env` and restart.")
                else:
                    st.caption(
                        f"Will embed **{keep_count} chunk(s)** via Voyage AI (`{config.VOYAGE_MODEL}`) "
                        f"and store in ChromaDB.  \n"
                        f"Re-committing the same PDF **replaces** its existing entries."
                    )

            with col_commit:
                commit_clicked = st.button(
                    "💾 Commit to Knowledge Base",
                    type="primary",
                    disabled=(not voyage_ok or keep_count == 0),
                    use_container_width=True,
                    key="commit_btn",
                )

            if commit_clicked:
                if st.session_state.edited_df is None or keep_count == 0:
                    st.warning("No chunks marked Keep — nothing to commit.")
                else:
                    commit_progress = st.progress(0, text="Starting…")

                    def _progress_cb(frac: float, msg: str):
                        commit_progress.progress(frac, text=msg)

                    try:
                        n = commit_chunks(
                            st.session_state.edited_df,
                            st.session_state.parsed_chunks,
                            progress_cb=_progress_cb,
                        )
                        commit_progress.empty()
                        st.session_state.last_committed = n
                        st.success(
                            f"✅ **{n} chunks** committed to the knowledge base.  \n"
                            f"Switch to the **Chat** tab to start asking questions."
                        )
                    except Exception as e:
                        commit_progress.empty()
                        st.error(f"Commit failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("💬 Chat with Your Manuals")

    total_chunks = get_total_chunk_count()
    indexed_pdfs = get_indexed_pdfs()

    if total_chunks == 0:
        st.info(
            "No manuals indexed yet.  \n"
            "Go to **Parse & Edit**, upload a PDF, parse it, and click **Commit to Knowledge Base**."
        )
    elif not anthropic_ok:
        st.error("Anthropic API key missing — add it to `.env` and restart.")
    else:
        st.caption(
            f"Searching **{total_chunks} chunks** across "
            f"**{len(indexed_pdfs)} manual(s)** · "
            f"Top {config.TOP_K} chunks retrieved per query · "
            f"Model: `{config.ANTHROPIC_MODEL}`"
        )

        # ── Clear chat button ──────────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        # ── Render existing chat history ───────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show source citations for assistant messages
                if msg["role"] == "assistant" and msg.get("chunks"):
                    _render_sources(msg["chunks"])

        # ── Chat input ─────────────────────────────────────────────────────────
        if question := st.chat_input("Ask about your pump manuals…"):

            # Show the user's message immediately
            st.session_state.chat_history.append(
                {"role": "user", "content": question, "chunks": None}
            )
            with st.chat_message("user"):
                st.markdown(question)

            # Retrieve relevant chunks, then stream Claude's answer
            with st.chat_message("assistant"):
                with st.spinner("Searching manuals…"):
                    chunks = retrieve(question)

                if not chunks:
                    answer = (
                        "I couldn't find any relevant content in the indexed manuals "
                        "for that question. Try rephrasing, or check that the right "
                        "manual is indexed."
                    )
                    st.markdown(answer)
                else:
                    context = build_context_prompt(chunks)
                    user_message = (
                        f"Context from pump manuals:\n\n{context}"
                        f"\n\n---\n\nQuestion: {question}"
                    )

                    # Stream the response so the user sees words appear in real time
                    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

                    def _stream_response():
                        with client.messages.stream(
                            model=config.ANTHROPIC_MODEL,
                            max_tokens=config.MAX_TOKENS,
                            system=_SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": user_message}],
                        ) as stream:
                            for text in stream.text_stream:
                                yield text

                    # st.write_stream displays tokens as they arrive and returns
                    # the full completed string when done
                    answer = st.write_stream(_stream_response())

                    _render_sources(chunks)

            # Save to history so citations persist when the user scrolls up
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "chunks": chunks}
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Manuals (basic version, full management in Phase 5)
# ══════════════════════════════════════════════════════════════════════════════
with tab_manuals:
    st.header("📚 Indexed Manuals")
    st.caption("Manuals currently in the knowledge base.")

    indexed_pdfs = get_indexed_pdfs()
    if not indexed_pdfs:
        st.info("No manuals indexed yet — use Parse & Edit to get started.")
    else:
        for entry in indexed_pdfs:
            col_name, col_count, col_del = st.columns([4, 2, 1])
            col_name.write(f"📄 {entry['source_pdf']}")
            col_count.caption(f"{entry['chunk_count']} chunks")
            if col_del.button("🗑️", key=f"del_{entry['source_pdf']}", help="Remove from index"):
                n = delete_pdf_from_index(entry["source_pdf"])
                st.success(f"Removed {n} chunks for **{entry['source_pdf']}**.")
                st.rerun()

        st.divider()
        st.caption(
            f"Total: **{get_total_chunk_count()} chunks** across "
            f"**{len(indexed_pdfs)} manual(s)**  \n"
            "Full management features (re-process, settings) coming in Phase 5."
        )
