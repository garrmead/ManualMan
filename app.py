# app.py — Main entry point. Run with: streamlit run app.py

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

import anthropic

import config
from utils.pdf_parser import extract_chunks_from_pdf, get_pdf_page_count
from utils.embedder import commit_chunks, get_indexed_pdfs, delete_pdf_from_index, get_total_chunk_count
from utils.retriever import retrieve, build_context_prompt, build_chat_messages

_SYSTEM_PROMPT = """\
You are ManualMan, a technical assistant for pump equipment manuals \
(Goulds, Aurora, Gorman-Rupp, and similar manufacturers).

Rules:
1. Answer using ONLY the provided context. Do not draw on outside knowledge.
2. Be concise and direct. Lead with the answer — skip preamble like \
   "Based on the provided context..." or restating the question.
3. Cite sources inline as [Source N]. If multiple sources back a point, cite all.
4. Some sources are IMAGE chunks (performance curves, drawings, etc.). \
   Reference them briefly, e.g. "See [Source 2]." The image displays automatically — \
   do not describe it in detail.
5. If the context lacks enough information, say so in one sentence. Do not guess.\
"""

# ── Manual metadata fields ────────────────────────────────────────────────────
_DOC_TYPES = [
    "Installation Manual",
    "Operation Manual",
    "Parts Manual",
    "Selection Guide",
    "Technical Data Sheet",
    "Other",
]


# ── Helper: render image chunks inline in chat ────────────────────────────────
def _render_inline_images(chunks: list[dict]) -> None:
    from utils.vision import IMAGE_TYPE_LABELS
    image_chunks = [
        c for c in chunks
        if c.get("chunk_type") == "image" and Path(c.get("image_path", "")).exists()
    ]
    for chunk in image_chunks:
        label    = IMAGE_TYPE_LABELS.get(chunk.get("image_type", ""), "Diagram")
        citation = _format_citation(chunk)
        st.markdown(f"**{label}** — {citation}")
        _, img_col, _ = st.columns([1, 5, 1])
        img_col.image(chunk["image_path"], use_container_width=True)


# ── Helper: render text sources expander ─────────────────────────────────────
def _render_sources(chunks: list[dict]) -> None:
    text_chunks = [c for c in chunks if c.get("chunk_type") != "image"]
    if not text_chunks:
        return
    with st.expander(f"📎 Text sources ({len(text_chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(text_chunks, 1):
            st.markdown(
                f"**{i}.** {_format_citation(chunk)} · relevance: {chunk['score']:.2f}"
                + (f" · `{chunk['tags']}`" if chunk.get("tags") else "")
            )
            preview = chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else "")
            st.caption(preview)
            if i < len(text_chunks):
                st.divider()


def _format_citation(chunk: dict) -> str:
    """Build a rich citation string from a chunk, including manual metadata."""
    parts = [f"`{chunk['source_pdf']}`"]
    meta_parts = []
    if chunk.get("manufacturer"):  meta_parts.append(chunk["manufacturer"])
    if chunk.get("product_line"):  meta_parts.append(chunk["product_line"])
    if chunk.get("doc_type"):      meta_parts.append(chunk["doc_type"])
    if chunk.get("revision"):      meta_parts.append(chunk["revision"])
    if meta_parts:
        parts.append(f"({' · '.join(meta_parts)})")
    parts.append(f"Page **{chunk['page_number']}**")
    return " · ".join(parts)


# ── Helper: build DataFrame for chunk editor ──────────────────────────────────
def _build_chunks_df(chunks: list[dict]) -> pd.DataFrame:
    rows = [
        {"keep": c["keep"], "source_pdf": c["source_pdf"],
         "page": c["page_number"], "text": c["text"], "tags": c["tags"]}
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

for d in [config.UPLOADS_DIR, config.IMAGES_DIR, config.CHROMA_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "parsed_chunks"  not in st.session_state: st.session_state.parsed_chunks  = []
if "chunks_df"      not in st.session_state: st.session_state.chunks_df      = None
if "parse_version"  not in st.session_state: st.session_state.parse_version  = 0
if "edited_df"      not in st.session_state: st.session_state.edited_df      = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "last_committed" not in st.session_state: st.session_state.last_committed = 0
if "pdf_metadata"   not in st.session_state: st.session_state.pdf_metadata   = {}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
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
    total   = get_total_chunk_count()
    indexed = get_indexed_pdfs()
    st.metric("Indexed chunks", total)
    if indexed:
        st.caption(f"{len(indexed)} manual(s) in the index")

    st.divider()
    st.subheader("Chat Settings")

    # Relevance threshold slider — adjustable at runtime without editing config.py.
    # DECISION: default 0.45. Raise if answers feel off-topic (fewer but better
    # chunks reach Claude). Lower if Claude says "not found" on things you know
    # are in the manual.
    relevance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.MIN_RELEVANCE_SCORE,
        step=0.05,
        help=(
            "Chunks below this similarity score are excluded before Claude sees them. "
            "Raise toward 0.6 for more precise answers; lower toward 0.3 if valid "
            "content is being missed."
        ),
    )

    history_turns = st.select_slider(
        "Conversation memory (turns)",
        options=[0, 1, 2, 3, 5],
        value=config.MAX_HISTORY_TURNS,
        help=(
            "How many past Q&A pairs are sent to Claude for follow-up context. "
            "0 = no memory (each question is independent). Higher = better follow-ups "
            "but more tokens per call."
        ),
    )

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
        "- Phase 4 — Chat ✅\n"
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
        "Upload PDFs → fill in metadata → parse into chunks → "
        "edit/tag → commit to knowledge base."
    )

    if not uploaded_files:
        st.info(
            "**Step 1:** Drop one or more PDFs in the sidebar uploader, "
            "fill in the metadata fields, then click **Parse PDFs**."
        )
    else:
        # ── Manual metadata form ───────────────────────────────────────────────
        st.subheader("Manual Metadata")
        st.caption(
            "Optional but recommended — stored with every chunk and shown in citations.  \n"
            "Fill in once per PDF before parsing."
        )
        for f in uploaded_files:
            with st.expander(f"📄 {f.name}", expanded=True):
                c1, c2 = st.columns(2)
                manufacturer = c1.text_input(
                    "Manufacturer", key=f"meta_{f.name}_mfr",
                    placeholder="e.g. Goulds Pumps",
                )
                product_line = c2.text_input(
                    "Product Line / Model", key=f"meta_{f.name}_prod",
                    placeholder="e.g. 3196",
                )
                c3, c4 = st.columns(2)
                doc_type = c3.selectbox(
                    "Document Type", _DOC_TYPES, key=f"meta_{f.name}_dtype",
                )
                revision = c4.text_input(
                    "Revision / Date", key=f"meta_{f.name}_rev",
                    placeholder="e.g. Rev. 2023-Q1",
                )

        # ── Parse button ───────────────────────────────────────────────────────
        st.divider()
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
            # Snapshot metadata widget values into session state at parse time
            st.session_state.pdf_metadata = {
                f.name: {
                    "manufacturer": st.session_state.get(f"meta_{f.name}_mfr", ""),
                    "product_line": st.session_state.get(f"meta_{f.name}_prod", ""),
                    "doc_type":     st.session_state.get(f"meta_{f.name}_dtype", ""),
                    "revision":     st.session_state.get(f"meta_{f.name}_rev", ""),
                }
                for f in uploaded_files
            }

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
                f"**{len(uploaded_files)} PDF(s)** — **{img_count} image(s)** saved."
            )

        # ── Editable chunk table ───────────────────────────────────────────────
        if st.session_state.chunks_df is not None:
            st.subheader("Chunk Editor")
            st.caption(
                "✏️ Double-click **Chunk Text** to edit.  \n"
                "🏷️ **Tags** — comma-separated, e.g. `model-number, curve-data`.  \n"
                "☑️ Uncheck **Keep?** to exclude a chunk."
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
                            grid_cols[i % 4].image(img_path, use_container_width=True,
                                                   caption=f"{p.parent.name} · {p.name}")
                        except Exception:
                            grid_cols[i % 4].caption(f"⚠️ Could not display {img_path}")
            else:
                st.caption("No images extracted (text-only PDF or all images below size threshold).")

            # ── Commit to Knowledge Base ───────────────────────────────────────
            st.divider()
            col_commit, col_info = st.columns([2, 5])

            with col_info:
                if not voyage_ok:
                    st.error("Voyage AI key missing — add it to `.env` and restart.")
                else:
                    img_count = len({p for c in st.session_state.parsed_chunks for p in c["image_paths"]})
                    st.caption(
                        f"Will embed **{keep_count} text chunk(s)** + classify & embed "
                        f"**{img_count} image(s)**.  \n"
                        f"Re-committing replaces existing entries for the same PDF."
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
                        counts = commit_chunks(
                            st.session_state.edited_df,
                            st.session_state.parsed_chunks,
                            pdf_metadata=st.session_state.pdf_metadata,
                            progress_cb=_progress_cb,
                        )
                        commit_progress.empty()
                        st.session_state.last_committed = counts["text_chunks"]
                        st.success(
                            f"✅ **{counts['text_chunks']} text chunks** + "
                            f"**{counts['image_chunks']} image chunks** committed.  \n"
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
            "Go to **Parse & Edit**, upload a PDF, parse it, and click **Commit**."
        )
    elif not anthropic_ok:
        st.error("Anthropic API key missing — add it to `.env` and restart.")
    else:
        st.caption(
            f"Searching **{total_chunks} chunks** across **{len(indexed_pdfs)} manual(s)** · "
            f"Threshold: **{relevance_threshold:.2f}** · "
            f"Memory: **{history_turns} turn(s)** · "
            f"Model: `{config.ANTHROPIC_MODEL}`"
        )

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        # ── Render chat history ────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("chunks"):
                    _render_inline_images(msg["chunks"])
                    _render_sources(msg["chunks"])

        # ── Chat input ─────────────────────────────────────────────────────────
        if question := st.chat_input("Ask about your pump manuals…"):

            st.session_state.chat_history.append(
                {"role": "user", "content": question, "chunks": None}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # For follow-up questions, combine with the last user question
                # to give retrieval more context (e.g. "what about stainless?"
                # needs the previous topic to retrieve correctly).
                past_user_qs = [
                    m["content"] for m in st.session_state.chat_history
                    if m["role"] == "user" and m["content"] != question
                ]
                retrieval_query = question
                if past_user_qs and len(question.split()) < 12:
                    # Short follow-up — prepend the previous question for context
                    retrieval_query = f"{past_user_qs[-1]} {question}"

                with st.spinner("Searching manuals…"):
                    chunks = retrieve(
                        retrieval_query,
                        min_score=relevance_threshold,
                    )

                if not chunks:
                    answer = (
                        "I couldn't find relevant content above the current relevance "
                        f"threshold ({relevance_threshold:.2f}). Try lowering the threshold "
                        "in the sidebar, rephrasing, or checking that the right manual is indexed."
                    )
                    st.markdown(answer)
                else:
                    context  = build_context_prompt(chunks)
                    messages = build_chat_messages(
                        question,
                        context,
                        # Pass history minus the question we just added
                        st.session_state.chat_history[:-1],
                    )

                    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

                    def _stream_response():
                        with client.messages.stream(
                            model=config.ANTHROPIC_MODEL,
                            max_tokens=config.MAX_TOKENS,
                            system=_SYSTEM_PROMPT,
                            messages=messages,
                        ) as stream:
                            for text in stream.text_stream:
                                yield text

                    answer = st.write_stream(_stream_response())

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "chunks": chunks}
            )
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Manuals
# ══════════════════════════════════════════════════════════════════════════════
with tab_manuals:
    st.header("📚 Indexed Manuals")
    st.caption("Manuals currently in the knowledge base.")

    indexed_pdfs = get_indexed_pdfs()
    if not indexed_pdfs:
        st.info("No manuals indexed yet — use Parse & Edit to get started.")
    else:
        for entry in indexed_pdfs:
            col_name, col_meta, col_count, col_del = st.columns([3, 3, 2, 1])
            col_name.write(f"📄 {entry['source_pdf']}")
            # Show manual metadata if present
            meta_str = " · ".join(filter(None, [
                entry.get("manufacturer", ""),
                entry.get("product_line", ""),
                entry.get("doc_type", ""),
                entry.get("revision", ""),
            ]))
            col_meta.caption(meta_str or "—")
            col_count.caption(f"{entry['text_chunks']} text · {entry['image_chunks']} img")
            if col_del.button("🗑️", key=f"del_{entry['source_pdf']}", help="Remove from index"):
                n = delete_pdf_from_index(entry["source_pdf"])
                st.success(f"Removed {n} chunks for **{entry['source_pdf']}**.")
                st.rerun()

        st.divider()
        st.caption(
            f"Total: **{get_total_chunk_count()} chunks** across "
            f"**{len(indexed_pdfs)} manual(s)**"
        )
