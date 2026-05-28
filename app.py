# app.py — Main entry point. Run with: streamlit run app.py

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

import re

import anthropic

import config
from utils.pdf_parser import extract_chunks_from_pdf, get_pdf_page_count
from utils.embedder import commit_chunks, get_indexed_pdfs, delete_pdf_from_index, get_total_chunk_count
from utils.retriever import retrieve, build_context_prompt, build_chat_messages

# ── Design system CSS ─────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap');

/* ── Design tokens ── */
:root {
  --bg:             #111113;
  --bg-2:           #1a1a1e;
  --surface:        #1f1f23;
  --surface-2:      #252529;
  --fg:             #f5f5f7;
  --fg-2:           #a1a1aa;
  --fg-3:           #6b6b74;
  --border:         #2a2a2f;
  --border-strong:  #3a3a40;
  --accent:         #ff7849;
  --accent-dim:     rgba(255,120,73,0.15);
  --accent-glow:    rgba(255,120,73,0.08);
  --radius-sm:      4px;
  --radius-md:      6px;
  --radius-lg:      8px;
  --radius-pill:    999px;
  --shadow-card:    0 1px 0 rgba(0,0,0,0.08), 2px 2px 0 var(--border);
  --shadow-float:   0 8px 24px rgba(0,0,0,0.24), 0 2px 6px rgba(0,0,0,0.16);
}

/* ── Base typography ── */
html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--fg) !important;
}

/* Headlines */
h1, h2, h3 {
  font-family: 'IBM Plex Sans', sans-serif !important;
  letter-spacing: -0.02em !important;
  color: var(--fg) !important;
}
h1 { font-size: 1.6rem !important; font-weight: 600 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; }

/* Mono labels */
.stCaption, small, caption, [data-testid="stCaptionContainer"] p {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.04em !important;
  color: var(--fg-3) !important;
}

/* ── App background ── */
.stApp {
  background-color: var(--bg) !important;
}
.stApp > header {
  background-color: var(--bg) !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: var(--bg-2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
  font-size: 13px !important;
  color: var(--fg-2) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 10px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--fg-3) !important;
  margin-top: 1.4rem !important;
  margin-bottom: 0.4rem !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border) !important;
  margin: 0.75rem 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background-color: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  color: var(--fg-3) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 16px !important;
  text-transform: uppercase !important;
  transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
  color: var(--fg) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 1.5rem !important;
}

/* ── Primary buttons ── */
.stButton > button[kind="primary"],
.stButton > button[data-testid*="primary"] {
  background-color: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 9px 16px !important;
  transition: all 0.15s ease !important;
  box-shadow: none !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: #ff6535 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(255,120,73,0.30) !important;
}

/* Secondary buttons */
.stButton > button[kind="secondary"],
.stButton > button:not([kind]) {
  background-color: var(--surface) !important;
  color: var(--fg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 13px !important;
  transition: all 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind]):hover {
  background-color: var(--surface-2) !important;
  border-color: var(--border-strong) !important;
  color: var(--fg) !important;
}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select,
[data-testid="stTextInput"] input,
[data-baseweb="select"] [data-baseweb="input"],
[data-baseweb="popover"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--fg) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 13px !important;
  transition: border-color 0.15s ease !important;
}
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-dim) !important;
}

/* Selectbox */
[data-baseweb="select"] {
  background-color: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--accent) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
  background-color: var(--accent) !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.6rem !important;
  color: var(--fg) !important;
  font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--fg-3) !important;
}

/* ── Alerts / info / success ── */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  font-size: 13px !important;
}
.stSuccess { border-left: 3px solid #4ade80 !important; }
.stError   { border-left: 3px solid #f87171 !important; }
.stWarning { border-left: 3px solid #fbbf24 !important; }
.stInfo    { border-left: 3px solid var(--accent) !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
  background-color: var(--accent) !important;
  border-radius: var(--radius-pill) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  background-color: var(--bg-2) !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stExpander"] summary {
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--fg-2) !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--fg) !important;
}

/* ── Data editor ── */
[data-testid="stDataEditor"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}
/* Force the glide-data-grid canvas wrapper to use dark bg so
   Streamlit's theme injection picks up the right cell colors */
[data-testid="stDataEditor"] > div,
.dvn-scroller,
.dvn-scroller > div {
  background-color: var(--bg-2) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  border-radius: var(--radius-lg) !important;
  border: 1px solid var(--border) !important;
  background-color: var(--bg-2) !important;
  padding: 16px !important;
  margin-bottom: 10px !important;
  box-shadow: var(--shadow-card) !important;
}
/* User messages — slightly different tint */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background-color: var(--surface) !important;
  border-color: var(--border-strong) !important;
}
/* Avatar */
[data-testid="chatAvatarIcon-assistant"] {
  background-color: #0a0a0b !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
[data-testid="chatAvatarIcon-user"] {
  background: linear-gradient(135deg, var(--accent) 0%, #ff9a76 100%) !important;
  border-radius: var(--radius-pill) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background-color: var(--surface) !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 14px !important;
  color: var(--fg) !important;
  background: transparent !important;
}

/* ── Dividers ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1rem 0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--radius-md) !important;
  background-color: var(--surface) !important;
  transition: border-color 0.15s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background-color: var(--accent-glow) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
  color: var(--accent) !important;
}

/* ── Checkbox (in data editor keep column) ── */
input[type="checkbox"]:checked {
  accent-color: var(--accent) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow-float) !important;
}

/* ── Code blocks ── */
code, pre {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Tooltip / help icon ── */
[data-testid="stTooltipIcon"] svg { color: var(--fg-3) !important; }

/* ── Accent inline text ── */
.mm-accent { color: var(--accent) !important; font-style: italic; }
.mm-mono   { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.06em; }
.mm-label  {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-3);
}
</style>
"""

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
5. If the context lacks enough information, say so in one sentence. Do not guess.
6. After your answer, on a new line write exactly: \
   FOLLOW-UPS: <question 1> | <question 2> | <question 3> \
   These are short suggested follow-up questions (max 10 words each) the user might ask next. \
   If the answer was "I don't know", skip the FOLLOW-UPS line.\
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


def _cited_source_indices(answer_text: str) -> set[int]:
    """Return 1-based source indices Claude cited in its answer."""
    return {int(n) for n in re.findall(r"\[Source\s+(\d+)\]", answer_text, re.IGNORECASE)}


def _parse_follow_ups(raw_answer: str) -> tuple[str, list[str]]:
    """
    Strip the FOLLOW-UPS line Claude appends and return (clean_answer, follow_up_list).
    If no FOLLOW-UPS line is present, follow_up_list is empty.
    """
    match = re.search(r"\n*FOLLOW-UPS?:\s*(.+?)$", raw_answer, re.IGNORECASE | re.DOTALL)
    if not match:
        return raw_answer, []
    clean = raw_answer[: match.start()].strip()
    parts = [q.strip().lstrip("•*-").strip() for q in re.split(r"\||\n", match.group(1))]
    follow_ups = [q for q in parts if q][:3]
    return clean, follow_ups


# ── Helper: render image chunks inline in chat ────────────────────────────────
def _render_inline_images(chunks: list[dict], answer_text: str = "") -> None:
    """
    Display image chunks inline in the chat message.

    If answer_text is provided, only images whose source number Claude
    actually cited are shown. This prevents all retrieved curves from
    appearing when only one was relevant to the answer.
    """
    from utils.vision import IMAGE_TYPE_LABELS

    cited = _cited_source_indices(answer_text) if answer_text else set()

    for source_num, chunk in enumerate(chunks, start=1):
        if chunk.get("chunk_type") != "image":
            continue
        if not Path(chunk.get("image_path", "")).exists():
            continue
        # If we know which sources were cited, skip un-cited image chunks.
        # If answer_text is empty (e.g. history render without stored text),
        # fall back to showing all image chunks.
        if cited and source_num not in cited:
            continue

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
    with st.expander(f"Sources · {len(text_chunks)} text chunk(s)", expanded=False):
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
        {
            "keep":         c["keep"],
            "type":         "📊 Table" if c.get("content_type") == "table" else "📝 Text",
            "source_pdf":   c["source_pdf"],
            "page":         c["page_number"],
            "text":         c["text"],
            "tags":         c["tags"],
        }
        for c in chunks
    ]
    df = pd.DataFrame(rows)
    df["keep"]       = df["keep"].astype(bool)
    df["type"]       = df["type"].astype(str)
    df["page"]       = df["page"].astype(int)
    df["source_pdf"] = df["source_pdf"].astype(str)
    df["text"]       = df["text"].astype(str)
    df["tags"]       = df["tags"].astype(str)
    return df


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ManualMan",
    page_icon="🟧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(_CSS, unsafe_allow_html=True)

for d in [config.UPLOADS_DIR, config.IMAGES_DIR, config.CHROMA_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "parsed_chunks"      not in st.session_state: st.session_state.parsed_chunks      = []
if "chunks_df"          not in st.session_state: st.session_state.chunks_df          = None
if "parse_version"      not in st.session_state: st.session_state.parse_version      = 0
if "edited_df"          not in st.session_state: st.session_state.edited_df          = None
if "chat_history"       not in st.session_state: st.session_state.chat_history       = []
if "show_img_gallery"   not in st.session_state: st.session_state.show_img_gallery   = True
if "last_committed"     not in st.session_state: st.session_state.last_committed     = 0
if "pdf_metadata"       not in st.session_state: st.session_state.pdf_metadata       = {}
if "pending_question"   not in st.session_state: st.session_state.pending_question   = ""
if "excluded_images"    not in st.session_state: st.session_state.excluded_images    = set()

# ── API key guard — fail loudly before rendering anything else ────────────────
_voyage_ok    = bool(config.VOYAGE_API_KEY)
_anthropic_ok = bool(config.ANTHROPIC_API_KEY)
if not _voyage_ok or not _anthropic_ok:
    missing = []
    if not _voyage_ok:    missing.append("VOYAGE_API_KEY")
    if not _anthropic_ok: missing.append("ANTHROPIC_API_KEY")
    st.error(
        f"**Missing API keys:** {', '.join(missing)}  \n"
        "Add them to your `.env` file and restart the app.  \n"
        "```\nVOYAGE_API_KEY=your-key-here\nANTHROPIC_API_KEY=your-key-here\n```"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;padding:4px 0 16px;">
          <div style="
            position:relative;width:28px;height:28px;
            background:#0a0a0b;border-radius:5px;
            display:flex;align-items:center;justify-content:center;
            flex-shrink:0;border:1px solid #2a2a2f;
          ">
            <span style="
              font-family:'IBM Plex Sans',sans-serif;
              font-weight:700;font-size:11px;
              color:#f5f5f7;letter-spacing:-0.04em;
            ">MM</span>
            <div style="
              position:absolute;top:3px;right:3px;
              width:5px;height:5px;border-radius:50%;
              background:#ff7849;
            "></div>
          </div>
          <div>
            <div style="
              font-family:'IBM Plex Sans',sans-serif;
              font-size:14px;font-weight:600;
              color:#f5f5f7;letter-spacing:-0.01em;line-height:1;
            ">ManualMan</div>
            <div style="
              font-family:'IBM Plex Mono',monospace;
              font-size:9px;color:#6b6b74;
              text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;
            ">Pump Manual RAG</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API status ─────────────────────────────────────────────────────────────
    st.subheader("API Status")
    st.markdown(
        f"<div style='font-size:12px;line-height:1.8;'>"
        f"<span style='color:#4ade80;'>●</span> <span style='color:#a1a1aa;'>Voyage AI</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:#4ade80;'>●</span> <span style='color:#a1a1aa;'>Anthropic</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Knowledge base ─────────────────────────────────────────────────────────
    st.subheader("Knowledge Base")
    total   = get_total_chunk_count()
    indexed = get_indexed_pdfs()
    st.metric("Indexed chunks", total)
    if indexed:
        st.caption(f"{len(indexed)} manual(s) indexed")

    # ── Chat settings ──────────────────────────────────────────────────────────
    st.subheader("Chat Settings")
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
        "Conversation memory",
        options=[0, 1, 2, 3, 5],
        value=config.MAX_HISTORY_TURNS,
        help=(
            "How many past Q&A pairs are sent to Claude for follow-up context. "
            "0 = no memory. Higher = better follow-ups but more tokens per call."
        ),
    )

    # ── Upload manuals ─────────────────────────────────────────────────────────
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
            st.caption(f"{f.name}  ({f.size / 1024:.0f} KB)")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_parse, tab_chat, tab_manuals = st.tabs(
    ["Parse & Edit", "Chat", "Manuals"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Parse & Edit
# ══════════════════════════════════════════════════════════════════════════════
with tab_parse:
    st.markdown(
        "<div class='mm-label' style='margin-bottom:4px;'>Step 1 of 2</div>"
        "<h2 style='margin-top:0;'>Parse & Edit Chunks</h2>"
        "<p style='color:#a1a1aa;font-size:13px;margin-bottom:1.5rem;'>"
        "Upload PDFs → fill in metadata → parse into chunks → edit/tag → commit to knowledge base."
        "</p>",
        unsafe_allow_html=True,
    )

    if not uploaded_files:
        st.info(
            "Drop one or more PDFs in the sidebar uploader, "
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
            st.session_state.parse_version  += 1
            st.session_state.edited_df      = None
            st.session_state.excluded_images = set()

            img_count = sum(len(c["image_paths"]) for c in all_chunks)
            st.success(
                f"Extracted **{len(all_chunks)} chunks** from "
                f"**{len(uploaded_files)} PDF(s)** — **{img_count} image(s)** saved."
            )

        # ── Editable chunk table ───────────────────────────────────────────────
        if st.session_state.chunks_df is not None:
            n_chunks = len(st.session_state.chunks_df)
            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:6px;'>"
                f"<span style='font-size:16px;font-weight:600;'>Chunk Editor</span>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#6b6b74;'>"
                f"{n_chunks} chunk(s) extracted</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if n_chunks == 0:
                st.warning(
                    "No text chunks were extracted from this PDF. "
                    "This usually means the PDF is **scanned** (image-only pages with no text layer). "
                    "OCR would be needed to extract text from scanned documents. "
                    "However, any images (curves, drawings) extracted above can still be committed."
                )
            else:
                st.caption(
                    "Double-click **Chunk Text** to edit · "
                    "**Tags** — comma-separated, e.g. `model-number, curve-data` · "
                    "Uncheck **Keep?** to exclude a chunk."
                )

            if n_chunks > 0:
                edited_df = st.data_editor(
                    st.session_state.chunks_df,
                    key=f"chunk_editor_{st.session_state.parse_version}",
                    column_config={
                        "keep":       st.column_config.CheckboxColumn("Keep?", width="small"),
                        "type":       st.column_config.TextColumn("Type", disabled=True, width="small"),
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
            else:
                # No text chunks — still allow committing images only
                st.session_state.edited_df = st.session_state.chunks_df
                keep_count = 0

            # ── Image gallery with exclusion checkboxes ────────────────────────
            all_image_paths: list[str] = []
            for chunk in st.session_state.parsed_chunks:
                for p in chunk["image_paths"]:
                    if p not in all_image_paths:
                        all_image_paths.append(p)

            if all_image_paths:
                excluded_count = len(st.session_state.excluded_images & set(all_image_paths))

                # Toggle button — stays open across reruns via session state
                hdr_col, toggle_col = st.columns([6, 1])
                hdr_col.markdown(
                    f"<div style='font-size:13px;font-weight:600;padding-top:6px;'>"
                    f"Extracted Images &nbsp;"
                    f"<span style='color:#6b6b74;font-weight:400;font-family:IBM Plex Mono,monospace;font-size:11px;'>"
                    f"{len(all_image_paths)} total"
                    + (f" · <span style='color:#ff7849;'>{excluded_count} excluded</span>" if excluded_count else "")
                    + "</span></div>",
                    unsafe_allow_html=True,
                )
                if toggle_col.button(
                    "Hide ▲" if st.session_state.show_img_gallery else "Show ▼",
                    key="toggle_gallery",
                    use_container_width=True,
                ):
                    st.session_state.show_img_gallery = not st.session_state.show_img_gallery
                    st.rerun()

                if st.session_state.show_img_gallery:
                    st.caption(
                        "Uncheck images to exclude from indexing. "
                        "Solid-color or blurry images are usually PDF background fills — safe to exclude."
                    )
                    sel_col1, sel_col2, _ = st.columns([1, 1, 5])
                    if sel_col1.button("Exclude all", key="excl_all", use_container_width=True):
                        st.session_state.excluded_images = set(all_image_paths)
                        st.rerun()
                    if sel_col2.button("Include all", key="incl_all", use_container_width=True):
                        st.session_state.excluded_images = set()
                        st.rerun()

                    st.divider()
                    grid_cols = st.columns(4)
                    for i, img_path in enumerate(all_image_paths):
                        col = grid_cols[i % 4]
                        try:
                            p = Path(img_path)
                            col.image(img_path, use_container_width=True)
                            included = col.checkbox(
                                "Include",
                                value=(img_path not in st.session_state.excluded_images),
                                key=f"img_include_{i}",
                            )
                            col.caption(p.name)
                            if included:
                                st.session_state.excluded_images.discard(img_path)
                            else:
                                st.session_state.excluded_images.add(img_path)
                        except Exception:
                            col.caption(f"⚠️ {img_path}")
            else:
                st.caption("No images extracted (text-only PDF or all images below size threshold).")

            # ── Commit to Knowledge Base ───────────────────────────────────────
            st.divider()
            col_commit, col_info = st.columns([2, 5])

            with col_info:
                if not _voyage_ok:
                    st.error("Voyage AI key missing — add it to `.env` and restart.")
                else:
                    all_imgs  = {p for c in st.session_state.parsed_chunks for p in c["image_paths"]}
                    incl_imgs = all_imgs - st.session_state.excluded_images
                    st.caption(
                        f"Will embed **{keep_count} text chunk(s)** + classify & embed "
                        f"**{len(incl_imgs)} image(s)**"
                        + (f" ({len(st.session_state.excluded_images)} excluded)" if st.session_state.excluded_images else "")
                        + ".  \nRe-committing replaces existing entries for the same PDF."
                    )

            with col_commit:
                commit_clicked = st.button(
                    "💾 Commit to Knowledge Base",
                    type="primary",
                    disabled=(not _voyage_ok or keep_count == 0),
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
                            excluded_images=st.session_state.excluded_images,
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

# Suggested starter prompts shown when the knowledge base is populated
_SUGGESTED_PROMPTS = [
    "Show me the performance curve",
    "What are the installation torque specs?",
    "What materials are available for wetted parts?",
    "Show me the dimensional drawing",
    "What is the max operating pressure?",
    "What fault codes are covered in this manual?",
]

with tab_chat:
    st.markdown(
        "<h2 style='margin-top:0;margin-bottom:0.25rem;'>Chat</h2>",
        unsafe_allow_html=True,
    )

    total_chunks = get_total_chunk_count()
    indexed_pdfs = get_indexed_pdfs()

    if total_chunks == 0:
        st.markdown(
            "<div style='text-align:center;padding:48px 0;'>"
            "<div style='font-size:2rem;margin-bottom:12px;'>📂</div>"
            "<div style='font-size:15px;font-weight:600;color:#f5f5f7;margin-bottom:6px;'>No manuals indexed yet</div>"
            "<div style='font-size:13px;color:#a1a1aa;'>Go to <b>Parse &amp; Edit</b>, upload a PDF, and click <b>Commit to Knowledge Base</b>.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Status bar ─────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='display:flex;gap:16px;align-items:center;"
            f"padding:8px 12px;background:#1a1a1e;border:1px solid #2a2a2f;"
            f"border-radius:6px;margin-bottom:1.25rem;'>"
            f"<span class='mm-mono' style='color:#6b6b74;'>{total_chunks} chunks</span>"
            f"<span style='color:#2a2a2f;'>|</span>"
            f"<span class='mm-mono' style='color:#6b6b74;'>{len(indexed_pdfs)} manual(s)</span>"
            f"<span style='color:#2a2a2f;'>|</span>"
            f"<span class='mm-mono' style='color:#6b6b74;'>threshold {relevance_threshold:.2f}</span>"
            f"<span style='color:#2a2a2f;'>|</span>"
            f"<span class='mm-mono' style='color:#6b6b74;'>memory {history_turns}T</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Clear chat button ───────────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.pending_question = ""
                st.rerun()

        # ── Suggested prompts (shown only when chat is empty) ──────────────────
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='margin-bottom:12px;'>"
                "<span class='mm-label'>Try asking</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            prompt_cols = st.columns(3)
            for i, prompt in enumerate(_SUGGESTED_PROMPTS):
                if prompt_cols[i % 3].button(
                    prompt,
                    key=f"suggested_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = prompt
                    st.rerun()

        # ── Render chat history ────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("chunks"):
                    _render_inline_images(msg["chunks"], answer_text=msg["content"])
                    _render_sources(msg["chunks"])
                # Follow-up chips after assistant messages
                if msg["role"] == "assistant" and msg.get("follow_ups"):
                    st.markdown(
                        "<div class='mm-label' style='margin-top:12px;margin-bottom:6px;'>Follow up</div>",
                        unsafe_allow_html=True,
                    )
                    fu_cols = st.columns(len(msg["follow_ups"]))
                    for fi, fu in enumerate(msg["follow_ups"]):
                        if fu_cols[fi].button(fu, key=f"fu_{id(msg)}_{fi}", use_container_width=True):
                            st.session_state.pending_question = fu
                            st.rerun()

        # ── Consume pending question or wait for input ─────────────────────────
        question = None
        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""

        typed = st.chat_input("Ask anything across your manuals…")
        if typed:
            question = typed

        if question:
            st.session_state.chat_history.append(
                {"role": "user", "content": question, "chunks": None, "follow_ups": []}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                past_user_qs = [
                    m["content"] for m in st.session_state.chat_history
                    if m["role"] == "user" and m["content"] != question
                ]
                retrieval_query = question
                if past_user_qs and len(question.split()) < 12:
                    retrieval_query = f"{past_user_qs[-1]} {question}"

                with st.spinner("Searching manuals…"):
                    chunks = retrieve(retrieval_query, min_score=relevance_threshold)

                if not chunks:
                    answer = (
                        "I couldn't find relevant content above the current relevance "
                        f"threshold ({relevance_threshold:.2f}). Try lowering the threshold "
                        "in the sidebar, rephrasing, or checking that the right manual is indexed."
                    )
                    st.markdown(answer)
                    follow_ups = []
                else:
                    context  = build_context_prompt(chunks)
                    messages = build_chat_messages(
                        question,
                        context,
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

                    raw_answer = st.write_stream(_stream_response())
                    answer, follow_ups = _parse_follow_ups(raw_answer)

            st.session_state.chat_history.append({
                "role":       "assistant",
                "content":    answer,
                "chunks":     chunks or [],
                "follow_ups": follow_ups,
            })
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Manuals
# ══════════════════════════════════════════════════════════════════════════════

# Color palettes for manual cover cards (cycles through these)
_COVER_PALETTES = [
    ("#ff7849", "#1a0f0a"),  # orange
    ("#3b82f6", "#0a0f1a"),  # blue
    ("#10b981", "#0a1a12"),  # green
    ("#f59e0b", "#1a150a"),  # amber
    ("#8b5cf6", "#120a1a"),  # purple
    ("#ef4444", "#1a0a0a"),  # red
]

with tab_manuals:
    st.markdown(
        "<h2 style='margin-top:0;margin-bottom:0.25rem;'>Indexed Manuals</h2>"
        "<p style='color:#a1a1aa;font-size:13px;margin-bottom:1.5rem;'>"
        "Manuals currently in your knowledge base.</p>",
        unsafe_allow_html=True,
    )

    indexed_pdfs = get_indexed_pdfs()
    if not indexed_pdfs:
        st.markdown(
            "<div style='text-align:center;padding:48px 0;'>"
            "<div style='font-size:2rem;margin-bottom:12px;'>📚</div>"
            "<div style='font-size:15px;font-weight:600;color:#f5f5f7;margin-bottom:6px;'>No manuals indexed</div>"
            "<div style='font-size:13px;color:#a1a1aa;'>Upload PDFs in <b>Parse &amp; Edit</b> to build your knowledge base.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Card grid — 3 per row
        cols = st.columns(3)
        for i, entry in enumerate(indexed_pdfs):
            accent, bg = _COVER_PALETTES[i % len(_COVER_PALETTES)]
            manufacturer = entry.get("manufacturer", "")
            product_line = entry.get("product_line", "")
            doc_type     = entry.get("doc_type", "")
            revision     = entry.get("revision", "")
            title        = product_line or entry["source_pdf"]
            subtitle     = manufacturer or doc_type or ""
            meta_tags    = " · ".join(filter(None, [doc_type, revision]))
            chunk_label  = f"{entry['text_chunks']} text · {entry['image_chunks']} img"

            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #2a2a2f;border-radius:8px;
                        overflow:hidden;margin-bottom:16px;
                        box-shadow:0 1px 0 rgba(0,0,0,0.08),2px 2px 0 #2a2a2f;
                        background:#1a1a1e;
                    ">
                      <!-- Cover spine -->
                      <div style="
                        height:6px;background:{accent};
                      "></div>
                      <!-- Cover body -->
                      <div style="
                        padding:16px;
                        background:repeating-linear-gradient(
                          135deg,{bg} 0px,{bg} 10px,
                          rgba(255,255,255,0.01) 10px,rgba(255,255,255,0.01) 20px
                        );
                      ">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                          text-transform:uppercase;letter-spacing:0.12em;
                          color:{accent};margin-bottom:6px;">{subtitle or "Manual"}</div>
                        <div style="font-size:14px;font-weight:600;color:#f5f5f7;
                          line-height:1.3;margin-bottom:8px;">{title}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                          color:#6b6b74;">{meta_tags or entry['source_pdf']}</div>
                      </div>
                      <!-- Footer -->
                      <div style="
                        padding:8px 16px;border-top:1px solid #2a2a2f;
                        display:flex;justify-content:space-between;align-items:center;
                      ">
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                          color:#6b6b74;">{chunk_label}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Remove", key=f"del_{entry['source_pdf']}", use_container_width=True):
                    n = delete_pdf_from_index(entry["source_pdf"])
                    st.success(f"Removed {n} chunks for **{entry['source_pdf']}**.")
                    st.rerun()

        st.divider()
        total_all = get_total_chunk_count()
        st.markdown(
            f"<span class='mm-mono' style='color:#6b6b74;'>"
            f"{total_all} total chunks · {len(indexed_pdfs)} manual(s)</span>",
            unsafe_allow_html=True,
        )
