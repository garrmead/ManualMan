# app.py — ManualMan: RAG-powered knowledge assistant for pump equipment manuals

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import re
import base64

load_dotenv()

import anthropic

import config
from utils.pdf_parser import extract_chunks_from_pdf, get_pdf_page_count
from utils.embedder import (
    commit_chunks, get_indexed_pdfs, delete_pdf_from_index,
    get_total_chunk_count, get_manufacturers, get_doc_types,
)
from utils.retriever import retrieve, build_context_prompt, build_chat_messages
from utils.troubleshoot import classify_query, get_troubleshoot_system_addendum

# ── Design system CSS ─────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap');

:root {
  --bg:            oklch(0.18 0.005 250);
  --surface:       oklch(0.22 0.006 250);
  --surface-2:     oklch(0.25 0.006 250);
  --text:          oklch(0.96 0.003 250);
  --text-2:        oklch(0.72 0.005 250);
  --text-3:        oklch(0.52 0.006 250);
  --border:        oklch(0.32 0.006 250);
  --border-strong: oklch(0.42 0.006 250);
  --accent:        oklch(0.78 0.16 250);
  --accent-strong: oklch(0.78 0.16 250);
  --accent-10:     oklch(0.28 0.04 250);
  --accent-30:     oklch(0.40 0.06 250);
  --accent-on:     #fff;
  /* backward compat aliases */
  --bg-2: var(--surface);
  --fg:   var(--text);
  --fg-2: var(--text-2);
  --fg-3: var(--text-3);
  /* semantic */
  --ts-color: #f59e0b;
  --ts-dim: rgba(245,158,11,0.15);
  --ok-color: #4ade80;
  /* radii */
  --radius-sm: 4px; --radius-md: 6px; --radius-lg: 8px; --radius-pill: 999px;
  /* shadows */
  --shadow-card:   0 6px 20px rgba(20,40,80,0.06);
  --shadow-askbar: 0 1px 0 rgba(0,0,0,0.03), 0 12px 28px rgba(20,40,80,0.06);
  --shadow-btn:    0 1px 0 rgba(255,255,255,0.18) inset, 0 6px 14px rgba(20,40,80,0.12);
  --shadow-float:  0 8px 24px rgba(0,0,0,0.24), 0 2px 6px rgba(0,0,0,0.16);
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: oklch(0.985 0.004 250);
    --surface: #fff;
    --surface-2: oklch(0.97 0.005 250);
    --text: oklch(0.18 0.01 250);
    --text-2: oklch(0.40 0.008 250);
    --text-3: oklch(0.58 0.006 250);
    --border: oklch(0.91 0.006 250);
    --border-strong: oklch(0.82 0.006 250);
    --accent: oklch(0.55 0.16 250);
    --accent-strong: oklch(0.42 0.16 250);
    --accent-10: oklch(0.96 0.04 250);
    --accent-30: oklch(0.88 0.08 250);
    --bg-2: var(--surface); --fg: var(--text); --fg-2: var(--text-2); --fg-3: var(--text-3);
  }
}

html, body, [class*="css"], .stApp {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-size: 14px !important;
  line-height: 1.5 !important;
}
h1 { font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: var(--text) !important; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; color: var(--text) !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: var(--text) !important; }

.stCaption, small, caption, [data-testid="stCaptionContainer"] p {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.05em !important;
  color: var(--text-3) !important;
}

.stApp { background-color: var(--bg) !important; }
.stApp > header {
  background-color: var(--bg) !important;
  border-bottom: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] {
  background-color: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
  font-size: 13px !important;
  color: var(--text-2) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 10px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--text-3) !important;
  margin-top: 1.4rem !important;
  margin-bottom: 0.4rem !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border) !important;
  margin: 0.75rem 0 !important;
}

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
  color: var(--text-3) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 16px !important;
  text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

.stButton > button[kind="primary"] {
  background-color: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 9px 16px !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: var(--accent-strong) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px var(--accent-30) !important;
}
.stButton > button[kind="secondary"],
.stButton > button:not([kind]) {
  background-color: var(--surface) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 13px !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind]):hover {
  background-color: var(--surface-2) !important;
  border-color: var(--border-strong) !important;
  color: var(--text) !important;
}

.stTextInput input, [data-testid="stTextInput"] input,
[data-baseweb="select"] [data-baseweb="input"] {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 13px !important;
}
.stTextInput input:focus, [data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-10) !important;
}
[data-baseweb="select"] {
  background-color: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-md) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background-color: var(--accent) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
  background-color: var(--accent) !important;
}

[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 1.6rem !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--text-3) !important;
}

[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  font-size: 13px !important;
}
.stSuccess { border-left: 3px solid #4ade80 !important; }
.stError   { border-left: 3px solid #f87171 !important; }
.stWarning { border-left: 3px solid #fbbf24 !important; }
.stInfo    { border-left: 3px solid var(--accent) !important; }

[data-testid="stProgressBar"] > div {
  background-color: var(--accent) !important;
  border-radius: var(--radius-pill) !important;
}

[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  background-color: var(--surface) !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stExpander"] summary {
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--text-2) !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

[data-testid="stDataEditor"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}
[data-testid="stDataEditor"] > div,
.dvn-scroller, .dvn-scroller > div {
  background-color: var(--surface) !important;
}

/* ── Claude-style chat messages — no avatars ── */
[data-testid="chatAvatarIcon-assistant"],
[data-testid="chatAvatarIcon-user"] {
  display: none !important;
  width: 0 !important;
  min-width: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}

[data-testid="stChatMessage"] {
  border: none !important;
  border-radius: 0 !important;
  background: transparent !important;
  padding: 20px 0 !important;
  margin-bottom: 0 !important;
  box-shadow: none !important;
  border-bottom: 1px solid var(--border) !important;
}
[data-testid="stChatMessage"]:last-of-type {
  border-bottom: none !important;
}

/* User turn — bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: var(--surface) !important;
  border-radius: 12px 12px 2px 12px !important;
  padding: 10px 14px !important;
  border: none !important;
  border-bottom: none !important;
  margin-bottom: 4px !important;
  max-width: 78% !important;
  margin-left: auto !important;
}

[data-testid="stChatInput"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background-color: var(--surface) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-10) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 14px !important;
  color: var(--text) !important;
  background: transparent !important;
}

hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1rem 0 !important;
}

[data-testid="stFileUploader"] {
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--radius-md) !important;
  background-color: var(--surface) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  background-color: var(--accent-10) !important;
}

[data-testid="stImage"] img {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow-float) !important;
}

code, pre {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

input[type="checkbox"]:checked { accent-color: var(--accent) !important; }

/* ── Custom badges ── */
.mm-accent { color: var(--accent) !important; font-style: italic; }
.mm-mono   { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.06em; }
.mm-label  {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-3);
}
.mm-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.mm-badge-ts    { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.mm-badge-spec  { background: var(--accent-10); color: var(--accent); border: 1px solid var(--accent-30); }
.mm-badge-proc  { background: rgba(74,222,128,0.10); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.mm-badge-table { background: rgba(167,139,250,0.12); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.mm-badge-ocr   { background: rgba(251,146,60,0.12); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
.mm-score-bar {
  display: inline-block;
  height: 3px;
  border-radius: 2px;
  background: var(--accent);
  opacity: 0.7;
}

/* ── Ask bar (home screen) ── */
.tmc-ask-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--surface);
  border: 1.5px solid var(--border-strong);
  border-radius: 12px;
  padding: 7px 16px;
  box-shadow: var(--shadow-askbar);
  transition: border-color 0.15s ease;
}
.tmc-ask-bar:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-10), var(--shadow-askbar);
}

/* ── Nav tab item in sidebar ── */
.tmc-nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
  border: 1px solid transparent;
  margin-bottom: 2px;
}
.tmc-nav-item.active {
  background: var(--accent-10);
  border-color: var(--accent-30);
}
.tmc-nav-item:hover:not(.active) {
  background: var(--surface-2);
}
.tmc-nav-label {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  line-height: 1.2;
}
.tmc-nav-sublabel {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-3);
}

/* ── Suggested prompt row ── */
.tmc-prompt-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
  margin-bottom: 6px;
  text-decoration: none;
}
.tmc-prompt-row:hover {
  background: var(--accent-10);
  border-color: var(--accent-30);
}
.tmc-prompt-sparkle { color: var(--accent); font-size: 13px; flex-shrink: 0; }
.tmc-prompt-text { font-size: 14px; color: var(--text); flex: 1; }
.tmc-prompt-arrow { color: var(--text-3); font-size: 13px; flex-shrink: 0; }

/* ── Impeller badge (home header) ── */
.tmc-impeller-badge {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  background: var(--accent-10);
  border: 1.5px solid var(--accent-30);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 16px;
}

/* ── Manual card ── */
.tmc-manual-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
  margin-bottom: 12px;
}
.tmc-manual-card:hover {
  border-color: var(--accent-30);
  box-shadow: var(--shadow-card);
  transform: translateY(-1px);
}
.tmc-manual-card-footer {
  padding: 8px 14px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* ── Upload drop zone ── */
.tmc-dropzone {
  border: 2px dashed var(--border-strong);
  border-radius: 14px;
  padding: 44px 24px;
  text-align: center;
  background: var(--surface);
  transition: border-color 0.15s ease, background 0.15s ease;
}
.tmc-dropzone:hover {
  border-color: var(--accent);
  background: var(--accent-10);
}

/* ── "New question" CTA button ── */
.tmc-new-question-btn {
  width: 100%;
  background: var(--accent);
  color: var(--accent-on);
  border: none;
  border-radius: 999px;
  padding: 10px 16px;
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  cursor: pointer;
  box-shadow: var(--shadow-btn);
  transition: opacity 0.12s ease, transform 0.12s ease;
  margin-bottom: 16px;
}
.tmc-new-question-btn:hover {
  opacity: 0.92;
  transform: translateY(-1px);
}

/* ── Thinking indicator ── */
@keyframes tmc-bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-5px); }
}
.tmc-dot { display: inline-block; width: 5px; height: 5px; border-radius: 50%;
  background: var(--accent); animation: tmc-bounce 1.2s infinite ease-in-out; }
.tmc-dot:nth-child(2) { animation-delay: 0.2s; }
.tmc-dot:nth-child(3) { animation-delay: 0.4s; }

@media (prefers-reduced-motion: reduce) {
  .tmc-dot { animation: none; }
  .tmc-manual-card, .tmc-prompt-row, .tmc-new-question-btn { transition: none !important; }
  [class*="stButton"] button { transition: none !important; }
}
</style>
"""

# ── Impeller SVG logo ─────────────────────────────────────────────────────────
_IMPELLER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-12 -12 24 24" fill="none">'
    '<g stroke="currentColor" stroke-width="1.6" stroke-linecap="round">'
    '<path d="M0,-3.0 C0.8,-5.5 5.5,-8.5 8.0,-7.0"/>'
    '<path d="M0,-3.0 C0.8,-5.5 5.5,-8.5 8.0,-7.0" transform="rotate(60)"/>'
    '<path d="M0,-3.0 C0.8,-5.5 5.5,-8.5 8.0,-7.0" transform="rotate(120)"/>'
    '<path d="M0,-3.0 C0.8,-5.5 5.5,-8.5 8.0,-7.0" transform="rotate(180)"/>'
    '<path d="M0,-3.0 C0.8,-5.5 5.5,-8.5 8.0,-7.0" transform="rotate(240)"/>'
    '<path d="M0,-3.0 C0.8,-5.5 5.5,-8.5 8.0,-7.0" transform="rotate(300)"/>'
    '</g>'
    '<circle cx="0" cy="0" r="3.2" fill="currentColor"/>'
    '<circle cx="0" cy="0" r="1.2" fill="var(--bg,#18192a)"/>'
    '<circle cx="0" cy="0" r="0.4" fill="var(--accent,#8ca8f5)"/>'
    '</svg>'
)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are ChatTMC, a technical knowledge assistant for pump equipment manuals \
(Goulds, Aurora, Gorman-Rupp, ITT, Grundfos, Xylem, Flowserve, and similar manufacturers).

Rules:
1. Answer using ONLY the provided context. Do not draw on outside knowledge.
2. BREVITY FIRST: Lead with the single most important fact or value in one sentence. \
   Put supporting detail after, not before.
3. If an IMAGE chunk is cited (performance curve, drawing, diagram): write ONE sentence \
   referencing it as "See [Source N]." DO NOT describe or recreate the data — the image \
   displays automatically and speaks for itself.
4. Cite sources inline as [Source N] for every claim.
5. For specs and tables: use a tight bullet list or table, not prose paragraphs.
6. If context lacks enough information, say "No supporting documentation found." — nothing more.
7. After your answer, on a new line write exactly: \
   FOLLOW-UPS: <question 1> | <question 2> | <question 3> \
   Max 10 words each. Skip if answer was "No supporting documentation found."\
"""

# ── Manual metadata fields ────────────────────────────────────────────────────
_DOC_TYPES = [
    "Installation Manual", "Operation Manual", "Parts Manual",
    "Selection Guide", "Technical Data Sheet", "Submittal Drawing", "Other",
]

_SUBTYPE_BADGES = {
    "troubleshooting": '<span class="mm-badge mm-badge-ts">⚡ Troubleshooting</span>',
    "specification":   '<span class="mm-badge mm-badge-spec">⚙ Specification</span>',
    "procedure":       '<span class="mm-badge mm-badge-proc">▶ Procedure</span>',
    "table":           '<span class="mm-badge mm-badge-table">▦ Table</span>',
    "image":           '<span class="mm-badge mm-badge-table">🖼 Image</span>',
}

_COVER_PALETTES = [
    ("#6186f5", "#0c0e1a"),  # blue (matches brand)
    ("#10b981", "#0a1a12"),  # green
    ("#f59e0b", "#1a150a"),  # amber
    ("#ef4444", "#1a0a0a"),  # red
    ("#8b5cf6", "#120a1a"),  # violet
    ("#06b6d4", "#0a1518"),  # teal
]

_SUGGESTED_PROMPTS = [
    "Show me the performance curve",
    "What are the installation torque specs?",
    "What materials are available for wetted parts?",
    "Show me the dimensional drawing",
    "Pump vibrating excessively — possible causes?",
    "What fault codes are covered in this manual?",
    "Bearing temperature running high — what to check?",
    "What is the mechanical seal replacement procedure?",
    "Compare impeller trim options",
    "What is the max operating pressure and temperature?",
    "Troubleshoot low discharge pressure",
    "List compatible spare parts",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cited_source_indices(answer_text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[Source\s+(\d+)\]", answer_text, re.IGNORECASE)}


def _parse_follow_ups(raw_answer: str) -> tuple[str, list[str]]:
    match = re.search(r"\n*FOLLOW-UPS?:\s*(.+?)$", raw_answer, re.IGNORECASE | re.DOTALL)
    if not match:
        return raw_answer, []
    clean  = raw_answer[: match.start()].strip()
    parts  = [q.strip().lstrip("•*-").strip() for q in re.split(r"\||\n", match.group(1))]
    follow_ups = [q for q in parts if q][:3]
    return clean, follow_ups


def _image_quality_score(image_path: str) -> float:
    try:
        from PIL import Image
        img    = Image.open(image_path).convert("L").resize((32, 32))
        pixels = list(img.getdata())
        mean   = sum(pixels) / len(pixels)
        std    = (sum((p - mean) ** 2 for p in pixels) / len(pixels)) ** 0.5
        return min(std / 80.0, 1.0)
    except Exception:
        return 0.5

_QUALITY_THRESHOLD = 0.20


def _score_bar(score: float, width_px: int = 60) -> str:
    """Mini inline score bar as HTML."""
    pct = min(max(score, 0.0), 1.0)
    bar_w = int(pct * width_px)
    color = "var(--accent)" if pct > 0.6 else "#f59e0b" if pct > 0.35 else "var(--text-3)"
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="width:{width_px}px;height:3px;background:var(--border);border-radius:2px;overflow:hidden;">'
        f'<div style="width:{bar_w}px;height:3px;background:{color};border-radius:2px;"></div>'
        f'</div>'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:10px;color:var(--text-3);">{pct:.2f}</span>'
        f'</div>'
    )


def _format_citation(chunk: dict) -> str:
    parts = [f"`{chunk['source_pdf']}`"]
    meta  = [chunk.get(k, "") for k in ("manufacturer", "product_line", "doc_type", "revision")]
    meta  = [m for m in meta if m]
    if meta:
        parts.append(f"({' · '.join(meta)})")
    if chunk.get("section_title"):
        parts.append(f"§ *{chunk['section_title']}*")
    parts.append(f"Page **{chunk['page_number']}**")
    return " · ".join(parts)


_AVATAR_USER      = "○"   # replaces human icon
_AVATAR_ASSISTANT = "▸"   # replaces robot icon


def _has_image_chunks(chunks: list[dict]) -> bool:
    return any(
        c.get("chunk_type") == "image" and Path(c.get("image_path", "")).exists()
        for c in chunks
    )


def _render_answer(
    content: str,
    chunks: list[dict],
    is_visual: bool,
    show_scores: bool,
) -> None:
    """
    Render an assistant answer with progressive disclosure.

    Visual queries (curves, drawings, diagrams):
      → Images shown large and first
      → Text answer in a collapsed expander ("Text notes ↓")

    All other queries:
      → First paragraph shown immediately
      → Everything after the first paragraph in "Show more ↓" expander
    """
    cited = _cited_source_indices(content)
    image_chunks = [
        c for c in chunks
        if c.get("chunk_type") == "image"
        and Path(c.get("image_path", "")).exists()
        and (not cited or (list(chunks).index(c) + 1) in cited)
    ]

    if is_visual and image_chunks:
        # ── Image-first layout ─────────────────────────────────────────────────
        _render_inline_images(chunks, answer_text=content)
        # Collapse the text — it adds nothing when the image is the answer
        clean = content.strip()
        if clean and clean.lower() != "no supporting documentation found.":
            with st.expander("Text notes ↓", expanded=False):
                st.markdown(clean)
    else:
        # ── Progressive disclosure ─────────────────────────────────────────────
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        if paragraphs:
            st.markdown(paragraphs[0])
        if len(paragraphs) > 1:
            with st.expander("Show more ↓", expanded=False):
                st.markdown("\n\n".join(paragraphs[1:]))
        # Inline images below the lead paragraph (text-mode citations)
        if image_chunks:
            _render_inline_images(chunks, answer_text=content)

    _render_sources(chunks, show_scores=show_scores)


def _render_inline_images(chunks: list[dict], answer_text: str = "") -> None:
    from utils.vision import IMAGE_TYPE_LABELS
    cited = _cited_source_indices(answer_text) if answer_text else set()

    for source_num, chunk in enumerate(chunks, start=1):
        if chunk.get("chunk_type") != "image":
            continue
        if not Path(chunk.get("image_path", "")).exists():
            continue
        if cited and source_num not in cited:
            continue

        label    = IMAGE_TYPE_LABELS.get(chunk.get("image_type", ""), "Diagram")
        citation = _format_citation(chunk)
        st.markdown(f"**{label}** — {citation}")
        _, img_col, _ = st.columns([1, 5, 1])
        img_col.image(chunk["image_path"], use_container_width=True)


def _render_sources(chunks: list[dict], show_scores: bool = True) -> None:
    text_chunks = [c for c in chunks if c.get("chunk_type") != "image"]
    if not text_chunks:
        return

    with st.expander(
        f"Sources · {len(text_chunks)} chunk(s)", expanded=False
    ):
        for i, chunk in enumerate(text_chunks, 1):
            subtype = chunk.get("chunk_subtype", "text")
            badge   = _SUBTYPE_BADGES.get(subtype, "")
            ocr_b   = '<span class="mm-badge mm-badge-ocr">OCR</span>' if chunk.get("ocr_used") == "true" else ""
            ts_b    = '<span class="mm-badge mm-badge-ts">⚡TS</span>' if chunk.get("is_troubleshooting") == "true" else ""

            header = (
                f"<div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;"
                f"margin-bottom:4px;'>"
                f"<strong style='color:var(--text);'>{i}.</strong>"
                f"<span style='color:var(--text-2);font-size:12px;'>{_format_citation(chunk)}</span>"
                f"{badge}{ocr_b}{ts_b}"
                f"</div>"
            )
            st.markdown(header, unsafe_allow_html=True)

            if show_scores:
                v_score = chunk.get("score", 0.0)
                b_score = min(chunk.get("bm25_score", 0.0) / 10.0, 1.0)  # normalize BM25
                r_score = chunk.get("rerank_score", None)
                score_html = (
                    f"<div style='display:flex;gap:16px;margin-bottom:6px;flex-wrap:wrap;'>"
                    f"<div><span class='mm-label'>Vector</span>&nbsp;{_score_bar(v_score)}</div>"
                    f"<div><span class='mm-label'>BM25</span>&nbsp;{_score_bar(b_score)}</div>"
                )
                if r_score is not None:
                    # Cross-encoder scores are logits; normalize via sigmoid approximation
                    import math
                    r_norm = 1.0 / (1.0 + math.exp(-r_score / 2))
                    score_html += f"<div><span class='mm-label'>Rerank</span>&nbsp;{_score_bar(r_norm)}</div>"
                score_html += "</div>"
                st.markdown(score_html, unsafe_allow_html=True)

            if chunk.get("tags"):
                st.caption(f"Tags: {chunk['tags']}")

            preview = chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else "")
            st.caption(preview)

            # "View in Manuals tab" button — sets cross-tab navigation target
            btn_key = f"view_src_{id(chunk)}_{i}"
            if st.button(
                f"View page {chunk['page_number']} →",
                key=btn_key,
                help=f"Open {chunk['source_pdf']} at page {chunk['page_number']} in Manuals tab",
            ):
                st.session_state.pdf_view_target = {
                    "source_pdf": chunk["source_pdf"],
                    "page":       chunk["page_number"],
                }
                st.toast(f"Opening {chunk['source_pdf']} p.{chunk['page_number']} — switch to Manuals tab", icon="📄")
                st.rerun()

            if i < len(text_chunks):
                st.divider()


def _build_chunks_df(chunks: list[dict]) -> pd.DataFrame:
    type_icon = {"table": "▦ Table", "procedure": "▶ Procedure",
                 "troubleshooting": "⚡ TS", "specification": "⚙ Spec"}
    rows = [
        {
            "keep":          c["keep"],
            "type":          type_icon.get(c.get("chunk_subtype", ""), "📝 Text"),
            "ocr":           "✓" if c.get("ocr_used") else "",
            "source_pdf":    c["source_pdf"],
            "page":          c["page_number"],
            "section":       c.get("section_title", "")[:50],
            "text":          c["text"],
            "tags":          c["tags"],
        }
        for c in chunks
    ]
    df = pd.DataFrame(rows)
    df["keep"]       = df["keep"].astype(bool)
    df["page"]       = df["page"].astype(int)
    df["source_pdf"] = df["source_pdf"].astype(str)
    df["text"]       = df["text"].astype(str)
    df["tags"]       = df["tags"].astype(str)
    df["section"]    = df["section"].astype(str)
    df["ocr"]        = df["ocr"].astype(str)
    return df


def _render_image_grid(paths: list[str], key_prefix: str) -> None:
    cols = st.columns(4)
    for i, img_path in enumerate(paths):
        col = cols[i % 4]
        try:
            col.image(img_path, use_container_width=True)
            included = col.checkbox(
                "Include",
                value=(img_path not in st.session_state.excluded_images),
                key=f"{key_prefix}_{i}",
            )
            col.caption(Path(img_path).name)
            if included:
                st.session_state.excluded_images.discard(img_path)
            else:
                st.session_state.excluded_images.add(img_path)
        except Exception:
            col.caption(f"⚠ {img_path}")


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatTMC",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_CSS, unsafe_allow_html=True)

for d in [config.UPLOADS_DIR, config.IMAGES_DIR, config.CHROMA_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Session state ─────────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "parsed_chunks":       [],
    "chunks_df":           None,
    "parse_version":       0,
    "edited_df":           None,
    "chat_history":        [],
    "show_img_gallery":    True,
    "show_unsure_gallery": False,
    "last_committed":      0,
    "pdf_metadata":        {},
    "pending_question":    "",
    "chat_input_draft":    "",
    "excluded_images":     set(),
    "pdf_view_target":     None,   # {"source_pdf": str, "page": int}
    "retrieval_filters":   {},
    "entity_extract_on":   True,
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API key guard ─────────────────────────────────────────────────────────────
_voyage_ok    = bool(config.VOYAGE_API_KEY)
_anthropic_ok = bool(config.ANTHROPIC_API_KEY)
if not _voyage_ok or not _anthropic_ok:
    missing = []
    if not _voyage_ok:    missing.append("VOYAGE_API_KEY")
    if not _anthropic_ok: missing.append("ANTHROPIC_API_KEY")
    st.error(
        f"**Missing API keys:** {', '.join(missing)}  \n"
        "Add them to your `.env` file and restart.  \n"
        "```\nVOYAGE_API_KEY=your-key\nANTHROPIC_API_KEY=your-key\n```"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:4px 0 20px;">
          <!-- Logo row -->
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
            <div style="width:32px;height:32px;background:var(--accent-10);border:1.5px solid var(--accent-30);
              border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;
              color:var(--accent);">
              <div style="width:20px;height:20px;">{_IMPELLER_SVG}</div>
            </div>
            <div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:15px;font-weight:600;
                color:var(--text);letter-spacing:-0.01em;line-height:1;">
                Chat<span style="color:var(--accent);font-weight:700;">TMC</span>
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                color:var(--text-3);text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;">
                Technical Library
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API status ────────────────────────────────────────────────────────────
    st.markdown("<div class='mm-label' style='margin-bottom:8px;'>System Status</div>", unsafe_allow_html=True)
    total   = get_total_chunk_count()
    indexed = get_indexed_pdfs()
    reranker_label = "ON" if config.RERANKER_ENABLED else "OFF"
    st.markdown(
        f"<div style='font-size:12px;line-height:2;'>"
        f"<span style='color:#4ade80;'>●</span> Voyage AI (embed)&nbsp;&nbsp;"
        f"<span style='color:#4ade80;'>●</span> Anthropic (LLM)<br>"
        f"<span style='color:#4ade80;'>●</span> Hybrid BM25+Vector&nbsp;&nbsp;"
        f"<span style='color:{'#4ade80' if config.RERANKER_ENABLED else '#6b6b74'};'>●</span>"
        f" Reranker {reranker_label}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Knowledge base stats ──────────────────────────────────────────────────
    st.markdown("<div class='mm-label' style='margin-bottom:8px;'>Knowledge Base</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Chunks", total)
    m2.metric("Manuals", len(indexed))

    # ── Retrieval filters ─────────────────────────────────────────────────────
    st.markdown("<div class='mm-label' style='margin-bottom:8px;'>Search Filters</div>", unsafe_allow_html=True)
    manufacturers = get_manufacturers()
    doc_types     = get_doc_types()

    filter_mfr = st.selectbox(
        "Manufacturer",
        ["All"] + manufacturers,
        key="filter_mfr",
        help="Filter results to a specific manufacturer",
    )
    filter_dtype = st.selectbox(
        "Document type",
        ["All"] + doc_types,
        key="filter_dtype",
        help="Filter to a specific document type",
    )

    active_filters: dict = {}
    if filter_mfr   != "All": active_filters["manufacturer"] = filter_mfr
    if filter_dtype != "All": active_filters["doc_type"]     = filter_dtype
    st.session_state.retrieval_filters = active_filters

    # ── Chat settings ─────────────────────────────────────────────────────────
    st.markdown("<div class='mm-label' style='margin-bottom:8px;'>Chat Settings</div>", unsafe_allow_html=True)
    relevance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.0, max_value=1.0,
        value=config.MIN_RELEVANCE_SCORE, step=0.05,
        help="Minimum vector similarity to include a chunk. Hybrid mode combines this with BM25.",
    )
    history_turns = st.select_slider(
        "Conversation memory",
        options=[0, 1, 2, 3, 5],
        value=config.MAX_HISTORY_TURNS,
        help="Past Q&A pairs sent to Claude for follow-up context.",
    )
    show_scores = st.toggle("Show retrieval scores", value=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("<div class='mm-label' style='margin-bottom:8px;'>Upload Manuals</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF manuals here",
        type=["pdf"],
        accept_multiple_files=True,
        key="sidebar_uploader",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) queued")
        for f in uploaded_files:
            st.caption(f"{f.name}  ({f.size / 1024:.0f} KB)")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_manuals, tab_upload = st.tabs(
    ["ChatTMC", "Document Library", "File Upload"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — File Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown(
        """
        <div style="padding:28px 0 20px;max-width:880px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
            text-transform:uppercase;letter-spacing:0.12em;color:var(--text-3);margin-bottom:8px;">Upload</div>
          <h1 style="font-size:1.5rem;font-weight:600;letter-spacing:-0.02em;color:var(--text);margin:0 0 8px;">
            Add documents to your library
          </h1>
          <p style="font-size:14px;color:var(--text-2);max-width:540px;margin:0 0 24px;line-height:1.6;">
            Upload service manuals, spec sheets, parts catalogs, or service bulletins.
            Documents are OCR&#8209;processed and indexed for retrieval.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not uploaded_files:
        st.info("Drop one or more PDFs in the sidebar uploader, fill in the metadata, then click **Parse PDFs**.")
    else:
        # ── Manual metadata ────────────────────────────────────────────────────
        st.subheader("Manual Metadata")
        st.caption("Stored with every chunk and shown in citations. Fill in once per PDF before parsing.")
        for f in uploaded_files:
            with st.expander(f"📄 {f.name}", expanded=True):
                c1, c2 = st.columns(2)
                c1.text_input("Manufacturer",         key=f"meta_{f.name}_mfr",   placeholder="e.g. Goulds Pumps")
                c2.text_input("Product Line / Model",  key=f"meta_{f.name}_prod",  placeholder="e.g. 3196")
                c3, c4 = st.columns(2)
                c3.selectbox("Document Type", _DOC_TYPES, key=f"meta_{f.name}_dtype")
                c4.text_input("Revision / Date",       key=f"meta_{f.name}_rev",   placeholder="e.g. Rev. 2023-Q1")

        # ── Parse options ──────────────────────────────────────────────────────
        st.divider()
        col_btn, col_opts, col_info = st.columns([2, 2, 4])
        with col_btn:
            parse_clicked = st.button("🔍 Parse PDFs", type="primary", use_container_width=True)
        with col_opts:
            extract_entities_opt = st.toggle(
                "Extract entities",
                value=st.session_state.entity_extract_on,
                help="Use Claude Haiku to extract model numbers, materials, RPM, etc. from each chunk. Improves faceted search. Adds ~30s/100 chunks.",
                key="entity_opt",
            )
            st.session_state.entity_extract_on = extract_entities_opt
        with col_info:
            total_pages = sum(get_pdf_page_count(f.getvalue()) for f in uploaded_files)
            st.caption(
                f"{len(uploaded_files)} PDF(s) · {total_pages} pages  \n"
                f"Chunk: ~{config.CHUNK_SIZE} tokens · Overlap: {config.CHUNK_OVERLAP} · "
                f"OCR threshold: {config.OCR_MIN_PAGE_CHARS} chars/page"
            )

        if parse_clicked:
            st.session_state.pdf_metadata = {
                f.name: {
                    "manufacturer": st.session_state.get(f"meta_{f.name}_mfr", ""),
                    "product_line": st.session_state.get(f"meta_{f.name}_prod", ""),
                    "doc_type":     st.session_state.get(f"meta_{f.name}_dtype", ""),
                    "revision":     st.session_state.get(f"meta_{f.name}_rev", ""),
                }
                for f in uploaded_files
            }
            all_chunks: list[dict] = []
            progress = st.progress(0, text="Starting…")
            for i, f in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Parsing {f.name}…")
                pdf_bytes = f.getvalue()
                (config.UPLOADS_DIR / f.name).write_bytes(pdf_bytes)
                all_chunks.extend(extract_chunks_from_pdf(pdf_bytes, f.name))
            progress.progress(1.0, text="Done!")
            progress.empty()

            st.session_state.parsed_chunks    = all_chunks
            st.session_state.chunks_df        = _build_chunks_df(all_chunks)
            st.session_state.parse_version   += 1
            st.session_state.edited_df        = None
            st.session_state.excluded_images  = set()

            # Stats
            n_text = sum(1 for c in all_chunks if c.get("chunk_subtype") not in ("table", "image"))
            n_tbl  = sum(1 for c in all_chunks if c.get("chunk_subtype") == "table")
            n_ts   = sum(1 for c in all_chunks if c.get("chunk_subtype") == "troubleshooting")
            n_ocr  = sum(1 for c in all_chunks if c.get("ocr_used"))
            img_ct = sum(len(c["image_paths"]) for c in all_chunks)
            st.success(
                f"Extracted **{len(all_chunks)} chunks** — "
                f"{n_text} text · {n_tbl} tables · {n_ts} troubleshooting · "
                f"{n_ocr} OCR-recovered · **{img_ct} image(s)**"
            )

        # ── Chunk editor ───────────────────────────────────────────────────────
        if st.session_state.chunks_df is not None:
            n_chunks = len(st.session_state.chunks_df)
            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:6px;'>"
                f"<span style='font-size:16px;font-weight:600;'>Chunk Editor</span>"
                f"<span class='mm-mono' style='color:var(--text-3);'>{n_chunks} chunk(s)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if n_chunks == 0:
                st.warning(
                    "No text chunks extracted. The PDF may be scanned (image-only pages). "
                    "OCR was attempted for pages with images. "
                    "Any classified images can still be committed."
                )
            else:
                st.caption(
                    "Double-click **Chunk Text** to edit · "
                    "**Tags** — comma-separated · "
                    "Uncheck **Keep?** to exclude · "
                    "**Type** shows detected content subtype · "
                    "**OCR** = Claude-extracted text"
                )

            if n_chunks > 0:
                edited_df = st.data_editor(
                    st.session_state.chunks_df,
                    key=f"chunk_editor_{st.session_state.parse_version}",
                    column_config={
                        "keep":       st.column_config.CheckboxColumn("Keep?", width="small"),
                        "type":       st.column_config.TextColumn("Type",    disabled=True, width="small"),
                        "ocr":        st.column_config.TextColumn("OCR",     disabled=True, width="small"),
                        "source_pdf": st.column_config.TextColumn("PDF",     disabled=True, width="medium"),
                        "page":       st.column_config.NumberColumn("Page",  disabled=True, width="small", format="%d"),
                        "section":    st.column_config.TextColumn("Section", disabled=True, width="medium"),
                        "text":       st.column_config.TextColumn("Chunk Text", width="large"),
                        "tags":       st.column_config.TextColumn("Tags",    width="medium"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=480,
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
                st.session_state.edited_df = st.session_state.chunks_df
                keep_count = 0

            # ── Image gallery ──────────────────────────────────────────────────
            all_image_paths: list[str] = []
            for chunk in st.session_state.parsed_chunks:
                for p in chunk["image_paths"]:
                    if p not in all_image_paths:
                        all_image_paths.append(p)

            if all_image_paths:
                recommended = [p for p in all_image_paths if _image_quality_score(p) >= _QUALITY_THRESHOLD]
                unsure      = [p for p in all_image_paths if _image_quality_score(p) <  _QUALITY_THRESHOLD]
                excluded_ct = len(st.session_state.excluded_images & set(all_image_paths))

                # Recommended
                hdr_c, tog_c = st.columns([6, 1])
                hdr_c.markdown(
                    f"<div style='font-size:13px;font-weight:600;padding-top:6px;'>"
                    f"Recommended &nbsp;"
                    f"<span class='mm-mono' style='color:var(--text-3);'>{len(recommended)} image(s)"
                    + (f" · <span style='color:var(--accent);'>{excluded_ct} excluded total</span>" if excluded_ct else "")
                    + "</span></div>",
                    unsafe_allow_html=True,
                )
                if tog_c.button(
                    "Hide ▲" if st.session_state.show_img_gallery else "Show ▼",
                    key="toggle_gallery",
                ):
                    st.session_state.show_img_gallery = not st.session_state.show_img_gallery
                    st.rerun()

                if st.session_state.show_img_gallery:
                    rc1, rc2, _ = st.columns([1, 1, 5])
                    if rc1.button("Exclude all",  key="excl_rec"):
                        st.session_state.excluded_images.update(recommended); st.rerun()
                    if rc2.button("Include all",  key="incl_rec"):
                        [st.session_state.excluded_images.discard(p) for p in recommended]; st.rerun()
                    st.divider()
                    if recommended:
                        _render_image_grid(recommended, "rec")
                    else:
                        st.caption("No recommended images on these pages.")

                # Unsure
                if unsure:
                    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                    uhdr, utog = st.columns([6, 1])
                    uhdr.markdown(
                        f"<div style='font-size:13px;font-weight:600;padding-top:6px;'>"
                        f"Unsure &nbsp;<span class='mm-mono' style='color:var(--text-3);'>"
                        f"{len(unsure)} image(s) · likely backgrounds or blurs</span></div>",
                        unsafe_allow_html=True,
                    )
                    if utog.button(
                        "Hide ▲" if st.session_state.show_unsure_gallery else "Show ▼",
                        key="toggle_unsure",
                    ):
                        st.session_state.show_unsure_gallery = not st.session_state.show_unsure_gallery
                        st.rerun()

                    if st.session_state.show_unsure_gallery:
                        uc1, uc2, _ = st.columns([1, 1, 5])
                        if uc1.button("Exclude all unsure", key="excl_unsure"):
                            st.session_state.excluded_images.update(unsure); st.rerun()
                        if uc2.button("Include all unsure", key="incl_unsure"):
                            [st.session_state.excluded_images.discard(p) for p in unsure]; st.rerun()
                        st.divider()
                        _render_image_grid(unsure, "uns")
            else:
                st.caption("No images extracted (text-only PDF or all images below size threshold).")

            # ── Commit ─────────────────────────────────────────────────────────
            st.divider()
            col_commit, col_info2 = st.columns([2, 5])

            with col_info2:
                if not _voyage_ok:
                    st.error("Voyage AI key missing — add it to `.env` and restart.")
                else:
                    all_imgs  = {p for c in st.session_state.parsed_chunks for p in c["image_paths"]}
                    incl_imgs = all_imgs - st.session_state.excluded_images
                    entity_note = " + entity extraction" if st.session_state.entity_extract_on else ""
                    st.caption(
                        f"Will embed **{keep_count} text chunk(s)** + classify & embed "
                        f"**{len(incl_imgs)} image(s)**{entity_note}.  \n"
                        "Re-committing replaces existing entries for the same PDF."
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
                            extract_entities=st.session_state.entity_extract_on,
                        )
                        commit_progress.empty()
                        st.session_state.last_committed = counts["text_chunks"]
                        st.success(
                            f"✅ **{counts['text_chunks']} text chunks** + "
                            f"**{counts['image_chunks']} image chunks** committed to knowledge base.  \n"
                            "Switch to the **Chat** tab to start asking questions."
                        )
                    except Exception as e:
                        commit_progress.empty()
                        st.error(f"Commit failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ChatTMC
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    total_chunks = get_total_chunk_count()
    indexed_pdfs = get_indexed_pdfs()

    if total_chunks == 0:
        st.markdown(
            "<div style='text-align:center;padding:48px 0;'>"
            "<div style='font-size:2rem;margin-bottom:12px;'>📂</div>"
            "<div style='font-size:15px;font-weight:600;color:var(--text);margin-bottom:6px;'>No manuals indexed yet</div>"
            "<div style='font-size:13px;color:var(--text-2);'>Go to <b>File Upload</b>, upload a PDF, and click <b>Commit to Knowledge Base</b>.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Branded header (empty state) or compact header ──────────────────────
        if not st.session_state.chat_history:
            st.markdown(
                f"""
                <div style="text-align:center;padding:10vh 32px 40px;max-width:680px;margin:0 auto;">
                  <!-- Impeller badge -->
                  <div class="tmc-impeller-badge" style="color:var(--accent);">
                    <div style="width:36px;height:36px;">{_IMPELLER_SVG}</div>
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    text-transform:uppercase;letter-spacing:0.12em;color:var(--text-3);margin-bottom:12px;">
                    ChatTMC &middot; {len(get_indexed_pdfs())} manuals indexed
                  </div>
                  <h1 style="font-size:1.75rem;font-weight:600;letter-spacing:-0.02em;
                    color:var(--text);margin:0 0 12px;text-wrap:balance;">
                    Ask your technical library anything.
                  </h1>
                  <p style="font-size:14px;color:var(--text-2);max-width:480px;margin:0 auto 0;line-height:1.6;">
                    Grounded, cited answers from every manual, spec sheet, and service bulletin in your library.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding:16px 0 8px;'>"
                "<span style='font-family:IBM Plex Mono,monospace;font-size:11px;"
                "text-transform:uppercase;letter-spacing:0.12em;color:var(--text-3);'>ChatTMC</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Status bar ──────────────────────────────────────────────────────────
        filter_desc = " · ".join(f"{k}: {v}" for k, v in st.session_state.retrieval_filters.items())
        mode_label  = "Hybrid BM25+Vector" + (" + Rerank" if config.RERANKER_ENABLED else "")
        st.markdown(
            f"<div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap;"
            f"padding:8px 12px;background:var(--surface);border:1px solid var(--border);"
            f"border-radius:6px;margin-bottom:1.25rem;'>"
            f"<span class='mm-mono' style='color:var(--text-3);'>{total_chunks} chunks</span>"
            f"<span style='color:var(--border);'>|</span>"
            f"<span class='mm-mono' style='color:var(--text-3);'>{len(indexed_pdfs)} manual(s)</span>"
            f"<span style='color:var(--border);'>|</span>"
            f"<span class='mm-mono' style='color:var(--accent);'>{mode_label}</span>"
            f"<span style='color:var(--border);'>|</span>"
            f"<span class='mm-mono' style='color:var(--text-3);'>threshold {relevance_threshold:.2f}</span>"
            + (f"<span style='color:var(--border);'>|</span><span class='mm-mono' style='color:#f59e0b;'>{filter_desc}</span>" if filter_desc else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Controls ────────────────────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chat_history    = []
                st.session_state.pending_question = ""
                st.rerun()

        # ── Suggested prompts ───────────────────────────────────────────────────
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='font-family:IBM Plex Mono,monospace;font-size:10px;text-transform:uppercase;"
                "letter-spacing:0.12em;color:var(--text-3);text-align:center;margin-bottom:10px;'>Try asking</div>",
                unsafe_allow_html=True,
            )
            prompt_cols = st.columns(3)
            for i, prompt in enumerate(_SUGGESTED_PROMPTS[:3]):
                if prompt_cols[i].button(prompt, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.chat_input_draft = prompt
                    st.rerun()

        # ── Render history ──────────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            avatar = _AVATAR_USER if msg["role"] == "user" else _AVATAR_ASSISTANT
            with st.chat_message(msg["role"], avatar=avatar):
                if msg["role"] == "assistant" and msg.get("is_troubleshooting"):
                    st.markdown(
                        '<span class="mm-badge mm-badge-ts" style="margin-bottom:8px;display:inline-block;">'
                        '⚡ Troubleshooting</span>',
                        unsafe_allow_html=True,
                    )
                if msg["role"] == "assistant" and msg.get("chunks"):
                    _render_answer(
                        msg["content"],
                        msg["chunks"],
                        is_visual=msg.get("is_visual_query", False),
                        show_scores=show_scores,
                    )
                else:
                    st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("follow_ups"):
                    st.markdown(
                        "<div class='mm-label' style='margin-top:12px;margin-bottom:6px;'>Follow up</div>",
                        unsafe_allow_html=True,
                    )
                    fu_cols = st.columns(min(len(msg["follow_ups"]), 3))
                    for fi, fu in enumerate(msg["follow_ups"]):
                        if fu_cols[fi].button(fu, key=f"fu_{id(msg)}_{fi}", use_container_width=True):
                            st.session_state.pending_question = fu
                            st.rerun()

        # ── Input handling ──────────────────────────────────────────────────────
        question = None

        # Follow-up chips auto-submit immediately (different from suggested prompts)
        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""

        # Pre-fillable chat input — suggested prompt clicks populate this field
        if not question:
            with st.form(key="chat_form", clear_on_submit=True):
                in_col, btn_col = st.columns([11, 1])
                with in_col:
                    user_text = st.text_input(
                        "",
                        value=st.session_state.chat_input_draft,
                        placeholder="Ask anything across your indexed manuals…",
                        label_visibility="collapsed",
                    )
                with btn_col:
                    submitted = st.form_submit_button("↑", use_container_width=True)

            # Clear draft after it has been loaded into the widget
            if st.session_state.chat_input_draft:
                st.session_state.chat_input_draft = ""

            if submitted and user_text.strip():
                question = user_text.strip()

        if question:
            # Detect intent before adding to history
            intent     = classify_query(question)
            is_ts      = intent["is_troubleshooting"]
            is_visual  = intent["is_visual_query"]

            st.session_state.chat_history.append(
                {"role": "user", "content": question, "chunks": None, "follow_ups": []}
            )
            with st.chat_message("user", avatar=_AVATAR_USER):
                st.markdown(question)

            with st.chat_message("assistant", avatar=_AVATAR_ASSISTANT):
                if is_ts:
                    st.markdown(
                        '<span class="mm-badge mm-badge-ts" style="margin-bottom:8px;display:inline-block;">'
                        '⚡ Troubleshooting mode</span>',
                        unsafe_allow_html=True,
                    )

                # Query enrichment for short follow-ups
                past_user_qs = [
                    m["content"] for m in st.session_state.chat_history
                    if m["role"] == "user" and m["content"] != question
                ]
                retrieval_query = question
                if past_user_qs and len(question.split()) < 12:
                    retrieval_query = f"{past_user_qs[-1]} {question}"

                with st.spinner("Searching manuals…"):
                    chunks = retrieve(
                        retrieval_query,
                        top_k=config.FINAL_TOP_K,
                        min_score=relevance_threshold,
                        filters=st.session_state.retrieval_filters or None,
                    )

                if not chunks:
                    answer = (
                        "No supporting documentation found for this query above the current "
                        f"relevance threshold ({relevance_threshold:.2f}).  \n"
                        "Try lowering the threshold, rephrasing, or checking that the correct manual is indexed."
                    )
                    st.markdown(answer)
                    follow_ups = []
                    is_ts      = False
                else:
                    context  = build_context_prompt(chunks)
                    messages = build_chat_messages(
                        question,
                        context,
                        st.session_state.chat_history[:-1],
                        is_troubleshooting=is_ts,
                    )

                    system = _SYSTEM_PROMPT
                    if is_ts:
                        system += get_troubleshoot_system_addendum()

                    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

                    # Stream into a placeholder so we can replace it with
                    # progressive-disclosure / image-first rendering once complete.
                    stream_placeholder = st.empty()
                    collected: list[str] = []
                    with client.messages.stream(
                        model=config.ANTHROPIC_MODEL,
                        max_tokens=config.MAX_TOKENS,
                        system=system,
                        messages=messages,
                    ) as stream:
                        for text in stream.text_stream:
                            collected.append(text)
                            stream_placeholder.markdown(
                                "".join(collected) + "▌"
                            )

                    raw_answer = "".join(collected)
                    answer, follow_ups = _parse_follow_ups(raw_answer)

                    # Replace streamed text with final progressive-disclosure render
                    stream_placeholder.empty()
                    _render_answer(answer, chunks, is_visual, show_scores)

            st.session_state.chat_history.append({
                "role":                "assistant",
                "content":             answer,
                "chunks":              chunks or [],
                "follow_ups":          follow_ups,
                "is_troubleshooting":  is_ts,
                "is_visual_query":     is_visual,
            })
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Document Library
# ══════════════════════════════════════════════════════════════════════════════
with tab_manuals:
    # Handle cross-tab PDF navigation target from chat citations
    nav_target = st.session_state.get("pdf_view_target")
    if nav_target:
        nf = nav_target.get("source_pdf", "")
        np = nav_target.get("page", 1)
        st.info(
            f"**Citation jump:** Opening `{nf}` at page {np}.  "
            "The viewer is highlighted below.",
            icon="📄",
        )
        # Auto-open viewer for this manual
        view_key = f"view_pdf_{nf}"
        st.session_state[view_key] = True

    indexed_pdfs = get_indexed_pdfs()
    st.markdown(
        f"""
        <div style="padding:28px 0 20px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
            text-transform:uppercase;letter-spacing:0.12em;color:var(--text-3);margin-bottom:8px;">
            Document Library
          </div>
          <h1 style="font-size:1.4rem;font-weight:600;letter-spacing:-0.02em;
            color:var(--text);margin:0 0 4px;">{len(indexed_pdfs)} manuals &middot; {len(get_manufacturers())} manufacturers</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not indexed_pdfs:
        st.markdown(
            "<div style='text-align:center;padding:64px 0;'>"
            "<div style='width:52px;height:52px;border-radius:12px;background:var(--accent-10);"
            "border:1.5px solid var(--accent-30);display:flex;align-items:center;justify-content:center;"
            "margin:0 auto 16px;font-size:24px;'>📚</div>"
            "<div style='font-size:15px;font-weight:600;color:var(--text);margin-bottom:6px;'>No manuals indexed</div>"
            "<div style='font-size:13px;color:var(--text-2);'>Upload PDFs in <b>File Upload</b>.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(3)
        for i, entry in enumerate(indexed_pdfs):
            accent, bg   = _COVER_PALETTES[i % len(_COVER_PALETTES)]
            source_pdf   = entry["source_pdf"]
            manufacturer = entry.get("manufacturer", "")
            product_line = entry.get("product_line", "")
            doc_type     = entry.get("doc_type", "")
            revision     = entry.get("revision", "")
            title        = product_line or source_pdf
            subtitle     = manufacturer or doc_type or ""
            meta_tags    = " · ".join(filter(None, [doc_type, revision]))
            chunk_label  = f"{entry['text_chunks']} text · {entry['image_chunks']} img"
            pdf_path     = config.UPLOADS_DIR / source_pdf
            pdf_on_disk  = pdf_path.exists()

            # Highlight card if it's the current navigation target
            is_nav = nav_target and nav_target.get("source_pdf") == source_pdf
            card_border = f"border:2px solid {accent};" if is_nav else "border:1px solid var(--border);"

            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div class="tmc-manual-card" style="{card_border}">
                      <div style="height:6px;background:{accent};"></div>
                      <div style="padding:16px;background:repeating-linear-gradient(
                        135deg,{bg} 0px,{bg} 10px,
                        rgba(255,255,255,0.01) 10px,rgba(255,255,255,0.01) 20px);">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;
                          text-transform:uppercase;letter-spacing:0.12em;
                          color:{accent};margin-bottom:6px;">{subtitle or "Manual"}</div>
                        <div style="font-size:14px;font-weight:600;color:var(--text);
                          line-height:1.3;margin-bottom:8px;">{title}</div>
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                          color:var(--text-3);">{meta_tags or source_pdf}</div>
                      </div>
                      <div style="padding:8px 16px;border-top:1px solid var(--border);
                        display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                          color:var(--text-3);">{chunk_label}</span>
                        {'<span style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4ade80;">● PDF saved</span>' if pdf_on_disk else '<span style="font-family:IBM Plex Mono,monospace;font-size:10px;color:var(--text-3);">re-parse to enable download</span>'}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_cols = st.columns(2) if pdf_on_disk else st.columns(1)

                if pdf_on_disk:
                    btn_cols[0].download_button(
                        "Download PDF",
                        data=pdf_path.read_bytes(),
                        file_name=source_pdf,
                        mime="application/pdf",
                        key=f"dl_{source_pdf}",
                        use_container_width=True,
                    )
                    view_key = f"view_pdf_{source_pdf}"
                    if view_key not in st.session_state:
                        st.session_state[view_key] = False
                    if btn_cols[1].button(
                        "Hide PDF ▲" if st.session_state[view_key] else "View PDF ▼",
                        key=f"viewbtn_{source_pdf}",
                        use_container_width=True,
                    ):
                        st.session_state[view_key] = not st.session_state[view_key]
                        if not st.session_state[view_key]:
                            # Clear nav target when closing the viewer
                            if nav_target and nav_target.get("source_pdf") == source_pdf:
                                st.session_state.pdf_view_target = None
                        st.rerun()

                remove_col = st.columns([1])[0]
                if remove_col.button("Remove from index", key=f"del_{source_pdf}", use_container_width=True):
                    n = delete_pdf_from_index(source_pdf)
                    st.success(f"Removed {n} chunks for **{source_pdf}**.")
                    if nav_target and nav_target.get("source_pdf") == source_pdf:
                        st.session_state.pdf_view_target = None
                    st.rerun()

        # ── Inline PDF viewers ─────────────────────────────────────────────────
        st.divider()
        for entry in indexed_pdfs:
            source_pdf = entry["source_pdf"]
            view_key   = f"view_pdf_{source_pdf}"
            if st.session_state.get(view_key, False):
                pdf_path = config.UPLOADS_DIR / source_pdf
                if pdf_path.exists():
                    # Determine jump page
                    jump_page = 1
                    if nav_target and nav_target.get("source_pdf") == source_pdf:
                        jump_page = nav_target.get("page", 1)

                    nav_note = ""
                    if jump_page > 1:
                        nav_note = (
                            f"<span style='color:var(--accent);font-family:IBM Plex Mono,monospace;"
                            f"font-size:11px;'> → page {jump_page}</span>"
                        )

                    st.markdown(
                        f"<div style='font-size:13px;font-weight:600;margin-bottom:8px;'>"
                        f"{source_pdf}{nav_note}</div>",
                        unsafe_allow_html=True,
                    )
                    b64 = base64.b64encode(pdf_path.read_bytes()).decode()
                    # #page= fragment instructs browser PDF plugin to jump to page
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}#page={jump_page}" '
                        f'width="100%" height="900px" '
                        f'style="border:1px solid var(--border);border-radius:6px;"></iframe>',
                        unsafe_allow_html=True,
                    )
                    # Dismiss navigation target after rendering
                    if nav_target and nav_target.get("source_pdf") == source_pdf:
                        if st.button("Clear navigation target", key=f"clear_nav_{source_pdf}"):
                            st.session_state.pdf_view_target = None
                            st.rerun()

        total_all = get_total_chunk_count()
        st.markdown(
            f"<span class='mm-mono' style='color:var(--text-3);'>"
            f"{total_all} total chunks · {len(indexed_pdfs)} manual(s)</span>",
            unsafe_allow_html=True,
        )
