# utils/pdf_parser.py
#
# Hybrid PDF parser:
#   - pdfplumber  → text (excluding table regions) + structured table extraction
#   - PyMuPDF     → image extraction + full-page vector rendering
#   - Claude Vision OCR → fallback for scanned / image-only pages
#
# Section-aware chunking preserves engineering context by splitting at
# detected section headers rather than fixed character counts.

import io
import re
import uuid
from pathlib import Path

import fitz        # PyMuPDF
import pdfplumber

import config

# ── Constants ─────────────────────────────────────────────────────────────────
_CHARS_PER_TOKEN  = 4
_MIN_IMAGE_PX     = 100
_MIN_CHUNK_CHARS  = 40
_MIN_TABLE_ROWS   = 2
_MIN_TABLE_COLS   = 2
_MIN_DRAWINGS              = 15
_MIN_DRAWINGS_WITH_RASTER  = 40
_PAGE_RENDER_DPI           = 2.0

# ── Section header detection ──────────────────────────────────────────────────

# Patterns that mark the start of a new section (higher specificity first)
_HEADER_PATTERNS = [
    # Numbered sections: "1.", "1.1", "1.1.1", "A.", "A.1"
    re.compile(r"^(\d{1,2}\.){1,3}\s+[A-Z]"),
    re.compile(r"^[A-Z]\.\s+[A-Z]"),
    # Roman numeral sections
    re.compile(r"^(I{1,3}|IV|V?I{0,3}|IX|X{0,3})\.\s+[A-Z]"),
    # ALLCAPS headings (3+ words or 6+ chars, not sentences)
    re.compile(r"^[A-Z][A-Z\s\/\-&]{5,}$"),
    # Warning/caution/note blocks — treat as section boundary
    re.compile(r"^(WARNING|CAUTION|NOTE|DANGER|IMPORTANT)\s*:", re.IGNORECASE),
    # Troubleshooting table headers
    re.compile(r"^(TROUBLE|SYMPTOM|PROBLEM|FAULT|CAUSE|REMEDY|CORRECTIVE)", re.IGNORECASE),
]

_PROCEDURE_LINE = re.compile(r"^\s*(\d+)\.\s+\S")  # numbered step


def _is_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    for pat in _HEADER_PATTERNS:
        if pat.match(line):
            return True
    return False


def _detect_chunk_subtype(text: str) -> str:
    """Classify chunk content type for metadata and retrieval routing."""
    lower = text.lower()
    ts_signals = [
        "symptom", "cause", "remedy", "corrective action",
        "trouble", "fault", "alarm", "error code",
    ]
    if sum(1 for s in ts_signals if s in lower) >= 2:
        return "troubleshooting"

    proc_lines = _PROCEDURE_LINE.findall(text)
    if len(proc_lines) >= 3:
        return "procedure"

    if "|" in text and "---" in text:
        return "table"

    spec_signals = ["gpm", "rpm", "psi", "hp", " in.", " mm", " ft", "lbs", "°f", "°c"]
    if sum(1 for s in spec_signals if s in lower) >= 3:
        return "specification"

    return "text"


# ── Section-aware text splitting ──────────────────────────────────────────────

def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split page text into (section_title, body) tuples at detected headers.
    Returns [(title, body), ...]. First element may have an empty title
    if page starts with body text before any header.
    """
    lines   = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_body:  list[str] = []

    for line in lines:
        if _is_header(line.strip()) and current_body:
            body_text = "\n".join(current_body).strip()
            if body_text:
                sections.append((current_title, body_text))
            current_title = line.strip()
            current_body  = []
        else:
            current_body.append(line)

    # flush last section
    body_text = "\n".join(current_body).strip()
    if body_text:
        sections.append((current_title, body_text))

    return sections if sections else [("", text)]


def _split_section_body(body: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split a single section body into overlapping chunks at sentence boundaries."""
    if not body.strip():
        return []
    chunks = []
    start  = 0
    length = len(body)
    while start < length:
        end = min(start + chunk_chars, length)
        if end < length:
            boundary = body.rfind(". ", max(start, end - chunk_chars // 5), end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(body[start:end])
        next_start = end - overlap_chars
        start = next_start if next_start > start else end
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def extract_chunks_from_pdf(pdf_bytes: bytes, pdf_name: str) -> list[dict]:
    """
    Extract text chunks, table chunks, and image paths from a PDF.

    Chunk dict keys:
        chunk_id      — UUIDv4
        source_pdf    — original filename
        page_number   — 1-based
        text          — chunk content (plain text or Markdown table)
        tags          — "table" for table chunks, "" otherwise
        image_paths   — list of image file paths from this page
        keep          — True (user can uncheck)
        content_type  — "text" | "table"
        section_title — detected section header (may be "")
        chunk_subtype — "text" | "table" | "procedure" | "specification" | "troubleshooting"
        ocr_used      — True if Claude OCR was used for this page
    """
    chunks: list[dict] = []
    chunk_chars   = config.CHUNK_SIZE * _CHARS_PER_TOKEN
    overlap_chars = config.CHUNK_OVERLAP * _CHARS_PER_TOKEN

    fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
            for page_idx in range(len(fitz_doc)):
                page_num     = page_idx + 1
                fitz_page    = fitz_doc[page_idx]
                plumber_page = plumber_doc.pages[page_idx]

                # ── Images (PyMuPDF) ───────────────────────────────────────────
                image_paths = _extract_page_images(fitz_doc, fitz_page, page_num, pdf_name)

                # ── Tables (pdfplumber) ────────────────────────────────────────
                table_bboxes: list[tuple] = []
                try:
                    for tbl in plumber_page.find_tables():
                        markdown = _table_to_markdown(tbl.extract())
                        if not markdown:
                            continue
                        table_bboxes.append(tbl.bbox)
                        chunks.append(_make_chunk(
                            pdf_name, page_num, markdown, "table", image_paths,
                            section_title="", chunk_subtype="table", ocr_used=False,
                        ))
                except Exception:
                    pass

                # ── Text (pdfplumber, table regions filtered) ──────────────────
                ocr_used = False
                try:
                    if table_bboxes:
                        filtered  = plumber_page.filter(
                            lambda obj: not _in_any_bbox(obj, table_bboxes)
                        )
                        page_text = filtered.extract_text() or ""
                    else:
                        page_text = plumber_page.extract_text() or ""
                except Exception:
                    page_text = fitz_page.get_text("text")

                # ── Force page render if no images yet and page text is thin ──
                # Scanned PDFs have no vector drawings, so the drawing-count
                # threshold in _extract_page_images never fires. Explicitly
                # render the page so OCR has something to work with.
                if not image_paths and len(page_text.strip()) < config.OCR_MIN_PAGE_CHARS:
                    render_path = _force_render_page(fitz_page, page_num, pdf_name)
                    if render_path:
                        image_paths = [render_path]

                # ── OCR fallback for near-empty pages ─────────────────────────
                if len(page_text.strip()) < config.OCR_MIN_PAGE_CHARS and image_paths:
                    ocr_text = _ocr_page(image_paths[0])
                    if len(ocr_text) > len(page_text):
                        page_text = ocr_text
                        ocr_used  = True

                # ── Section-aware chunking ─────────────────────────────────────
                sections = _split_into_sections(page_text)
                for section_title, body in sections:
                    for raw in _split_section_body(body, chunk_chars, overlap_chars):
                        text = raw.strip()
                        if len(text) < _MIN_CHUNK_CHARS:
                            continue
                        # Prepend section header to each chunk so retrieval
                        # context includes the section name even for body chunks
                        full_text = f"{section_title}\n{text}".strip() if section_title else text
                        subtype   = _detect_chunk_subtype(full_text)
                        chunks.append(_make_chunk(
                            pdf_name, page_num, full_text, "", image_paths,
                            section_title=section_title,
                            chunk_subtype=subtype,
                            ocr_used=ocr_used,
                        ))
    finally:
        fitz_doc.close()

    return chunks


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_chunk(
    pdf_name: str,
    page_num: int,
    text: str,
    tags: str,
    image_paths: list[str],
    *,
    section_title: str,
    chunk_subtype: str,
    ocr_used: bool,
) -> dict:
    return {
        "chunk_id":      str(uuid.uuid4()),
        "source_pdf":    pdf_name,
        "page_number":   page_num,
        "text":          text,
        "tags":          tags,
        "image_paths":   image_paths,
        "keep":          True,
        "content_type":  "table" if tags == "table" else "text",
        "section_title": section_title,
        "chunk_subtype": chunk_subtype,
        "ocr_used":      ocr_used,
    }


def _ocr_page(image_path: str) -> str:
    """Call Claude Vision OCR; gracefully return '' if unavailable."""
    try:
        from utils.ocr import ocr_page_image
        return ocr_page_image(image_path)
    except Exception:
        return ""


def _force_render_page(page: fitz.Page, page_num: int, pdf_name: str) -> str:
    """Render a page to PNG unconditionally and return the file path (or '')."""
    try:
        safe_stem = re.sub(r"[^\w\-]", "_", Path(pdf_name).stem)
        image_dir = config.IMAGES_DIR / safe_stem
        image_dir.mkdir(parents=True, exist_ok=True)
        filename  = f"p{page_num:03d}_page.png"
        path      = image_dir / filename
        if not path.exists():
            mat = fitz.Matrix(_PAGE_RENDER_DPI, _PAGE_RENDER_DPI)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(path))
        return str(path)
    except Exception:
        return ""


def _in_any_bbox(obj: dict, bboxes: list[tuple]) -> bool:
    ox0 = obj.get("x0", 0)
    ox1 = obj.get("x1", 0)
    ot  = obj.get("top", 0)
    ob  = obj.get("bottom", 0)
    for (bx0, bt, bx1, bb) in bboxes:
        if ox0 >= bx0 - 2 and ox1 <= bx1 + 2 and ot >= bt - 2 and ob <= bb + 2:
            return True
    return False


def _table_to_markdown(table_data: list[list]) -> str:
    if not table_data:
        return ""
    rows = []
    for row in table_data:
        cleaned = [
            re.sub(r"\s+", " ", str(cell).strip()) if cell is not None else ""
            for cell in row
        ]
        rows.append(cleaned)
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    if len(rows) < _MIN_TABLE_ROWS or n_cols < _MIN_TABLE_COLS:
        return ""
    for row in rows:
        while len(row) < n_cols:
            row.append("")

    def _escape(cell: str) -> str:
        return cell.replace("|", "\\|")

    header    = [_escape(c) for c in rows[0]]
    body_rows = [[_escape(c) for c in r] for r in rows[1:]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    pdf_name: str,
) -> list[str]:
    saved = []
    safe_stem = re.sub(r"[^\w\-]", "_", Path(pdf_name).stem)
    image_dir = config.IMAGES_DIR / safe_stem
    image_dir.mkdir(parents=True, exist_ok=True)

    # Strategy 1: embedded raster images
    for img_index, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base = doc.extract_image(xref)
            w, h = base.get("width", 0), base.get("height", 0)
            if w < _MIN_IMAGE_PX or h < _MIN_IMAGE_PX:
                continue
            ext      = base["ext"]
            filename = f"p{page_num:03d}_img{img_index:02d}.{ext}"
            path     = image_dir / filename
            if not path.exists():
                path.write_bytes(base["image"])
            saved.append(str(path))
        except Exception:
            continue

    # Strategy 2: full-page render for vector-heavy pages
    try:
        n_drawings = len(page.get_drawings())
        threshold  = _MIN_DRAWINGS if not saved else _MIN_DRAWINGS_WITH_RASTER
        if n_drawings >= threshold:
            filename = f"p{page_num:03d}_page.png"
            path     = image_dir / filename
            if not path.exists():
                mat = fitz.Matrix(_PAGE_RENDER_DPI, _PAGE_RENDER_DPI)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(str(path))
            saved.append(str(path))
    except Exception:
        pass

    return saved
