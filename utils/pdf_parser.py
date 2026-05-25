# utils/pdf_parser.py
#
# Hybrid PDF parser:
#   - pdfplumber  → text (excluding table regions) + structured table extraction
#   - PyMuPDF     → image extraction only
#
# Why two libraries? PyMuPDF is fast and reliable for images. pdfplumber has
# purpose-built table detection that preserves column structure — critical for
# pump selection tables, dimensional tables, and performance data that PyMuPDF
# would extract as garbled single-column text.

import io
import re
import uuid
from pathlib import Path

import fitz        # PyMuPDF
import pdfplumber

import config

# ── Constants ─────────────────────────────────────────────────────────────────
_CHARS_PER_TOKEN = 4      # rough chars/token for English technical text
_MIN_IMAGE_PX    = 100    # skip images smaller than this in either dimension
_MIN_CHUNK_CHARS = 40     # skip near-empty text chunks (headers, page numbers)
_MIN_TABLE_ROWS  = 2      # skip single-row "tables" (often just a styled box)
_MIN_TABLE_COLS  = 2      # skip single-column "tables" (often just a list)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_chunks_from_pdf(pdf_bytes: bytes, pdf_name: str) -> list[dict]:
    """
    Extract text chunks, table chunks, and image paths from a PDF.

    Returns a list of chunk dicts, each with:
        chunk_id      — unique ID (UUIDv4)
        source_pdf    — original filename
        page_number   — 1-based page number
        text          — chunk text (plain for text chunks, Markdown for tables)
        tags          — "table" for table chunks, "" for text chunks
        image_paths   — images extracted from the same page
        keep          — True (user can uncheck in the editor)
        content_type  — "text" or "table" (shown as a badge in the editor)
    """
    chunks = []

    # Open with both libraries from the same bytes
    fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
            for page_idx in range(len(fitz_doc)):
                page_num     = page_idx + 1
                fitz_page    = fitz_doc[page_idx]
                plumber_page = plumber_doc.pages[page_idx]

                # ── Images (PyMuPDF only) ──────────────────────────────────────
                image_paths = _extract_page_images(fitz_doc, fitz_page, page_num, pdf_name)

                # ── Tables (pdfplumber) ────────────────────────────────────────
                table_bboxes = []
                try:
                    found_tables = plumber_page.find_tables()
                    for tbl in found_tables:
                        data     = tbl.extract()
                        markdown = _table_to_markdown(data)
                        if not markdown:
                            continue
                        table_bboxes.append(tbl.bbox)
                        chunks.append({
                            "chunk_id":     str(uuid.uuid4()),
                            "source_pdf":   pdf_name,
                            "page_number":  page_num,
                            "text":         markdown,
                            "tags":         "table",
                            "image_paths":  image_paths,
                            "keep":         True,
                            "content_type": "table",
                        })
                except Exception:
                    # If pdfplumber can't read tables on this page, skip tables
                    # and fall through to plain text extraction
                    pass

                # ── Non-table text (pdfplumber with table regions filtered) ────
                try:
                    if table_bboxes:
                        # Filter out characters that fall inside any table bbox
                        # so table content doesn't appear twice (once as Markdown
                        # table chunk, once as garbled text chunk).
                        filtered = plumber_page.filter(
                            lambda obj: not _in_any_bbox(obj, table_bboxes)
                        )
                        page_text = filtered.extract_text() or ""
                    else:
                        page_text = plumber_page.extract_text() or ""
                except Exception:
                    # Fall back to PyMuPDF text if pdfplumber fails
                    page_text = fitz_page.get_text("text")

                chunk_chars   = config.CHUNK_SIZE * _CHARS_PER_TOKEN
                overlap_chars = config.CHUNK_OVERLAP * _CHARS_PER_TOKEN

                for raw in _split_text(page_text, chunk_chars, overlap_chars):
                    text = raw.strip()
                    if len(text) < _MIN_CHUNK_CHARS:
                        continue
                    chunks.append({
                        "chunk_id":     str(uuid.uuid4()),
                        "source_pdf":   pdf_name,
                        "page_number":  page_num,
                        "text":         text,
                        "tags":         "",
                        "image_paths":  image_paths,
                        "keep":         True,
                        "content_type": "text",
                    })
    finally:
        fitz_doc.close()

    return chunks


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Return page count without extracting anything."""
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


# ── Internal helpers ──────────────────────────────────────────────────────────

def _in_any_bbox(obj: dict, bboxes: list[tuple]) -> bool:
    """
    Return True if a pdfplumber character/word object falls inside any of the
    given bounding boxes. Used to filter table regions from text extraction.

    pdfplumber bbox format: (x0, top, x1, bottom) — top measured from page top.
    Object attributes use the same coordinate names.
    """
    ox0 = obj.get("x0",    0)
    ox1 = obj.get("x1",    0)
    ot  = obj.get("top",   0)
    ob  = obj.get("bottom", 0)
    for (bx0, bt, bx1, bb) in bboxes:
        # Small tolerance (2pt) to catch chars right on the table border
        if ox0 >= bx0 - 2 and ox1 <= bx1 + 2 and ot >= bt - 2 and ob <= bb + 2:
            return True
    return False


def _table_to_markdown(table_data: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows, each row a list of cell strings)
    into a GitHub-flavoured Markdown table string.

    Returns an empty string if the table doesn't meet minimum size thresholds
    or has no actual content.
    """
    if not table_data:
        return ""

    # Clean cells: None → "", strip whitespace, collapse internal newlines
    rows = []
    for row in table_data:
        cleaned = [
            re.sub(r"\s+", " ", str(cell).strip()) if cell is not None else ""
            for cell in row
        ]
        rows.append(cleaned)

    # Drop entirely empty rows
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""

    # Enforce minimum dimensions
    n_cols = max(len(r) for r in rows)
    if len(rows) < _MIN_TABLE_ROWS or n_cols < _MIN_TABLE_COLS:
        return ""

    # Pad all rows to the same column count
    for row in rows:
        while len(row) < n_cols:
            row.append("")

    # Escape pipe characters inside cells to avoid breaking Markdown syntax
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


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if not text.strip():
        return []

    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_chars, length)
        if end < length:
            boundary = text.rfind(". ", max(start, end - chunk_chars // 5), end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end])
        next_start = end - overlap_chars
        start = next_start if next_start > start else end

    return chunks


def _extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    pdf_name: str,
) -> list[str]:
    """Extract images from a page, save to disk, return file paths."""
    saved = []
    safe_stem = re.sub(r"[^\w\-]", "_", Path(pdf_name).stem)
    image_dir = config.IMAGES_DIR / safe_stem
    image_dir.mkdir(parents=True, exist_ok=True)

    for img_index, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base  = doc.extract_image(xref)
            w, h  = base.get("width", 0), base.get("height", 0)
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

    return saved
