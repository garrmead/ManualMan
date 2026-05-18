# utils/pdf_parser.py
#
# Handles all PDF work: opening the file, pulling text page-by-page,
# splitting it into chunks, and saving embedded images to disk.
# Everything here is pure extraction — no embeddings, no API calls.

import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF — imported as "fitz" (historical name)

import config

# ── Constants ─────────────────────────────────────────────────────────────────
# DECISION: Rough chars-per-token estimate for English technical text.
# 4 chars/token works well for most manuals. Adjust in config if chunks
# feel too large or too small.
_CHARS_PER_TOKEN = 4

# DECISION: Skip images smaller than 100×100 px — they're almost always
# logos, watermarks, bullet icons, or PDF artifacts rather than diagrams.
_MIN_IMAGE_PX = 100

# DECISION: Skip chunks with fewer than 40 characters — they're usually
# isolated page numbers, headers like "Section 3", or blank lines that
# crept through the text extraction.
_MIN_CHUNK_CHARS = 40


# ── Public API ────────────────────────────────────────────────────────────────

def extract_chunks_from_pdf(pdf_bytes: bytes, pdf_name: str) -> list[dict]:
    """
    Main entry point. Given raw PDF bytes and the original filename, return
    a list of chunk dicts ready for the editable preview table.

    Each chunk dict has these keys:
        chunk_id    — unique string (UUIDv4), used as the ChromaDB document ID
        source_pdf  — original filename (e.g. "Goulds_3196.pdf")
        page_number — 1-based page number the text came from
        text        — the extracted/editable chunk text
        tags        — empty string; user fills this in the UI
        image_paths — list of file paths for images on the same page
        keep        — True by default; user unchecks to exclude from indexing
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1  # 1-based for display

        # Pull plain text — "text" mode preserves reading order reasonably well
        page_text = page.get_text("text")

        # Save any usable images found on this page
        image_paths = _extract_page_images(doc, page, page_num, pdf_name)

        # Split the page text into overlapping chunks
        chunk_chars = config.CHUNK_SIZE * _CHARS_PER_TOKEN
        overlap_chars = config.CHUNK_OVERLAP * _CHARS_PER_TOKEN
        page_chunks = _split_text(page_text, chunk_chars, overlap_chars)

        for raw_text in page_chunks:
            text = raw_text.strip()
            if len(text) < _MIN_CHUNK_CHARS:
                continue
            chunks.append({
                "chunk_id":    str(uuid.uuid4()),
                "source_pdf":  pdf_name,
                "page_number": page_num,
                "text":        text,
                "tags":        "",
                "image_paths": image_paths,
                "keep":        True,
            })

    doc.close()
    return chunks


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF without extracting anything."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_chars length.
    Tries to break at sentence boundaries (". ") rather than mid-sentence.
    Returns an empty list for blank/whitespace-only pages.
    """
    if not text.strip():
        return []

    chunks = []
    text_len = len(text)
    start = 0

    while start < text_len:
        end = min(start + chunk_chars, text_len)

        # If not at the end of the text, nudge the cut point to a sentence end
        if end < text_len:
            # Look backwards through the last ~20% of the chunk for ". "
            search_from = max(start, end - chunk_chars // 5)
            boundary = text.rfind(". ", search_from, end)
            if boundary != -1:
                end = boundary + 1   # keep the period, cut before the space

        chunks.append(text[start:end])

        # Advance by chunk_chars minus overlap so consecutive chunks share context
        next_start = end - overlap_chars
        # Guard against infinite loops if overlap >= chunk size
        start = next_start if next_start > start else end

    return chunks


def _extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    pdf_name: str,
) -> list[str]:
    """
    Extract images from a single page, write them to data/extracted_images/,
    and return a list of file paths (strings).

    Images smaller than _MIN_IMAGE_PX in either dimension are skipped.
    Images that already exist on disk are not re-extracted (idempotent).
    """
    saved_paths = []

    # Build a safe directory name from the PDF filename (strip extension, replace
    # any non-alphanumeric characters with underscores)
    safe_stem = re.sub(r"[^\w\-]", "_", Path(pdf_name).stem)
    image_dir = config.IMAGES_DIR / safe_stem
    image_dir.mkdir(parents=True, exist_ok=True)

    for img_index, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]   # cross-reference ID for the image object in the PDF
        try:
            base_image = doc.extract_image(xref)
            width  = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < _MIN_IMAGE_PX or height < _MIN_IMAGE_PX:
                continue   # too small — probably a decoration

            ext      = base_image["ext"]   # e.g. "png", "jpeg"
            filename = f"p{page_num:03d}_img{img_index:02d}.{ext}"
            img_path = image_dir / filename

            if not img_path.exists():
                img_path.write_bytes(base_image["image"])

            saved_paths.append(str(img_path))

        except Exception:
            # Some PDF objects look like images but aren't extractable; skip them
            continue

    return saved_paths
