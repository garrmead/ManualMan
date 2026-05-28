# utils/ocr.py
#
# OCR fallback for scanned or image-heavy PDF pages.
# Sends a rendered page image to Claude Vision and extracts machine-readable text.
# Called by pdf_parser when a page yields fewer than OCR_MIN_PAGE_CHARS of text.

import base64
from pathlib import Path

import anthropic

import config

_OCR_PROMPT = """\
This is a page from a pump or industrial equipment manual.
Extract ALL text exactly as printed, preserving:
- Section headings and numbering
- Table structure (use | separators for columns, --- for header separators)
- Lists and step numbering
- All specifications: numbers, units, model numbers, part numbers, dimensions

Output ONLY the extracted text — no preamble, no commentary, no "Here is the text:".
If the page is mostly graphical (a diagram, curve, or drawing with minimal text),
output any visible labels, axis titles, figure numbers, and captions only.\
"""


def ocr_page_image(image_path: str) -> str:
    """
    Send a rendered page image to Claude Vision and return extracted text.
    Returns an empty string on failure.
    """
    path = Path(image_path)
    if not path.exists():
        return ""

    ext = path.suffix.lstrip(".").lower()
    media_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    media_type = media_map.get(ext, "image/png")

    try:
        image_b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        client    = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message   = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       image_b64,
                            },
                        },
                        {"type": "text", "text": _OCR_PROMPT},
                    ],
                }
            ],
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"[ocr] Failed for {image_path}: {e}")
        return ""
