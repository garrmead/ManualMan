# utils/vision.py
#
# Sends extracted pump manual images to Claude's vision API to get:
#   1. A controlled-vocabulary type label (performance_curve, dimensional_drawing, etc.)
#   2. A detailed text description suitable for embedding and semantic search
#
# These descriptions become searchable "image chunks" in ChromaDB, so users
# can find diagrams by asking natural-language questions about their content.

import base64
from pathlib import Path

import anthropic

import config

# ── Controlled vocabulary ─────────────────────────────────────────────────────
# DECISION: These 8 types cover the vast majority of images in pump manuals.
# If you encounter a type that doesn't fit, it falls back to "other" and is
# still indexed — just less precisely categorized.
IMAGE_TYPES = {
    "performance_curve":    "Head/flow performance or characteristic curve",
    "dimensional_drawing":  "Dimensioned drawing with measurements and tolerances",
    "cross_section":        "Cross-sectional or cutaway internal view",
    "parts_list":           "Parts list, bill of materials, or parts diagram",
    "installation_diagram": "Installation, assembly, or mounting diagram",
    "nameplate_data":       "Nameplate, rating plate, or data table",
    "wiring_diagram":       "Wiring, electrical, or control schematic",
    "specifications_table": "Specifications, selection, or performance table",
    "other":                "Other image type",
}

# Human-readable labels shown in the UI
IMAGE_TYPE_LABELS = {
    "performance_curve":    "Performance Curve",
    "dimensional_drawing":  "Dimensional Drawing",
    "cross_section":        "Cross-Section View",
    "parts_list":           "Parts List",
    "installation_diagram": "Installation Diagram",
    "nameplate_data":       "Nameplate / Data Table",
    "wiring_diagram":       "Wiring Diagram",
    "specifications_table": "Specifications Table",
    "other":                "Other",
}

_MEDIA_TYPES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "gif":  "image/gif",
    "webp": "image/webp",
    "bmp":  "image/png",   # ChromaDB extracts some BMP as PNG bytes
}

_CLASSIFICATION_PROMPT = """\
You are analyzing an image extracted from a pump equipment manual.
Your job is to classify it and write a detailed description for a search index.

Respond in EXACTLY this two-line format — no extra text:
TYPE: <one of: performance_curve, dimensional_drawing, cross_section, parts_list, installation_diagram, nameplate_data, wiring_diagram, specifications_table, other>
DESCRIPTION: <2-5 sentences. Include ALL visible technical details: model numbers, pump sizes, \
flow rates (GPM/m³/h), head values (ft/m), pressures (PSI/bar), impeller diameters, shaft sizes, \
RPM, dimensions (inches/mm), material callouts, connection sizes, or any other specs shown. \
Be specific enough that someone searching for those specs can find this image.>\
"""


def classify_image(image_path: str, source_pdf: str, page_number: int) -> dict | None:
    """
    Send one image to Claude vision and return a classification dict.

    Returns None if the file doesn't exist or the API call fails.

    Returned dict keys:
        image_path   — original file path (str)
        image_type   — vocabulary key, e.g. "performance_curve"
        description  — detailed text description for embedding
        source_pdf   — original PDF filename
        page_number  — 1-based page number
    """
    path = Path(image_path)
    if not path.exists():
        return None

    # Read image and base64-encode it for the Anthropic messages API
    image_bytes = path.read_bytes()
    image_b64   = base64.standard_b64encode(image_bytes).decode("utf-8")
    ext         = path.suffix.lstrip(".").lower()
    media_type  = _MEDIA_TYPES.get(ext, "image/png")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    try:
        message = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=400,
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
                        {
                            "type": "text",
                            "text": (
                                f"This image is from pump manual: {source_pdf}, page {page_number}.\n\n"
                                + _CLASSIFICATION_PROMPT
                            ),
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        # Don't crash the whole commit if one image fails — log and skip
        print(f"[vision] Failed to classify {image_path}: {e}")
        return None

    response = message.content[0].text.strip()

    # Parse the two-line structured response
    image_type  = "other"
    description = response   # fallback: use full response if parsing fails

    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("TYPE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in IMAGE_TYPES:
                image_type = raw
        elif line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()

    return {
        "image_path":   image_path,
        "image_type":   image_type,
        "description":  description,
        "source_pdf":   source_pdf,
        "page_number":  page_number,
    }


def classify_images_batch(
    image_entries: list[dict],
    progress_cb=None,
) -> list[dict]:
    """
    Classify a list of images sequentially, calling progress_cb(frac, msg) if provided.
    Each entry must have: image_path, source_pdf, page_number.
    Returns only the entries that were successfully classified (skips None results).
    """
    results = []
    total   = len(image_entries)

    for i, entry in enumerate(image_entries):
        if progress_cb:
            progress_cb(
                i / total,
                f"Classifying image {i + 1} of {total}: {Path(entry['image_path']).name}…",
            )
        result = classify_image(
            entry["image_path"],
            entry["source_pdf"],
            entry["page_number"],
        )
        if result:
            results.append(result)

    return results
