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
IMAGE_TYPES = {
    "performance_curve":    "Head/flow performance or characteristic curve",
    "dimensional_drawing":  "Dimensioned drawing with measurements and tolerances",
    "cross_section":        "Cross-sectional or cutaway internal view",
    "parts_list":           "Parts list, bill of materials, or parts diagram",
    "installation_diagram": "Installation, assembly, or mounting diagram",
    "nameplate_data":       "Nameplate, rating plate, or data table",
    "wiring_diagram":       "Wiring, electrical, or control schematic",
    "specifications_table": "Specifications, selection, or performance table",
    # Non-technical — classified so we can skip indexing them
    "pump_photo":           "Photograph of a pump or equipment (not a technical drawing)",
    "logo_branding":        "Company logo, trademark, decorative graphic, or cover art",
    "other":                "Other image that doesn't fit the above categories",
}

# Types that are worth indexing and showing in chat.
# pump_photo, logo_branding, and other are intentionally excluded.
INDEXABLE_TYPES = {
    "performance_curve",
    "dimensional_drawing",
    "cross_section",
    "parts_list",
    "installation_diagram",
    "nameplate_data",
    "wiring_diagram",
    "specifications_table",
    "pump_photo",
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
    "pump_photo":           "Pump Photo",
    "logo_branding":        "Logo / Branding",
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
Classify it accurately using the type list below, then write a search-index description.

TYPE OPTIONS:
- performance_curve    — head/flow curve, efficiency curve, or pump characteristic curve (a technical graph)
- dimensional_drawing  — engineering drawing with dimensions and measurements
- cross_section        — cutaway or cross-sectional view showing internal parts
- parts_list           — exploded parts diagram or bill of materials
- installation_diagram — installation, assembly, or piping schematic
- nameplate_data       — data plate, rating table, or spec nameplate
- wiring_diagram       — electrical wiring or control schematic
- specifications_table — tabular specs, model selection table, or performance table
- pump_photo           — a PHOTOGRAPH of a pump or physical equipment (NOT a line drawing or graph)
- logo_branding        — company logo, trademark, decorative graphic, cover image, or page border
- other                — anything that doesn't fit the above

Respond in EXACTLY this two-line format — no extra text:
TYPE: <one type from the list above>
DESCRIPTION: <2-5 sentences. \
For PERFORMANCE CURVES: your FIRST sentence MUST state (a) whether this is a \
"single-speed [N] RPM performance curve" (one speed line) or a \
"composite multi-speed performance curve at [N1], [N2], [N3] RPM" (multiple speed lines), \
and (b) the pump size designation (e.g. "1.5x3-13", "3x4-13") or model/series name. \
Example first sentences: "Single-speed 1750 RPM performance curve for 3656 3x4-13 pump." \
or "Composite multi-speed performance curve at 1150, 1450, 1750 RPM for 3656 3x4-13 pump." \
Then add: flow range (GPM), head range (ft), impeller diameters shown (in), efficiency contours if visible. \
For OTHER image types: lead with pump size designation or model number, then describe what is shown. \
For photos/logos: one brief sentence is enough.>\
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
