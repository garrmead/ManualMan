# utils/entity_extractor.py
#
# Extracts structured engineering entities from chunk text during ingestion.
# Results are stored as ChromaDB metadata so chunks can be found by exact
# attribute values (model number, material, RPM, etc.) via metadata filters.
#
# Uses Claude Haiku for speed and cost efficiency.

import json
import re
from typing import Optional

import anthropic

import config

_ENTITY_PROMPT = """\
Extract engineering entities from this pump/industrial equipment text.
Return ONLY a JSON object with these keys (omit keys with no value found):

{
  "model_numbers": ["3196-LTX", "2x3-10"],
  "part_numbers":  ["P/N 12345", "400-123"],
  "rpm":           ["1750", "3500"],
  "flow_gpm":      ["150 GPM", "200-400 GPM"],
  "head_ft":       ["120 ft", "50-200 ft"],
  "pressure_psi":  ["150 PSI", "300 PSI"],
  "hp":            ["15 HP", "25 HP"],
  "materials":     ["316SS", "Cast Iron", "Bronze"],
  "seal_types":    ["mechanical seal", "packing"],
  "bearing_types": ["ball bearing", "sleeve bearing"],
  "frame_sizes":   ["ST frame", "MTI frame"],
  "dimensions":    ["12.5 in", "300 mm bore"],
  "temperatures":  ["250°F max", "-20 to 250°F"],
  "is_troubleshooting": false
}

Rules:
- Extract ONLY values explicitly present in the text
- "is_troubleshooting": true if text contains symptom-cause-remedy content, fault codes, or alarm descriptions
- Return valid JSON only, no markdown fences, no explanation

TEXT:
"""


def extract_entities(text: str) -> dict:
    """
    Extract structured entities from a text chunk.
    Returns a flat dict suitable for ChromaDB metadata (string values only).
    Returns {} if extraction fails or yields nothing useful.
    """
    if len(text) < 50:
        return {}

    try:
        client  = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=config.ENTITY_MODEL,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": _ENTITY_PROMPT + text[:2000],
                }
            ],
        )
        raw = message.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception:
        return {}

    # Flatten lists to comma-separated strings for ChromaDB metadata compatibility
    flat: dict[str, str] = {}
    for key, val in parsed.items():
        if isinstance(val, list) and val:
            flat[key] = ", ".join(str(v) for v in val[:8])  # cap length
        elif isinstance(val, bool):
            flat[key] = "true" if val else "false"
        elif val:
            flat[key] = str(val)

    return flat


def is_troubleshooting_text(text: str) -> bool:
    """
    Quick heuristic check for troubleshooting content before calling the API.
    Used to avoid unnecessary API calls for clearly non-troubleshooting text.
    """
    ts_keywords = [
        "symptom", "cause", "remedy", "corrective action", "fault",
        "alarm", "error code", "troubleshoot", "excessive vibration",
        "seal leak", "overheating", "low flow", "no flow", "cavitation",
        "bearing failure", "motor overload", "check valve",
    ]
    lower = text.lower()
    return sum(1 for kw in ts_keywords if kw in lower) >= 2
