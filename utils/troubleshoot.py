# utils/troubleshoot.py
#
# Query intent detection for troubleshooting mode.
# When a user's question matches known failure-mode patterns, the retrieval
# pipeline boosts troubleshooting-tagged chunks and the system prompt is
# adjusted for root-cause / corrective-action output.

import re

# ── Single-word symptom indicators ───────────────────────────────────────────
_SYMPTOM_WORDS = {
    "vibration", "vibrating", "shaking", "noise", "noisy", "rattling",
    "cavitation", "cavitating", "leak", "leaking", "leaks",
    "overheat", "overheating", "hot",
    "failure", "failed", "failing",
    "tripping", "trips", "overload",
    "wear", "worn", "corrosion", "erosion",
    "surge", "pulsation",
}

# ── Multi-word phrases checked against the raw query string ──────────────────
_SYMPTOM_PHRASES = [
    "seal leak", "seal leaking", "mechanical seal",
    "low flow", "no flow", "loss of prime", "won't prime", "not pumping",
    "low pressure", "no pressure", "pressure drop",
    "motor overload", "motor overheating", "motor tripping",
    "bearing failure", "bearing noise", "bearing hot",
    "excessive vibration", "high vibration",
    "won't start", "not starting", "stopped working",
    "shaft broken", "coupling failure", "air bound", "air leak",
    "stuffing box", "packing leak",
    "impeller wear", "impeller damage",
]

# ── Action / diagnostic keywords ─────────────────────────────────────────────
_ACTION_WORDS = {
    "troubleshoot", "diagnose", "fix", "repair", "solve", "cause",
    "causes", "caused", "reason", "why", "check", "inspect",
    "correct", "remedy", "prevent", "issue", "problem",
}

_VISUAL_KEYWORDS = {
    "curve", "curves", "performance curve", "head curve", "efficiency curve",
    "dimensional", "dimensions", "drawing", "drawings", "diagram", "diagrams",
    "chart", "charts", "cross section", "cutaway", "exploded", "schematic",
    "wiring", "nameplate", "rating plate", "photo", "picture", "image",
    "show me", "show the", "display", "view",
}

_VISUAL_PHRASES = [
    "performance curve", "head curve", "dimensional drawing", "cross section",
    "parts diagram", "wiring diagram", "installation diagram", "show me the",
    "show the", "what does it look like", "pump photo",
]

_FAULT_CODE_PATTERN = re.compile(
    r"\b([A-Z]{1,4}[-_]?\d{2,6}|error\s+\d+|fault\s+\d+|alarm\s+\d+|code\s+\d+)\b",
    re.IGNORECASE,
)


def classify_query(question: str) -> dict:
    """
    Classify a user question into retrieval intent categories.

    Returns:
        is_troubleshooting: bool
        is_spec_lookup:     bool
        is_part_lookup:     bool
        fault_codes:        list[str]
        boost_keywords:     list[str]
    """
    lower = question.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    # Single-word matches
    symptom_hits = words & _SYMPTOM_WORDS
    action_hits  = words & _ACTION_WORDS
    fault_codes  = _FAULT_CODE_PATTERN.findall(question)

    # Multi-word phrase matches
    phrase_hits  = [p for p in _SYMPTOM_PHRASES if p in lower]

    is_troubleshooting = bool(
        fault_codes                         # any fault code
        or (symptom_hits and action_hits)   # symptom + action combo
        or len(symptom_hits) >= 2           # multiple symptoms
        or len(phrase_hits) >= 1            # any known failure phrase
        or any(phrase in lower for phrase in [
            "won't start", "not starting", "stopped working",
            "stopped pumping", "running hot", "making noise",
            "keeps tripping", "blowing fuse",
        ])
    )

    is_part_lookup = bool(
        re.search(r"\b(p/?n|part\s*#|part\s+number|bom|parts\s+list|spare\s+part)\b", lower)
        or re.search(r"\b\d{4,8}\b", question)
    )

    is_spec_lookup = bool(
        re.search(
            r"\b(gpm|rpm|psi|hp|horsepower|torque|ft-lb|nm|"
            r"impeller|diameter|bore|frame|temperature|pressure|flow|"
            r"max|minimum|rating|range|curve)\b",
            lower,
        )
        and not is_troubleshooting
    )

    boost_keywords = list(symptom_hits | action_hits) + fault_codes + phrase_hits

    visual_hits   = words & _VISUAL_KEYWORDS
    visual_phrases = [p for p in _VISUAL_PHRASES if p in lower]
    is_visual_query = bool(visual_hits or visual_phrases)

    return {
        "is_troubleshooting": is_troubleshooting,
        "is_spec_lookup":     is_spec_lookup,
        "is_part_lookup":     is_part_lookup,
        "is_visual_query":    is_visual_query,
        "fault_codes":        fault_codes,
        "boost_keywords":     boost_keywords,
    }


def get_troubleshoot_system_addendum() -> str:
    return (
        "\n\nThis is a TROUBLESHOOTING query. Structure your response as:\n"
        "**Likely Cause(s):** [list causes from the manual]\n"
        "**Recommended Action(s):** [numbered steps from the manual]\n"
        "Cite the manual section and page number for every action. "
        "If multiple causes are listed in the manual, present all of them."
    )
