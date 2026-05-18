# config.py — Central place for all tunable settings.
# If you want to experiment, change values here — no hunting through code.

from pathlib import Path
import os

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
UPLOADS_DIR     = DATA_DIR / "uploads"
IMAGES_DIR      = DATA_DIR / "extracted_images"
CHROMA_DIR      = DATA_DIR / "chroma_db"

# ── Chunking ──────────────────────────────────────────────────────────────────
# DECISION: 512 tokens with 64 overlap is a solid starting point for technical
# manuals. If answers feel incomplete, raise CHUNK_SIZE to 1024. If they feel
# unfocused or off-topic, drop it to 256.
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Retrieval ─────────────────────────────────────────────────────────────────
# DECISION: Top 5 chunks per query. Raise to 8 if answers miss important
# context; lower to 3 if responses feel padded or you hit token-limit errors.
TOP_K = 5

# ── Embeddings ────────────────────────────────────────────────────────────────
VOYAGE_MODEL        = "voyage-3"   # Voyage AI embedding model
EMBEDDING_DIMENSION = 1024         # voyage-3 output size (don't change)

# ── Generation model ──────────────────────────────────────────────────────────
# DECISION: claude-sonnet-4-6 balances quality and cost well for this use case.
# Swap to claude-opus-4-7 for harder questions; claude-haiku-4-5-20251001 if
# you want faster/cheaper responses.
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_TOKENS      = 2048   # Maximum length of each generated answer

# ── Vector store ──────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "pump_manuals"   # Name of the ChromaDB collection

# ── API keys (loaded from .env) ───────────────────────────────────────────────
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
