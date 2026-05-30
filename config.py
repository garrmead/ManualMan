# config.py — Central place for all tunable settings.

from pathlib import Path
import os

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
UPLOADS_DIR     = DATA_DIR / "uploads"
IMAGES_DIR      = DATA_DIR / "extracted_images"
CHROMA_DIR      = DATA_DIR / "chroma_db"
BM25_INDEX_FILE = DATA_DIR / "bm25_index.json"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens
CHUNK_OVERLAP = 64    # tokens

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K               = 8      # candidates from each retriever before reranking
FINAL_TOP_K         = 5      # results returned after reranking
MIN_RELEVANCE_SCORE = 0.35   # lowered slightly; reranker provides second-pass filtering

# Hybrid fusion weights (must sum to 1.0)
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT   = 0.4

# When the query contains explicit numbers (RPM, part numbers, torque values),
# flip the weights so exact-match BM25 leads over semantic vector search.
EXACT_QUERY_BM25_WEIGHT   = 0.65
EXACT_QUERY_VECTOR_WEIGHT = 0.35

# Score multiplier for image chunks on visual queries (curves, drawings, diagrams)
VISUAL_IMAGE_BOOST = 1.4

# Reciprocal Rank Fusion constant (higher → less steep score falloff)
RRF_K = 60

# ── Cross-encoder reranking ───────────────────────────────────────────────────
RERANKER_ENABLED = True
# HuggingFace model ID — downloaded lazily on first use (~80 MB)
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── OCR ───────────────────────────────────────────────────────────────────────
# Pages with fewer than this many chars trigger Claude Vision OCR fallback
OCR_MIN_PAGE_CHARS = 80

# ── Entity extraction ─────────────────────────────────────────────────────────
ENTITY_EXTRACTION_ENABLED = True
# Use haiku for entity extraction (fast, cheap, good enough for structured output)
ENTITY_MODEL = "claude-haiku-4-5-20251001"

# ── Troubleshooting mode ──────────────────────────────────────────────────────
TROUBLESHOOT_BOOST = 1.25   # multiply score for troubleshooting chunks on TS queries

# ── Conversation memory ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 3

# ── Embeddings ────────────────────────────────────────────────────────────────
VOYAGE_MODEL        = "voyage-3"
EMBEDDING_DIMENSION = 1024

# ── Generation ────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_TOKENS      = 2048

# ── Vector store ──────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "pump_manuals"

# ── API keys ──────────────────────────────────────────────────────────────────
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
