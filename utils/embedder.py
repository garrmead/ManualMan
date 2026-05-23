# utils/embedder.py
#
# Handles all vector-store work: embedding chunks via Voyage AI and
# upserting them into ChromaDB. Also provides helpers for listing and
# deleting indexed PDFs (used by the Manuals tab in Phase 5).
#
# Design choice: we use the raw voyageai and chromadb clients here rather
# than going through LlamaIndex. This makes the data flow explicit and easy
# to follow — you can see exactly what gets sent to Voyage and what gets
# stored in ChromaDB. Phase 4 will add LlamaIndex on top for querying.

import json
import time
from typing import Callable

import chromadb
import voyageai

import config

# ── Constants ─────────────────────────────────────────────────────────────────
# DECISION: Voyage AI allows up to 128 texts per embed call, but large PDFs
# can push against the per-request token limit. 32 is a safe batch size that
# keeps progress updates frequent and avoids timeout errors.
_EMBED_BATCH_SIZE = 32

# Seconds to wait between batches to stay inside Voyage AI's rate limits.
# At 32 chunks/batch this rarely triggers, but it's a good safety net.
_RATE_LIMIT_PAUSE = 0.5


# ── ChromaDB client ───────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    """
    Return (or create) the persistent ChromaDB collection.
    PersistentClient writes to disk so the index survives restarts.
    cosine similarity is standard for semantic search with normalized embeddings.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Main public functions ─────────────────────────────────────────────────────

def commit_chunks(
    edited_df,           # pandas DataFrame from st.data_editor
    parsed_chunks: list[dict],
    progress_cb: Callable[[float, str], None] | None = None,
) -> int:
    """
    Embed every 'keep' chunk from edited_df and upsert it into ChromaDB.

    Steps:
      1. Merge edits (text, tags, keep) from the DataFrame back onto the
         original chunk metadata (chunk_id, page_number, image_paths).
      2. Delete any existing ChromaDB entries for the same source PDFs so
         re-committing a manual replaces it cleanly instead of duplicating.
      3. Split chunks into batches, embed each batch via Voyage AI, upsert
         to ChromaDB.

    Returns the number of chunks successfully committed.
    """
    def _progress(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    # ── Step 1: merge edited DataFrame back onto original chunk metadata ───────
    # edited_df rows align 1-to-1 with parsed_chunks by position (same order,
    # same count — guaranteed because we built the DataFrame from parsed_chunks).
    keep_chunks = []
    for i, (_, row) in enumerate(edited_df.iterrows()):
        if not row["keep"]:
            continue
        original = parsed_chunks[i]
        keep_chunks.append({
            "chunk_id":    original["chunk_id"],
            "source_pdf":  original["source_pdf"],
            "page_number": original["page_number"],
            "image_paths": original["image_paths"],
            "text":        str(row["text"]),
            "tags":        str(row["tags"]),
        })

    if not keep_chunks:
        return 0

    # ── Step 2: delete existing entries for these PDFs ────────────────────────
    # This makes re-commits idempotent: re-processing a manual replaces it.
    affected_pdfs = list({c["source_pdf"] for c in keep_chunks})
    _progress(0.05, f"Clearing old entries for {len(affected_pdfs)} PDF(s)…")
    collection = _get_collection()
    for pdf_name in affected_pdfs:
        existing = collection.get(
            where={"source_pdf": pdf_name},
            include=[],   # we only need the IDs
        )
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    # ── Step 3: embed in batches and upsert ───────────────────────────────────
    voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    total = len(keep_chunks)
    committed = 0

    for batch_start in range(0, total, _EMBED_BATCH_SIZE):
        batch = keep_chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        frac = 0.1 + 0.85 * (batch_start / total)
        _progress(frac, f"Embedding chunks {batch_start + 1}–{min(batch_start + len(batch), total)} of {total}…")

        # input_type="document" tells Voyage this is content to be stored
        # (not a query) — it affects how the model encodes the text.
        result = voyage_client.embed(texts, model=config.VOYAGE_MODEL, input_type="document")
        embeddings = result.embeddings

        # Build the metadata dicts. ChromaDB requires all metadata values to
        # be str, int, float, or bool — no lists — so image_paths is JSON-encoded.
        metadatas = [
            {
                "source_pdf":  c["source_pdf"],
                "page_number": c["page_number"],
                "tags":        c["tags"],
                "image_paths": json.dumps(c["image_paths"]),  # list → JSON string
            }
            for c in batch
        ]

        # upsert = insert if new, update if chunk_id already exists
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        committed += len(batch)

        # Brief pause between batches to respect Voyage AI rate limits
        if batch_start + _EMBED_BATCH_SIZE < total:
            time.sleep(_RATE_LIMIT_PAUSE)

    _progress(1.0, f"Done — {committed} chunks committed.")
    return committed


def get_indexed_pdfs() -> list[dict]:
    """
    Return a summary of every PDF currently stored in ChromaDB.
    Each entry: {"source_pdf": str, "chunk_count": int}
    Returns an empty list if the collection is empty or doesn't exist yet.
    """
    try:
        collection = _get_collection()
        results = collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            name = meta.get("source_pdf", "Unknown")
            counts[name] = counts.get(name, 0) + 1
        return [{"source_pdf": k, "chunk_count": v} for k, v in sorted(counts.items())]
    except Exception:
        return []


def delete_pdf_from_index(pdf_name: str) -> int:
    """
    Remove all chunks belonging to pdf_name from ChromaDB.
    Returns the number of chunks deleted.
    """
    collection = _get_collection()
    existing = collection.get(where={"source_pdf": pdf_name}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_total_chunk_count() -> int:
    """Return the total number of chunks currently in ChromaDB."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
