# utils/embedder.py
#
# Handles all vector-store work and parallel BM25 index updates.
# After text chunks are committed to ChromaDB, the same chunks are added to
# the BM25 corpus so hybrid retrieval works without reindexing.
#
# Optional entity extraction (config.ENTITY_EXTRACTION_ENABLED) enriches each
# chunk with structured metadata: model numbers, materials, RPM, HP, etc.

import hashlib
import json
import time
import threading
from typing import Callable

import chromadb
import voyageai

import config
from utils.bm25_index import add_chunks as bm25_add, remove_pdf as bm25_remove

# ── Constants ─────────────────────────────────────────────────────────────────
_EMBED_BATCH_SIZE = 32
_RATE_LIMIT_PAUSE = 0.5


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _stable_image_id(image_path: str) -> str:
    return "img_" + hashlib.md5(image_path.encode()).hexdigest()


def _embed_and_upsert(
    items: list[dict],
    collection: chromadb.Collection,
    voyage_client,
    progress_cb,
    progress_start: float,
    progress_end: float,
    label: str,
) -> int:
    total     = len(items)
    committed = 0
    for batch_start in range(0, total, _EMBED_BATCH_SIZE):
        batch = items[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [item["text"] for item in batch]

        frac = progress_start + (progress_end - progress_start) * (batch_start / max(total, 1))
        if progress_cb:
            end_idx = min(batch_start + len(batch), total)
            progress_cb(frac, f"{label} {batch_start + 1}–{end_idx} of {total}…")

        result     = voyage_client.embed(texts, model=config.VOYAGE_MODEL, input_type="document")
        embeddings = result.embeddings

        collection.upsert(
            ids        = [item["chunk_id"] for item in batch],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [item["metadata"] for item in batch],
        )
        committed += len(batch)
        if batch_start + _EMBED_BATCH_SIZE < total:
            time.sleep(_RATE_LIMIT_PAUSE)
    return committed


# ── Entity extraction helper ──────────────────────────────────────────────────

def _extract_entity_metadata(text: str, chunk_subtype: str) -> dict:
    """Call entity extractor if enabled; return flat metadata dict."""
    if not config.ENTITY_EXTRACTION_ENABLED:
        return {}
    try:
        from utils.entity_extractor import extract_entities, is_troubleshooting_text
        # Skip entity extraction on known non-text subtypes (logos, plain headers)
        if chunk_subtype in ("image",):
            return {}
        entities = extract_entities(text)
        return entities
    except Exception:
        return {}


# ── Main public function ──────────────────────────────────────────────────────

def commit_chunks(
    edited_df,
    parsed_chunks: list[dict],
    pdf_metadata:   dict | None = None,
    progress_cb:    Callable[[float, str], None] | None = None,
    classify_images: bool = True,
    excluded_images: set | None = None,
    extract_entities: bool = True,
) -> dict:
    """
    Embed and store all 'keep' text chunks plus their associated images.

    Steps:
      1. Merge user edits from the DataFrame back onto raw chunk metadata.
      2. Delete existing ChromaDB + BM25 entries for affected PDFs.
      3. (Optional) Extract structured entities from each text chunk.
      4. Embed text chunks → upsert to ChromaDB + BM25 corpus.
      5. Classify images → embed descriptions → upsert image chunks.

    Returns {"text_chunks": int, "image_chunks": int}
    """
    def _prog(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    # ── Step 1: merge edits ───────────────────────────────────────────────────
    keep_chunks = []
    for i, (_, row) in enumerate(edited_df.iterrows()):
        if not row["keep"]:
            continue
        orig = parsed_chunks[i]
        keep_chunks.append({
            "chunk_id":      orig["chunk_id"],
            "source_pdf":    orig["source_pdf"],
            "page_number":   orig["page_number"],
            "image_paths":   orig["image_paths"],
            "section_title": orig.get("section_title", ""),
            "chunk_subtype": orig.get("chunk_subtype", "text"),
            "ocr_used":      orig.get("ocr_used", False),
            "text":          str(row["text"]),
            "tags":          str(row["tags"]),
        })

    if not keep_chunks:
        return {"text_chunks": 0, "image_chunks": 0}

    # ── Step 2: clear existing entries ───────────────────────────────────────
    affected_pdfs = list({c["source_pdf"] for c in keep_chunks})
    _prog(0.02, f"Clearing old entries for {len(affected_pdfs)} PDF(s)…")
    collection = _get_collection()
    for pdf_name in affected_pdfs:
        existing = collection.get(where={"source_pdf": pdf_name}, include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        bm25_remove(pdf_name)

    voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    pdf_meta      = pdf_metadata or {}

    # ── Step 3 + 4: entity extraction + embed text chunks ────────────────────
    _prog(0.05, f"Extracting entities and embedding {len(keep_chunks)} text chunk(s)…")
    entity_progress_window = 0.20 if extract_entities and config.ENTITY_EXTRACTION_ENABLED else 0.0
    embed_start = 0.05 + entity_progress_window

    text_items   = []
    bm25_entries = []

    for ci, c in enumerate(keep_chunks):
        manual_meta = {
            k: pdf_meta.get(c["source_pdf"], {}).get(k, "")
            for k in ("manufacturer", "product_line", "doc_type", "revision")
        }

        # Entity extraction (optional, uses Claude haiku)
        entity_meta: dict = {}
        if extract_entities and config.ENTITY_EXTRACTION_ENABLED:
            frac = 0.05 + entity_progress_window * (ci / max(len(keep_chunks), 1))
            _prog(frac, f"Extracting entities from chunk {ci + 1}/{len(keep_chunks)}…")
            entity_meta = _extract_entity_metadata(c["text"], c["chunk_subtype"])

        # Determine if this chunk is troubleshooting content
        is_ts = entity_meta.pop("is_troubleshooting", "false")

        metadata = {
            "chunk_type":       "text",
            "chunk_id":         c["chunk_id"],   # stored so BM25 hydration can look up by id
            "source_pdf":       c["source_pdf"],
            "page_number":      c["page_number"],
            "tags":             c["tags"],
            "image_paths":      json.dumps(c["image_paths"]),
            "section_title":    c["section_title"],
            "chunk_subtype":    c["chunk_subtype"],
            "ocr_used":         "true" if c["ocr_used"] else "false",
            "is_troubleshooting": is_ts,
            **manual_meta,
            **entity_meta,
        }

        text_items.append({
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "metadata": metadata,
        })

        bm25_entries.append({
            "chunk_id":         c["chunk_id"],
            "text":             c["text"],
            "source_pdf":       c["source_pdf"],
            "page_number":      c["page_number"],
            "chunk_type":       "text",
            "section_title":    c["section_title"],
            "chunk_subtype":    c["chunk_subtype"],
            "is_troubleshooting": is_ts,
            **manual_meta,
        })

    text_committed = _embed_and_upsert(
        text_items, collection, voyage_client, progress_cb,
        progress_start=embed_start, progress_end=0.50,
        label="Embedding text chunk",
    )

    # Update BM25 index with all text chunks at once
    _prog(0.50, "Updating BM25 index…")
    if bm25_entries:
        bm25_add(bm25_entries)

    _prog(0.52, f"Text chunks done ({text_committed}). Starting image classification…")

    # ── Step 5: classify and embed images ─────────────────────────────────────
    image_committed = 0

    if classify_images:
        _excluded    = excluded_images or set()
        seen_paths:  set[str] = set()
        unique_imgs: list[dict] = []
        for c in keep_chunks:
            for img_path in c["image_paths"]:
                if img_path not in seen_paths and img_path not in _excluded:
                    seen_paths.add(img_path)
                    unique_imgs.append({
                        "image_path":  img_path,
                        "source_pdf":  c["source_pdf"],
                        "page_number": c["page_number"],
                    })

        if unique_imgs:
            from utils.vision import classify_images_batch, INDEXABLE_TYPES

            def _vision_prog(frac: float, msg: str):
                _prog(0.52 + frac * 0.40, msg)

            classified = classify_images_batch(unique_imgs, progress_cb=_vision_prog)
            technical  = [img for img in classified if img["image_type"] in INDEXABLE_TYPES]
            skipped    = len(classified) - len(technical)
            if skipped:
                _prog(0.92, f"Skipped {skipped} non-technical image(s)…")

            image_items   = []
            img_bm25      = []
            for img in technical:
                m = pdf_meta.get(img["source_pdf"], {})
                cid = _stable_image_id(img["image_path"])
                meta = {
                    "chunk_type":   "image",
                    "chunk_id":     cid,
                    "source_pdf":   img["source_pdf"],
                    "page_number":  img["page_number"],
                    "image_path":   img["image_path"],
                    "image_type":   img["image_type"],
                    "tags":         "",
                    "image_paths":  json.dumps([img["image_path"]]),
                    "section_title": "",
                    "chunk_subtype": "image",
                    "is_troubleshooting": "false",
                    **{k: m.get(k, "") for k in ("manufacturer", "product_line", "doc_type", "revision")},
                }
                image_items.append({
                    "chunk_id": cid,
                    "text":     img["description"],
                    "metadata": meta,
                })
                img_bm25.append({
                    "chunk_id":    cid,
                    "text":        img["description"],
                    "source_pdf":  img["source_pdf"],
                    "page_number": img["page_number"],
                    "chunk_type":  "image",
                    **{k: m.get(k, "") for k in ("manufacturer", "product_line", "doc_type", "revision")},
                })

            if image_items:
                _prog(0.92, f"Embedding {len(image_items)} image description(s)…")
                image_committed = _embed_and_upsert(
                    image_items, collection, voyage_client, progress_cb,
                    progress_start=0.92, progress_end=0.99,
                    label="Embedding image chunk",
                )
                if img_bm25:
                    bm25_add(img_bm25)

    _prog(1.0, f"Done — {text_committed} text + {image_committed} image chunks committed.")
    return {"text_chunks": text_committed, "image_chunks": image_committed}


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_image_chunks() -> list[dict]:
    """Return all indexed image chunks with descriptions and metadata."""
    try:
        collection = _get_collection()
        results    = collection.get(
            where={"chunk_type": "image"},
            include=["documents", "metadatas"],
        )
        chunks = []
        for cid, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            chunks.append({
                "chunk_id":    cid,
                "description": doc,
                "image_path":  meta.get("image_path", ""),
                "image_type":  meta.get("image_type", ""),
                "source_pdf":  meta.get("source_pdf", ""),
                "page_number": int(meta.get("page_number", 0)),
                "manufacturer": meta.get("manufacturer", ""),
                "product_line": meta.get("product_line", ""),
            })
        return sorted(chunks, key=lambda c: (c["source_pdf"], c["page_number"]))
    except Exception:
        return []


def update_image_descriptions(updates: dict[str, str]) -> int:
    """
    Re-embed and upsert updated image descriptions.

    Parameters
    ----------
    updates : {chunk_id: new_description_text}

    Returns number of chunks successfully updated.
    """
    if not updates:
        return 0

    collection    = _get_collection()
    voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    updated       = 0

    chunk_ids = list(updates.keys())
    result    = collection.get(ids=chunk_ids, include=["metadatas"])
    meta_map  = {cid: meta for cid, meta in zip(result["ids"], result["metadatas"])}

    items     = []
    bm25_rows = []
    for cid, new_text in updates.items():
        meta = meta_map.get(cid)
        if not meta:
            continue
        items.append({"chunk_id": cid, "text": new_text, "metadata": meta})
        bm25_rows.append({
            "chunk_id":    cid,
            "text":        new_text,
            "source_pdf":  meta.get("source_pdf", ""),
            "page_number": meta.get("page_number", 0),
            "chunk_type":  "image",
            "image_type":  meta.get("image_type", ""),
            "manufacturer": meta.get("manufacturer", ""),
            "product_line": meta.get("product_line", ""),
            "doc_type":    meta.get("doc_type", ""),
            "revision":    meta.get("revision", ""),
        })

    for batch_start in range(0, len(items), _EMBED_BATCH_SIZE):
        batch = items[batch_start: batch_start + _EMBED_BATCH_SIZE]
        texts = [b["text"] for b in batch]
        emb   = voyage_client.embed(texts, model=config.VOYAGE_MODEL, input_type="document")
        collection.upsert(
            ids        = [b["chunk_id"] for b in batch],
            embeddings = emb.embeddings,
            documents  = texts,
            metadatas  = [b["metadata"] for b in batch],
        )
        updated += len(batch)
        if batch_start + _EMBED_BATCH_SIZE < len(items):
            time.sleep(_RATE_LIMIT_PAUSE)

    if bm25_rows:
        bm25_add(bm25_rows)

    return updated


def get_indexed_pdfs() -> list[dict]:
    try:
        collection = _get_collection()
        results    = collection.get(include=["metadatas"])
        stats: dict[str, dict] = {}
        for meta in results["metadatas"]:
            name  = meta.get("source_pdf", "Unknown")
            ctype = meta.get("chunk_type", "text")
            if name not in stats:
                stats[name] = {
                    "text_chunks":  0,
                    "image_chunks": 0,
                    "manufacturer": meta.get("manufacturer", ""),
                    "product_line": meta.get("product_line", ""),
                    "doc_type":     meta.get("doc_type", ""),
                    "revision":     meta.get("revision", ""),
                }
            if ctype == "image":
                stats[name]["image_chunks"] += 1
            else:
                stats[name]["text_chunks"] += 1
        return [{"source_pdf": k, **v} for k, v in sorted(stats.items())]
    except Exception:
        return []


def delete_pdf_from_index(pdf_name: str) -> int:
    collection = _get_collection()
    existing   = collection.get(where={"source_pdf": pdf_name}, include=[])
    ids        = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    bm25_remove(pdf_name)
    return len(ids)


def get_total_chunk_count() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0


def get_manufacturers() -> list[str]:
    """Return unique non-empty manufacturer values for the filter sidebar."""
    try:
        results = _get_collection().get(include=["metadatas"])
        seen    = set()
        for meta in results["metadatas"]:
            m = meta.get("manufacturer", "").strip()
            if m:
                seen.add(m)
        return sorted(seen)
    except Exception:
        return []


def get_doc_types() -> list[str]:
    """Return unique non-empty doc_type values for the filter sidebar."""
    try:
        results = _get_collection().get(include=["metadatas"])
        seen    = set()
        for meta in results["metadatas"]:
            d = meta.get("doc_type", "").strip()
            if d:
                seen.add(d)
        return sorted(seen)
    except Exception:
        return []
