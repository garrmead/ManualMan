# utils/retriever.py
#
# Hybrid retrieval pipeline:
#   1. Dense vector search (Voyage AI + ChromaDB)
#   2. BM25 keyword search
#   3. Reciprocal Rank Fusion (RRF) merge
#   4. Troubleshooting-chunk boost
#   5. Cross-encoder reranking
#   6. Relevance threshold filter
#
# For exact-match technical queries (model numbers, part numbers, torque specs),
# BM25 handles what semantic search misses. RRF prevents either system from
# dominating when both have strong signals.

import json
import re

import chromadb
import voyageai

import config
from utils.bm25_index import search as bm25_search
from utils.troubleshoot import classify_query, get_troubleshoot_system_addendum

# Detects numeric tokens in a query (e.g. "1450", "316", "3-13")
_NUMBERS_RE = re.compile(r'\b\d+\b')


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _rrf_merge(
    vector_ids:  list[str],
    bm25_ids:    list[str],
    vector_w:    float = config.HYBRID_VECTOR_WEIGHT,
    bm25_w:      float = config.HYBRID_BM25_WEIGHT,
    k:           int   = config.RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    Returns [(chunk_id, rrf_score), ...] sorted descending.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(vector_ids):
        scores[cid] = scores.get(cid, 0.0) + vector_w * (1.0 / (k + rank + 1))
    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + bm25_w * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Main retrieval function ───────────────────────────────────────────────────

def retrieve(
    question:  str,
    top_k:     int   = config.FINAL_TOP_K,
    min_score: float = config.MIN_RELEVANCE_SCORE,
    filters:   dict  = None,
) -> list[dict]:
    """
    Hybrid retrieval with reranking.

    Parameters
    ----------
    question  : user's query
    top_k     : number of results to return after reranking
    min_score : minimum vector similarity to keep a chunk
    filters   : optional ChromaDB where-clause e.g. {"manufacturer": "Goulds"}

    Returns
    -------
    List of chunk dicts, each with:
        text, source_pdf, page_number, tags, image_paths,
        score (vector similarity), bm25_score, rrf_score, rerank_score,
        chunk_type, image_path, image_type, manufacturer, product_line,
        doc_type, revision, section_title, chunk_subtype, is_troubleshooting
    """
    # ── Query intent ──────────────────────────────────────────────────────────
    intent = classify_query(question)
    is_visual      = intent["is_visual_query"]
    has_exact_nums = bool(_NUMBERS_RE.search(question))

    # Numeric queries (RPM, part #, torque): BM25 wins on exact-match values.
    # Semantic search can't distinguish "1450 RPM" from "1750 RPM" — they embed
    # nearly identically. Flip the weights so BM25 leads when numbers are present.
    if has_exact_nums:
        vec_w  = config.EXACT_QUERY_VECTOR_WEIGHT   # 0.35
        bm25_w = config.EXACT_QUERY_BM25_WEIGHT     # 0.65
    else:
        vec_w  = config.HYBRID_VECTOR_WEIGHT        # 0.6
        bm25_w = config.HYBRID_BM25_WEIGHT          # 0.4

    # ── Vector search ─────────────────────────────────────────────────────────
    voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    result = voyage_client.embed(
        [question],
        model=config.VOYAGE_MODEL,
        input_type="query",
    )
    query_embedding = result.embeddings[0]

    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    count = collection.count()
    if count == 0:
        return []

    # Larger candidate pool for visual+exact queries — more images need to survive
    # the initial fetch so the reranker can pick the right one.
    if is_visual and has_exact_nums:
        candidate_k = max(config.TOP_K, top_k * 5, 25)
    else:
        candidate_k = max(config.TOP_K, top_k * 3)

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, count),
        include=["documents", "metadatas", "distances"],
    )
    if filters:
        query_kwargs["where"] = filters

    vector_results = collection.query(**query_kwargs)

    # Build chunk dicts from vector results
    vector_chunks: dict[str, dict] = {}
    vector_rank_ids: list[str] = []

    for doc, meta, dist in zip(
        vector_results["documents"][0],
        vector_results["metadatas"][0],
        vector_results["distances"][0],
    ):
        score = round(1.0 - dist, 4)
        if score < min_score:
            continue
        cid = meta.get("chunk_id", doc[:32])  # use stored chunk_id if available
        vector_chunks[cid] = _build_chunk_dict(doc, meta, score)
        vector_rank_ids.append(cid)

    # ── BM25 search ───────────────────────────────────────────────────────────
    # Enrich BM25 query with troubleshooting keywords when in TS mode
    bm25_query = question
    if intent["is_troubleshooting"] and intent["boost_keywords"]:
        bm25_query += " " + " ".join(intent["boost_keywords"])

    bm25_hits = bm25_search(bm25_query, top_k=candidate_k)
    bm25_rank_ids = [h["chunk_id"] for h in bm25_hits]
    bm25_score_map = {h["chunk_id"]: h["bm25_score"] for h in bm25_hits}

    # ── RRF merge ─────────────────────────────────────────────────────────────
    all_ids   = set(vector_rank_ids) | set(bm25_rank_ids)
    merged    = _rrf_merge(vector_rank_ids, bm25_rank_ids, vector_w=vec_w, bm25_w=bm25_w)

    # Hydrate any BM25-only results from ChromaDB
    bm25_only_ids = [cid for cid, _ in merged if cid not in vector_chunks]
    if bm25_only_ids:
        _hydrate_from_chroma(bm25_only_ids, vector_chunks, collection)

    # Build final candidate list in RRF order
    candidates: list[dict] = []
    for cid, rrf_score in merged:
        if cid not in vector_chunks:
            continue
        chunk = dict(vector_chunks[cid])
        chunk["rrf_score"]   = round(rrf_score, 6)
        chunk["bm25_score"]  = round(bm25_score_map.get(cid, 0.0), 4)
        candidates.append(chunk)

    if not candidates:
        return []

    # ── Troubleshooting boost ─────────────────────────────────────────────────
    if intent["is_troubleshooting"]:
        for chunk in candidates:
            if chunk.get("chunk_subtype") == "troubleshooting" or \
               chunk.get("is_troubleshooting") == "true":
                chunk["rrf_score"] *= config.TROUBLESHOOT_BOOST
        candidates.sort(key=lambda c: c["rrf_score"], reverse=True)

    # ── Visual query boost ────────────────────────────────────────────────────
    # For visual queries (curves, drawings, dimensions) boost image chunks so
    # the right diagram surfaces above generic text chunks in the reranker pool.
    if is_visual:
        for chunk in candidates:
            if chunk.get("chunk_type") == "image":
                chunk["rrf_score"] *= config.VISUAL_IMAGE_BOOST
        candidates.sort(key=lambda c: c["rrf_score"], reverse=True)

    # ── Cross-encoder reranking ───────────────────────────────────────────────
    pool_size   = max(top_k * 5, 25) if is_visual else max(top_k * 4, 20)
    rerank_pool = candidates[:pool_size]
    try:
        from utils.reranker import rerank
        reranked = rerank(question, rerank_pool, top_n=top_k)
    except Exception:
        reranked = rerank_pool[:top_k]

    # Annotate with query intent for downstream UI use
    for chunk in reranked:
        chunk["query_is_troubleshooting"] = intent["is_troubleshooting"]

    return reranked


# ── Context builders ──────────────────────────────────────────────────────────

def build_context_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "(No relevant context found in the indexed manuals.)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta_parts = []
        for k in ("manufacturer", "product_line", "doc_type", "revision"):
            if chunk.get(k):
                meta_parts.append(chunk[k])
        meta_str = " · ".join(meta_parts)
        base = chunk["source_pdf"]
        if meta_str:
            base += f" ({meta_str})"
        base += f", Page {chunk['page_number']}"
        if chunk.get("section_title"):
            base += f" § {chunk['section_title']}"

        if chunk.get("chunk_type") == "image":
            from utils.vision import IMAGE_TYPE_LABELS
            label  = IMAGE_TYPE_LABELS.get(chunk.get("image_type", ""), "Image")
            header = f"[Source {i}: {base} — {label} (image)]"
        else:
            header = f"[Source {i}: {base}]"

        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def build_chat_messages(
    question:     str,
    context:      str,
    chat_history: list[dict],
    is_troubleshooting: bool = False,   # reserved for future per-turn routing
) -> list[dict]:
    messages = []
    past     = [m for m in chat_history if m["role"] in ("user", "assistant")]
    recent   = past[-(config.MAX_HISTORY_TURNS * 2):]
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Troubleshooting addendum is injected into the system prompt in app.py —
    # not here — to avoid duplicating it across the system + user turn.
    content = f"Context from pump manuals:\n\n{context}\n\n---\n\nQuestion: {question}"
    messages.append({"role": "user", "content": content})
    return messages


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_chunk_dict(doc: str, meta: dict, score: float) -> dict:
    return {
        "text":         doc,
        "source_pdf":   meta.get("source_pdf", "Unknown"),
        "page_number":  int(meta.get("page_number", 0)),
        "tags":         meta.get("tags", ""),
        "image_paths":  json.loads(meta.get("image_paths", "[]")),
        "score":        score,
        "chunk_type":   meta.get("chunk_type", "text"),
        "image_path":   meta.get("image_path", ""),
        "image_type":   meta.get("image_type", ""),
        "manufacturer": meta.get("manufacturer", ""),
        "product_line": meta.get("product_line", ""),
        "doc_type":     meta.get("doc_type", ""),
        "revision":     meta.get("revision", ""),
        "section_title":   meta.get("section_title", ""),
        "chunk_subtype":   meta.get("chunk_subtype", "text"),
        "is_troubleshooting": meta.get("is_troubleshooting", "false"),
        # Scores filled in later
        "rrf_score":    0.0,
        "bm25_score":   0.0,
        "rerank_score": 0.0,
    }


def _hydrate_from_chroma(
    chunk_ids: list[str],
    target: dict,
    collection: chromadb.Collection,
) -> None:
    """Fetch BM25-only results from ChromaDB by chunk_id and add to target dict."""
    if not chunk_ids:
        return
    try:
        res = collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(res["documents"], res["metadatas"]):
            cid = meta.get("chunk_id", doc[:32])
            if cid not in target:
                target[cid] = _build_chunk_dict(doc, meta, 0.0)
    except Exception:
        pass
