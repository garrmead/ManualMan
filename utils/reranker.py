# utils/reranker.py
#
# Cross-encoder reranking pass after hybrid retrieval.
# Uses sentence-transformers with a lightweight MS-MARCO cross-encoder model.
# Model is downloaded lazily (~80 MB) and cached for the lifetime of the process.
#
# Falls back to score-passthrough if sentence-transformers is unavailable
# or the model fails to load.

import threading
from typing import Optional

import config

_lock  = threading.Lock()
_model = None          # CrossEncoder instance after first use
_failed = False        # set True if load fails so we don't retry every call


def _load_model():
    global _model, _failed
    if _model is not None or _failed:
        return
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANKER_MODEL, max_length=512)
        print(f"[reranker] Loaded {config.RERANKER_MODEL}")
    except Exception as e:
        print(f"[reranker] Could not load cross-encoder ({e}); falling back to fusion scores.")
        _failed = True


def rerank(query: str, chunks: list[dict], top_n: int = None) -> list[dict]:
    """
    Rerank chunks using a cross-encoder.  Returns the list sorted by
    cross-encoder score (best first), truncated to top_n.

    Each chunk dict must have a "text" key.
    The returned dicts gain a "rerank_score" key.
    If the cross-encoder is unavailable, returns the input list unchanged
    (already sorted by RRF fusion score).
    """
    if not config.RERANKER_ENABLED or not chunks:
        return chunks[:top_n] if top_n else chunks

    with _lock:
        _load_model()

    if _model is None:
        return chunks[:top_n] if top_n else chunks

    pairs  = [(query, c["text"][:1024]) for c in chunks]
    scores = _model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n] if top_n else ranked
