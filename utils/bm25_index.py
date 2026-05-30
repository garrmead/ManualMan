# utils/bm25_index.py
#
# Persistent BM25 index that lives alongside ChromaDB.
# Enables exact-keyword retrieval for model numbers, part numbers, specs, etc. —
# content that semantic embeddings handle poorly because the terms are rare or
# highly technical (e.g. "3196-LTX 2x3-13", "316SS", "35 ft-lbs").
#
# Corpus is stored as a JSON file so it survives restarts and stays in sync with
# ChromaDB.  The in-memory BM25Okapi object is rebuilt from the corpus on each
# retrieval call (fast for <50k chunks) and cached at module level.

import json
import re
import threading
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Plus  # BM25Plus avoids zero IDF scores for small corpora

import config

# ── Module-level cache ────────────────────────────────────────────────────────
_lock        = threading.Lock()
_corpus_ver  = 0          # incremented whenever the corpus changes on disk
_cached_ver  = -1         # version the current _bm25 was built from
_bm25: Optional[BM25Plus] = None
_corpus_ids: list[str]   = []
_corpus_meta: list[dict] = []


# ── Tokenization ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Tokenize for BM25 with sub-token expansion.

    Keeps hyphenated/dotted model numbers intact ("3196-LTX") AND generates
    their constituent parts ("3196", "ltx") so that a query of "3196" matches
    a document token of "3196-LTX".
    """
    # Normalise comma-formatted numbers before tokenising (1,450 → 1450)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = text.lower()
    raw  = re.findall(r"[a-z0-9][a-z0-9\-\.]*", text)
    seen: set[str] = set()
    tokens: list[str] = []
    for tok in raw:
        if tok not in seen:
            tokens.append(tok)
            seen.add(tok)
        # Add sub-parts for compound tokens
        if "-" in tok or "." in tok:
            for part in re.split(r"[-.]", tok):
                if part and part not in seen:
                    tokens.append(part)
                    seen.add(part)
    return tokens


# ── Corpus persistence ────────────────────────────────────────────────────────

def _corpus_path() -> Path:
    return Path(config.BM25_INDEX_FILE)


def _load_raw() -> dict:
    p = _corpus_path()
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    return {"version": 0, "chunks": []}


def _save_raw(data: dict) -> None:
    p = _corpus_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), "utf-8")


# ── Index rebuild ─────────────────────────────────────────────────────────────

def _rebuild_if_stale() -> None:
    global _bm25, _corpus_ids, _corpus_meta, _cached_ver
    raw  = _load_raw()
    ver  = raw.get("version", 0)
    if ver == _cached_ver and _bm25 is not None:
        return
    chunks = raw.get("chunks", [])
    if not chunks:
        _bm25        = None
        _corpus_ids  = []
        _corpus_meta = []
        _cached_ver  = ver
        return
    tokenized    = [_tokenize(c["text"]) for c in chunks]
    _bm25        = BM25Plus(tokenized)
    _corpus_ids  = [c["chunk_id"] for c in chunks]
    _corpus_meta = [{k: c[k] for k in c if k != "text"} for c in chunks]
    _cached_ver  = ver


# ── Public API ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Public alias for the BM25 tokenizer — usable by other modules."""
    return _tokenize(text)


def add_chunks(chunks: list[dict]) -> None:
    """
    Add or replace chunks in the BM25 corpus.
    Each chunk must have: chunk_id, text, source_pdf, page_number, chunk_type.
    Existing chunks with the same chunk_id are replaced.
    """
    with _lock:
        raw = _load_raw()
        existing = {c["chunk_id"]: i for i, c in enumerate(raw["chunks"])}
        for chunk in chunks:
            entry = {
                "chunk_id":    chunk["chunk_id"],
                "text":        chunk["text"],
                "source_pdf":  chunk.get("source_pdf", ""),
                "page_number": chunk.get("page_number", 0),
                "chunk_type":  chunk.get("chunk_type", "text"),
                "manufacturer":chunk.get("manufacturer", ""),
                "product_line":chunk.get("product_line", ""),
                "doc_type":    chunk.get("doc_type", ""),
                "revision":    chunk.get("revision", ""),
                "section_title": chunk.get("section_title", ""),
                "chunk_subtype": chunk.get("chunk_subtype", ""),
            }
            if chunk["chunk_id"] in existing:
                raw["chunks"][existing[chunk["chunk_id"]]] = entry
            else:
                raw["chunks"].append(entry)
        raw["version"] = raw.get("version", 0) + 1
        _save_raw(raw)


def remove_pdf(pdf_name: str) -> int:
    """Remove all BM25 corpus entries for a given PDF. Returns count removed."""
    with _lock:
        raw    = _load_raw()
        before = len(raw["chunks"])
        raw["chunks"] = [c for c in raw["chunks"] if c["source_pdf"] != pdf_name]
        removed = before - len(raw["chunks"])
        if removed:
            raw["version"] = raw.get("version", 0) + 1
            _save_raw(raw)
        return removed


def search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 search. Returns up to top_k results sorted by score (descending).

    Each result dict:
        chunk_id, score, source_pdf, page_number, chunk_type,
        manufacturer, product_line, doc_type, revision, section_title
    """
    with _lock:
        _rebuild_if_stale()
        if _bm25 is None or not _corpus_ids:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores   = _bm25.get_scores(tokens)
        pairs    = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results  = []
        for idx, score in pairs[:top_k]:
            if score <= 0:
                break
            meta = dict(_corpus_meta[idx])
            meta["chunk_id"] = _corpus_ids[idx]
            meta["bm25_score"] = float(score)
            results.append(meta)
        return results


def get_corpus_size() -> int:
    """Return number of chunks in the BM25 corpus."""
    raw = _load_raw()
    return len(raw.get("chunks", []))
