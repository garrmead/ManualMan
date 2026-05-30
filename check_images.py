import json
from pathlib import Path

# ── BM25 index ────────────────────────────────────────────────────────────────
print("=" * 60)
print("BM25 INDEX")
print("=" * 60)
data = json.loads(open("data/bm25_index.json").read())
imgs = [c for c in data["chunks"] if c.get("chunk_type") == "image"]
print(f"{len(imgs)} image chunks\n")
for c in imgs:
    print(f"chunk_id : {c['chunk_id']}")
    print(f"pdf/page : {c['source_pdf']}  pg {c['page_number']}")
    print(f"desc     : {c['text'][:200]}")
    print()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("CHROMADB")
print("=" * 60)
try:
    import chromadb, config
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col    = client.get_or_create_collection(
        config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Total chunks in ChromaDB: {col.count()}")
    res  = col.get(where={"chunk_type": "image"}, include=["documents", "metadatas"])
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    print(f"{len(docs)} image chunks\n")
    for doc, meta in zip(docs, metas):
        cid = meta.get("chunk_id", "?")
        img_path = meta.get("image_path", "")
        exists   = Path(img_path).exists() if img_path else False
        print(f"chunk_id   : {cid}")
        print(f"pdf/page   : {meta.get('source_pdf')}  pg {meta.get('page_number')}")
        print(f"image_path : {img_path}  (exists={exists})")
        print(f"desc       : {doc[:200]}")
        print()
except Exception as e:
    print(f"ChromaDB error: {e}")

# ── ID match check ────────────────────────────────────────────────────────────
print("=" * 60)
print("ID MATCH (BM25 vs ChromaDB)")
print("=" * 60)
try:
    bm25_ids   = {c["chunk_id"] for c in imgs}
    chroma_ids = {meta.get("chunk_id") for meta in metas}
    only_bm25  = bm25_ids - chroma_ids
    only_chroma = chroma_ids - bm25_ids
    print(f"In BM25 only  : {only_bm25 or 'none'}")
    print(f"In ChromaDB only: {only_chroma or 'none'}")
    print(f"In both       : {bm25_ids & chroma_ids}")
except Exception as e:
    print(f"Match check error: {e}")
