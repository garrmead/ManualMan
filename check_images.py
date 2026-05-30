import json

data = json.loads(open("data/bm25_index.json").read())
imgs = [c for c in data["chunks"] if c.get("chunk_type") == "image"]
print(f"{len(imgs)} image chunks in BM25 index\n")
for c in imgs:
    print(f"pg={c['page_number']}  {c['source_pdf']}")
    print(c["text"][:300])
    print()
