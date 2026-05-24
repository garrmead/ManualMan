# utils/retriever.py
#
# Handles the retrieval half of RAG: embed the user's question via Voyage AI,
# search ChromaDB for the most similar chunks, and return them with metadata.
# Generation (calling Claude) happens in app.py so the streaming response can
# be written directly into the Streamlit UI.

import json

import chromadb
import voyageai

import config


def retrieve(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """
    Embed the question and return the top_k most relevant chunks from ChromaDB.

    Each returned dict contains:
        text        — the chunk text (what gets sent to Claude as context)
        source_pdf  — original filename
        page_number — 1-based page number
        tags        — user-assigned tags (may be empty)
        image_paths — list of image file paths on the same page
        score       — cosine similarity 0–1 (higher = more relevant)

    input_type="query" is the asymmetric counterpart to input_type="document"
    used at index time. Voyage AI uses different encoding for queries vs.
    stored documents, which improves retrieval quality.
    """
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

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine *distance* (0 = identical, 1 = orthogonal).
        # Convert to similarity so higher = better match.
        score = 1.0 - dist
        chunks.append({
            "text":        doc,
            "source_pdf":  meta.get("source_pdf", "Unknown"),
            "page_number": int(meta.get("page_number", 0)),
            "tags":        meta.get("tags", ""),
            "image_paths": json.loads(meta.get("image_paths", "[]")),
            "score":       round(score, 3),
            # Image-chunk fields — empty string for regular text chunks
            "chunk_type":  meta.get("chunk_type", "text"),
            "image_path":  meta.get("image_path", ""),
            "image_type":  meta.get("image_type", ""),
        })

    return chunks


def build_context_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for Claude's prompt.
    Each chunk is labelled with its source so Claude can cite it.
    """
    if not chunks:
        return "(No relevant context found in the indexed manuals.)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        if chunk.get("chunk_type") == "image":
            from utils.vision import IMAGE_TYPE_LABELS
            label = IMAGE_TYPE_LABELS.get(chunk.get("image_type", ""), "Image")
            header = (
                f"[Source {i}: {chunk['source_pdf']}, Page {chunk['page_number']} "
                f"— {label} (image)]"
            )
        else:
            header = f"[Source {i}: {chunk['source_pdf']}, Page {chunk['page_number']}]"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)
