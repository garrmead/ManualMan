# utils/retriever.py
#
# Retrieval half of RAG: embed the question, search ChromaDB, apply the
# relevance threshold, and format results for Claude's context window.

import json

import chromadb
import voyageai

import config


def retrieve(
    question: str,
    top_k: int = config.TOP_K,
    min_score: float = config.MIN_RELEVANCE_SCORE,
) -> list[dict]:
    """
    Embed the question and return the top_k most relevant chunks from ChromaDB,
    filtered to those scoring >= min_score.

    Each returned dict contains:
        text         — chunk text sent to Claude as context
        source_pdf   — original PDF filename
        page_number  — 1-based page number
        tags         — user-assigned tags (may be empty)
        image_paths  — page-level image file paths (text chunks)
        score        — cosine similarity 0–1
        chunk_type   — "text" or "image"
        image_path   — file path (image chunks only)
        image_type   — vocabulary key (image chunks only)
        manufacturer — from manual metadata (may be empty)
        product_line — from manual metadata (may be empty)
        doc_type     — from manual metadata (may be empty)
        revision     — from manual metadata (may be empty)
    """
    voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    result = voyage_client.embed(
        [question],
        model=config.VOYAGE_MODEL,
        input_type="query",   # asymmetric: "query" for questions, "document" at index time
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
        score = round(1.0 - dist, 3)   # distance → similarity

        # Apply relevance threshold — drop chunks that aren't similar enough
        if score < min_score:
            continue

        chunks.append({
            "text":         doc,
            "source_pdf":   meta.get("source_pdf", "Unknown"),
            "page_number":  int(meta.get("page_number", 0)),
            "tags":         meta.get("tags", ""),
            "image_paths":  json.loads(meta.get("image_paths", "[]")),
            "score":        score,
            "chunk_type":   meta.get("chunk_type", "text"),
            "image_path":   meta.get("image_path", ""),
            "image_type":   meta.get("image_type", ""),
            # Manual metadata fields (empty string if not set at commit time)
            "manufacturer": meta.get("manufacturer", ""),
            "product_line": meta.get("product_line", ""),
            "doc_type":     meta.get("doc_type", ""),
            "revision":     meta.get("revision", ""),
        })

    return chunks


def build_context_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for Claude.
    Includes manual metadata in each source header so Claude can cite it.
    """
    if not chunks:
        return "(No relevant context found in the indexed manuals.)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Build a rich source label that includes manual metadata when present
        meta_parts = []
        if chunk.get("manufacturer"):
            meta_parts.append(chunk["manufacturer"])
        if chunk.get("product_line"):
            meta_parts.append(chunk["product_line"])
        if chunk.get("doc_type"):
            meta_parts.append(chunk["doc_type"])
        if chunk.get("revision"):
            meta_parts.append(chunk["revision"])

        meta_str = " · ".join(meta_parts)
        base = f"{chunk['source_pdf']}"
        if meta_str:
            base += f" ({meta_str})"
        base += f", Page {chunk['page_number']}"

        if chunk.get("chunk_type") == "image":
            from utils.vision import IMAGE_TYPE_LABELS
            label = IMAGE_TYPE_LABELS.get(chunk.get("image_type", ""), "Image")
            header = f"[Source {i}: {base} — {label} (image)]"
        else:
            header = f"[Source {i}: {base}]"

        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def build_chat_messages(
    question: str,
    context: str,
    chat_history: list[dict],
) -> list[dict]:
    """
    Build the messages array for Claude, prepending recent conversation history
    so follow-up questions have context.

    History pairs are included oldest-first, up to MAX_HISTORY_TURNS complete
    pairs. The current question (with retrieved context) is always last.
    """
    messages = []

    # Extract only user/assistant turns (skip any other roles)
    past = [m for m in chat_history if m["role"] in ("user", "assistant")]

    # Take the last N complete pairs (N*2 messages)
    recent = past[-(config.MAX_HISTORY_TURNS * 2):]

    # If history starts with an assistant message, drop it to keep proper
    # user→assistant alternation required by the Anthropic API
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]

    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Current turn: question with freshly retrieved context
    messages.append({
        "role": "user",
        "content": f"Context from pump manuals:\n\n{context}\n\n---\n\nQuestion: {question}",
    })

    return messages
