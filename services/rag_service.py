"""
RAG Service — Chunking, embedding, hybrid retrieval, and grounding for legal documents.
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def keyword_overlap_score(query: str, text: str) -> float:
    """Fast lexical overlap score for query words matching within chunk text."""
    query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for token in query_tokens if token in text_lower)
    return matches / len(query_tokens)


class RAGService:
    def __init__(self, client):
        self._client = client

    def chunk_document(
        self, text: str, target_size: int = 800, overlap: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Split legal text into overlapping, clause-aware passages.
        Preserves paragraph and clause boundaries wherever possible.
        """
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        chunk_id = 1

        for para in raw_paras:
            # If adding this paragraph exceeds target size and current_chunk is non-empty,
            # finalize current chunk
            if len(current_chunk) + len(para) > target_size and len(current_chunk) >= target_size // 2:
                heading = self._detect_heading(current_chunk, chunk_id)
                chunks.append({
                    "id": chunk_id,
                    "heading": heading,
                    "text": current_chunk.strip(),
                    "snippet": current_chunk.strip()[:180] + ("…" if len(current_chunk) > 180 else ""),
                    "vector": None,
                })
                chunk_id += 1
                # Keep tail for overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            heading = self._detect_heading(current_chunk, chunk_id)
            chunks.append({
                "id": chunk_id,
                "heading": heading,
                "text": current_chunk.strip(),
                "snippet": current_chunk.strip()[:180] + ("…" if len(current_chunk) > 180 else ""),
                "vector": None,
            })

        # If document was small and produced only 1 chunk or 0 chunks
        if not chunks and text.strip():
            chunks.append({
                "id": 1,
                "heading": "Full Document",
                "text": text.strip(),
                "snippet": text.strip()[:180] + ("…" if len(text) > 180 else ""),
                "vector": None,
            })

        logger.info("Chunked document into %d passages", len(chunks))
        return chunks

    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute embeddings for all chunks in batches using Gemini Embeddings API."""
        if not self._client or not chunks:
            return chunks

        batch_size = 15
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            contents = [c["text"] for c in batch]
            try:
                res = await self._client.aio.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents,
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    for chunk, emb in zip(batch, res.embeddings):
                        chunk["vector"] = emb.values
                elif hasattr(res, "embedding") and res.embedding:
                    batch[0]["vector"] = res.embedding.values
            except Exception as e:
                logger.warning("Embedding batch %d-%d failed: %s. Using lexical retriever fallback.", i, i + len(batch), e)

        return chunks

    async def retrieve(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: combines cosine similarity of vector embeddings
        with keyword overlap scoring.
        """
        if not chunks:
            return []

        query_vector: Optional[List[float]] = None
        if self._client:
            try:
                res = await self._client.aio.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=query,
                )
                if hasattr(res, "embedding") and res.embedding:
                    query_vector = res.embedding.values
                elif hasattr(res, "embeddings") and res.embeddings:
                    query_vector = res.embeddings[0].values
            except Exception as e:
                logger.warning("Query embedding failed: %s. Falling back to keyword scoring.", e)

        scored = []
        for c in chunks:
            vec_score = 0.0
            if query_vector and c.get("vector"):
                vec_score = max(0.0, cosine_similarity(query_vector, c["vector"]))

            kw_score = keyword_overlap_score(query, c["text"])

            if query_vector and c.get("vector"):
                # Hybrid weighting: 75% vector semantics, 25% exact keywords
                final_score = (0.75 * vec_score) + (0.25 * kw_score)
            else:
                final_score = kw_score

            scored.append({
                "id": c["id"],
                "heading": c["heading"],
                "text": c["text"],
                "snippet": c["snippet"],
                "score": round(final_score, 3),
            })

        # Sort by relevance score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _detect_heading(chunk_text: str, fallback_id: int) -> str:
        """Extract a readable heading or clause identifier from chunk text."""
        first_line = chunk_text.strip().split("\n")[0].strip()
        # Look for patterns like "Section 1", "Article IV", "1. Definitions", "CLAUSE 3"
        pattern = re.search(
            r"^(?:article|section|clause|paragraph|\d+[\.\)])\s*[^:\n]{2,40}",
            first_line,
            re.IGNORECASE,
        )
        if pattern:
            return pattern.group(0).strip(" :.-")
        if len(first_line) <= 60 and not first_line.endswith("."):
            return first_line
        return f"Clause / Section {fallback_id}"
