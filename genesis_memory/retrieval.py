"""Hybrid retrieval: vectors + FTS5 text + activation scoring, fused via RRF.

12-step pipeline:
1. Embed query → 2. Vector search → 3. Classify intent → 4. Query expansion
→ 5. FTS5 search → 6. Union candidates → 7. Activation scores → 8. Build
ranked lists → 9. RRF fusion → 10. Filter → 11. Sort → 12. Build results
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from genesis_memory.activation import compute_activation
from genesis_memory.embeddings import EmbeddingProvider, EmbeddingUnavailableError
from genesis_memory.protocols import LinkBackend, TextBackend, VectorBackend
from genesis_memory.types import RetrievalResult

logger = logging.getLogger(__name__)

def _normalize_tags(raw: object) -> list[str]:
    """Normalize tags from backend payload — may be list or comma-separated string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        return raw.split(",")
    return []


_SOURCE_TO_COLLECTIONS: dict[str, list[str]] = {
    "episodic": ["episodic_memory"],
    "knowledge": ["knowledge_base"],
    "both": ["episodic_memory", "knowledge_base"],
}


def _rrf_fuse(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion. Returns {memory_id: fused_score}."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, mid in enumerate(ranked, 1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Hybrid retrieval: vectors + FTS5 text + activation scoring, fused via RRF."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_backend: VectorBackend,
        text_backend: TextBackend,
        link_backend: LinkBackend,
    ) -> None:
        self._embeddings = embedding_provider
        self._vector = vector_backend
        self._text = text_backend
        self._links = link_backend

    async def recall(
        self,
        query: str,
        *,
        source: str = "both",
        limit: int = 10,
        min_activation: float = 0.0,
        expand_query_terms: bool = False,
        wing: str | None = None,
        room: str | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval: vectors + FTS5 + activation, fused via RRF."""
        if source not in _SOURCE_TO_COLLECTIONS:
            msg = f"source must be one of {list(_SOURCE_TO_COLLECTIONS)}, got {source!r}"
            raise ValueError(msg)

        collections = _SOURCE_TO_COLLECTIONS[source]
        candidate_limit = limit * 3

        # 1. Embed query (with fallback to FTS5-only)
        embedding_available = True
        vector = None
        try:
            vector = await self._embeddings.embed(query)
        except EmbeddingUnavailableError:
            embedding_available = False
            logger.warning("Embedding unavailable, falling back to FTS5-only retrieval")

        # 2. Vector search across collections (skip if no embedding)
        qdrant_results: list[dict] = []
        qdrant_by_id: dict[str, dict] = {}
        if embedding_available and vector is not None:
            for coll in collections:
                hits = await self._vector.search(
                    vector,
                    limit=candidate_limit,
                    collection=coll,
                    wing=wing,
                    room=room,
                )
                for hit in hits:
                    hit["_collection"] = coll
                qdrant_results.extend(hits)

            qdrant_results.sort(key=lambda h: h["score"], reverse=True)

            for hit in qdrant_results:
                mid = hit["id"]
                if mid not in qdrant_by_id:
                    qdrant_by_id[mid] = hit

        # 2b. Classify query intent (for RRF bias in step 7)
        # Import here to avoid circular dependency at module level
        from genesis_memory.intent import classify_intent, rank_by_intent

        intent = classify_intent(query)

        # 2c. Expand query via tag co-occurrence (opt-in)
        fts_query = query
        if expand_query_terms:
            try:
                from genesis_memory.intent import expand_query

                fts_query = await expand_query(
                    query, self._vector, collections, max_expansions=5,
                )
            except Exception:
                logger.warning("Query expansion failed, using original", exc_info=True)

        # 3. FTS5 text search
        fts_collection = collections[0] if len(collections) == 1 else None
        fts_is_boolean = fts_query != query
        fts_results = await self._text.search_ranked(
            fts_query,
            collection=fts_collection,
            limit=candidate_limit,
            boolean=fts_is_boolean,
        )

        fts_by_id: dict[str, dict] = {}
        for row in fts_results:
            mid = row["memory_id"]
            if mid not in fts_by_id:
                fts_by_id[mid] = row

        # 4. Union of all candidate memory_ids
        all_ids = set(qdrant_by_id) | set(fts_by_id)
        if not all_ids:
            return []

        # 5. Compute activation scores
        now_str = datetime.now(UTC).isoformat()
        activation_by_id: dict[str, float] = {}
        for mid in all_ids:
            qdrant_hit = qdrant_by_id.get(mid)
            if qdrant_hit:
                payload = qdrant_hit.get("payload", {})
                confidence = payload.get("confidence", 0.5)
                created_at = payload.get("created_at", now_str)
                retrieved_count = payload.get("retrieved_count", 0)
            else:
                confidence = 0.5
                created_at = now_str
                retrieved_count = 0
                payload = {}

            link_count = await self._links.count_links(mid)
            mem_class = payload.get("memory_class", "fact") if qdrant_hit else "fact"
            act = compute_activation(
                confidence=confidence,
                created_at=created_at,
                retrieved_count=retrieved_count,
                link_count=link_count,
                source=payload.get("source", "") if qdrant_hit else "",
                tags=_normalize_tags(payload.get("tags")) if qdrant_hit else [],
                now=now_str,
                memory_class=mem_class,
                memory_id=mid,
            )
            activation_by_id[mid] = act.final_score

        # 6. Build ranked lists for RRF
        vector_ranked_dedup: list[str] = []
        seen: set[str] = set()
        if embedding_available:
            vector_ranked = [h["id"] for h in qdrant_results if h["id"] in all_ids]
            for mid in vector_ranked:
                if mid not in seen:
                    seen.add(mid)
                    vector_ranked_dedup.append(mid)

        # Deduplicate FTS5 list (FTS5 can return dupes across content/tag matches)
        fts_ranked: list[str] = []
        fts_seen: set[str] = set()
        for r in fts_results:
            mid = r["memory_id"]
            if mid in all_ids and mid not in fts_seen:
                fts_seen.add(mid)
                fts_ranked.append(mid)

        activation_ranked = sorted(all_ids, key=lambda m: activation_by_id[m], reverse=True)

        # 6b. Build intent-biased ranked list
        intent_ranked: list[str] = []
        if intent.category != "GENERAL":
            candidate_meta: dict[str, dict] = {}
            for mid in all_ids:
                qhit = qdrant_by_id.get(mid)
                fhit = fts_by_id.get(mid)
                if qhit:
                    p = qhit.get("payload", {})
                    candidate_meta[mid] = {
                        "source": p.get("source", ""),
                        "tags": _normalize_tags(p.get("tags")),
                        "content": p.get("content", ""),
                    }
                elif fhit:
                    candidate_meta[mid] = {
                        "source": fhit.get("source_type", ""),
                        "tags": [],
                        "content": fhit.get("content", ""),
                    }
            intent_ranked = rank_by_intent(intent, candidate_meta)

        # 7. RRF fusion
        if embedding_available:
            ranked_lists = [vector_ranked_dedup, fts_ranked, activation_ranked]
        else:
            ranked_lists = [fts_ranked, activation_ranked]
        if intent_ranked:
            ranked_lists.append(intent_ranked)
        fused = _rrf_fuse(ranked_lists)

        # 8. Filter by min_activation
        candidates = [
            mid for mid in fused if activation_by_id.get(mid, 0.0) >= min_activation
        ]

        # 9. Sort by fused score descending
        candidates.sort(key=lambda m: fused[m], reverse=True)

        # 9b. Filter FTS5-only candidates by wing/room
        # Only apply when embedding is available — if it's unavailable, all
        # candidates are FTS5-only and filtering would drop everything.
        if (wing or room) and embedding_available:
            filtered: list[str] = []
            for mid in candidates:
                qhit = qdrant_by_id.get(mid)
                if qhit:
                    filtered.append(mid)
                # FTS5-only candidates excluded — can't verify wing/room
            candidates = filtered

        # 10. Take top limit
        top = candidates[:limit]

        # 11. Increment retrieved_count for returned results
        for mid in top:
            qdrant_hit = qdrant_by_id.get(mid)
            if qdrant_hit:
                coll = qdrant_hit.get("_collection", "default")
                old_count = qdrant_hit.get("payload", {}).get("retrieved_count", 0)
                try:
                    await self._vector.update_payload(
                        mid,
                        {"retrieved_count": old_count + 1},
                        collection=coll,
                    )
                except Exception:
                    logger.warning(
                        "Failed to update retrieved_count for %s in %s",
                        mid, coll, exc_info=True,
                    )

        # 12. Build RetrievalResult objects
        # Pre-build index lookups for O(1) rank computation
        vector_rank_map = {mid: i + 1 for i, mid in enumerate(vector_ranked_dedup)}
        fts_rank_map = {mid: i + 1 for i, mid in enumerate(fts_ranked)}

        results: list[RetrievalResult] = []
        for mid in top:
            qdrant_hit = qdrant_by_id.get(mid)
            fts_hit = fts_by_id.get(mid)

            if qdrant_hit:
                payload = qdrant_hit.get("payload", {})
                content = payload.get("content", "")
                src = payload.get("source", "")
                mem_type = payload.get("memory_type", "")
            elif fts_hit:
                content = fts_hit.get("content", "")
                src = fts_hit.get("source_type", "")
                mem_type = fts_hit.get("collection", "")
                payload = fts_hit
            else:
                continue

            v_rank = vector_rank_map.get(mid) if embedding_available else None
            f_rank = fts_rank_map.get(mid)

            _p = payload if qdrant_hit else {}
            _line_range = _p.get("source_line_range")
            results.append(
                RetrievalResult(
                    memory_id=mid,
                    content=content,
                    source=src,
                    memory_type=mem_type,
                    score=fused[mid],
                    vector_rank=v_rank,
                    fts_rank=f_rank,
                    activation_score=activation_by_id.get(mid, 0.0),
                    payload=_p,
                    source_session_id=_p.get("source_session_id"),
                    transcript_path=_p.get("transcript_path"),
                    source_line_range=tuple(_line_range) if _line_range else None,
                    source_pipeline=_p.get("source_pipeline"),
                    memory_class=_p.get("memory_class", "fact"),
                    query_intent=intent.category,
                    intent_confidence=intent.confidence,
                ),
            )

        return results
