"""Memory linker — auto-link by similarity + typed links via extraction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from difflib import SequenceMatcher

from genesis_memory.protocols import LinkBackend, TextBackend, VectorBackend
from genesis_memory.types import LinkRecord

logger = logging.getLogger(__name__)

# Valid typed link types from schema CHECK constraint
_VALID_LINK_TYPES = frozenset({
    "supports", "contradicts", "extends", "elaborates",
    "discussed_in", "evaluated_for", "decided",
    "action_item_for", "categorized_as", "related_to",
    "succeeded_by", "preceded_by",
})


class MemoryLinker:
    """Find similar memories and create bidirectional links."""

    def __init__(
        self,
        *,
        vector_backend: VectorBackend,
        text_backend: TextBackend,
        link_backend: LinkBackend,
    ) -> None:
        self._vector = vector_backend
        self._text = text_backend
        self._links = link_backend

    async def auto_link(
        self,
        memory_id: str,
        vector: list[float],
        *,
        collection: str = "default",
        similarity_threshold: float = 0.75,
        max_links: int = 5,
    ) -> list[LinkRecord]:
        """Find similar memories and create links."""
        results = await self._vector.search(
            vector,
            limit=max_links + 1,
            collection=collection,
        )

        now = datetime.now(UTC).isoformat()
        links: list[LinkRecord] = []

        for hit in results:
            target_id = hit["id"]
            score = hit["score"]

            if target_id == memory_id:
                continue
            if score < similarity_threshold:
                continue
            if len(links) >= max_links:
                break

            link_type = "extends" if score >= 0.90 else "supports"

            try:
                await self._links.create(
                    memory_id, target_id, link_type, score, now,
                )
            except Exception:
                logger.debug("Link %s→%s already exists, skipping", memory_id, target_id)
                continue

            links.append(
                LinkRecord(
                    source_id=memory_id,
                    target_id=target_id,
                    link_type=link_type,
                    strength=score,
                    created_at=now,
                )
            )

        return links

    async def count_links(self, memory_id: str) -> int:
        """Count links for a memory."""
        return await self._links.count_links(memory_id)

    async def create_typed_links(
        self,
        memory_id: str,
        relationships: list[dict],
    ) -> list[LinkRecord]:
        """Create typed links from extraction relationships.

        For each relationship, searches for the target entity via FTS5
        (with difflib fallback for fuzzy matching) and creates a link.
        """
        if not relationships:
            return []

        now = datetime.now(UTC).isoformat()
        links: list[LinkRecord] = []

        for rel in relationships:
            link_type = rel.get("type", "")
            target_name = rel.get("to", "")

            if not link_type or not target_name:
                continue

            if link_type not in _VALID_LINK_TYPES:
                logger.debug(
                    "Skipping invalid link type %r for memory %s",
                    link_type, memory_id,
                )
                continue

            target_id = await self._find_entity_by_name(target_name)
            if not target_id:
                logger.debug(
                    "No matching memory found for entity %r (link from %s)",
                    target_name, memory_id,
                )
                continue

            if target_id == memory_id:
                continue

            try:
                await self._links.create(
                    memory_id, target_id, link_type, 0.7, now,
                )
                links.append(
                    LinkRecord(
                        source_id=memory_id,
                        target_id=target_id,
                        link_type=link_type,
                        strength=0.7,
                        created_at=now,
                    )
                )
            except Exception:
                logger.debug(
                    "Link %s -> %s (%s) already exists or failed",
                    memory_id, target_id, link_type,
                    exc_info=True,
                )

        return links

    async def _find_entity_by_name(self, entity_name: str) -> str | None:
        """Find a memory ID matching the given entity name.

        Uses FTS5 keyword search first, then falls back to difflib
        SequenceMatcher for fuzzy matching if FTS5 returns no results.
        """
        results = await self._text.search(entity_name, limit=5)

        if results:
            for r in results:
                if entity_name.lower() in r["content"].lower():
                    return r["memory_id"]
            return results[0]["memory_id"]

        # Difflib fallback
        words = entity_name.split()
        for word in words:
            if len(word) < 3:
                continue
            results = await self._text.search(word, limit=10)
            if results:
                best_match = None
                best_ratio = 0.0
                for r in results:
                    ratio = SequenceMatcher(
                        None,
                        entity_name.lower(),
                        r["content"][:200].lower(),
                    ).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = r["memory_id"]
                if best_ratio >= 0.6 and best_match is not None:
                    return best_match

        return None
