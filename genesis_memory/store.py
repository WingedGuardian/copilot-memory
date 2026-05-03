"""Memory store — full pipeline: classify -> embed -> vector -> FTS5 -> auto-link."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from genesis_memory.classification import classify_memory
from genesis_memory.embeddings import EmbeddingProvider, EmbeddingUnavailableError
from genesis_memory.linker import MemoryLinker
from genesis_memory.protocols import (
    LinkBackend,
    MetadataBackend,
    PendingBackend,
    TextBackend,
    VectorBackend,
)

logger = logging.getLogger(__name__)

_COLLECTION_MAP = {
    "episodic": "episodic_memory",
    "knowledge": "knowledge_base",
}

# Broad catch for vector store connectivity errors
_VECTOR_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class MemoryStore:
    """Full store pipeline: classify -> embed -> vector -> FTS5 -> auto-link."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_backend: VectorBackend,
        text_backend: TextBackend,
        metadata_backend: MetadataBackend,
        link_backend: LinkBackend,
        pending_backend: PendingBackend | None = None,
        linker: MemoryLinker | None = None,
        collections: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self._embeddings = embedding_provider
        self._vector = vector_backend
        self._text = text_backend
        self._metadata = metadata_backend
        self._links = link_backend
        self._pending = pending_backend
        self._linker = linker
        self._collections = collections or list(set(_COLLECTION_MAP.values()))
        self._min_confidence = min_confidence

    @property
    def linker(self) -> MemoryLinker | None:
        return self._linker

    async def store(
        self,
        content: str,
        source: str,
        *,
        memory_type: str = "episodic",
        collection: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        auto_link: bool = True,
        memory_class: str | None = None,
        source_session_id: str | None = None,
        transcript_path: str | None = None,
        source_line_range: tuple[int, int] | None = None,
        extraction_timestamp: str | None = None,
        source_pipeline: str | None = None,
        wing: str | None = None,
        room: str | None = None,
        force_fts5_only: bool = False,
    ) -> str:
        """Full store pipeline. Returns memory_id."""
        # Dedup
        try:
            existing = await self._text.find_exact_duplicate(content)
            if existing:
                logger.debug("Skipping duplicate memory store: %s", existing)
                return existing
        except Exception:
            logger.warning("Dedup check failed, proceeding with store", exc_info=True)

        # Confidence gate
        if (
            self._min_confidence is not None
            and confidence is not None
            and confidence < self._min_confidence
        ):
            force_fts5_only = True

        memory_id = str(uuid.uuid4())
        now_iso = datetime.now(UTC).isoformat()
        resolved_tags = tags or []
        resolved_collection = collection or _COLLECTION_MAP.get(memory_type, "episodic_memory")
        resolved_class = memory_class or classify_memory(
            content, source=source, source_pipeline=source_pipeline or "",
        )

        # Append class tag for FTS5 discoverability
        class_tag = f"class:{resolved_class}"
        if class_tag not in resolved_tags:
            resolved_tags = [*resolved_tags, class_tag]

        # Taxonomy classification (optional — only if taxonomy module available)
        try:
            from genesis_memory.taxonomy import classify as classify_taxonomy

            if not wing or not room:
                taxo = classify_taxonomy(
                    content, tags=resolved_tags,
                    source=source, source_pipeline=source_pipeline or "",
                )
                wing = wing or taxo.wing
                room = room or taxo.room
        except ImportError:
            pass  # taxonomy not available — wing/room stay as provided

        if wing:
            wing_tag = f"wing:{wing}"
            if wing_tag not in resolved_tags:
                resolved_tags = [*resolved_tags, wing_tag]

        # Embed and store vector
        vector = None
        embedding_ok = not force_fts5_only
        if embedding_ok:
            try:
                enriched = EmbeddingProvider.enrich(content, memory_type, resolved_tags)
                vector = await self._embeddings.embed(enriched)

                payload = {
                    "content": content,
                    "source": source,
                    "memory_type": memory_type,
                    "tags": resolved_tags,
                    "confidence": confidence if confidence is not None else 0.5,
                    "created_at": now_iso,
                    "retrieved_count": 0,
                    "source_type": "memory",
                    "memory_class": resolved_class,
                    "wing": wing,
                    "room": room,
                }
                if source_session_id:
                    payload["source_session_id"] = source_session_id
                if transcript_path:
                    payload["transcript_path"] = transcript_path
                if source_line_range:
                    payload["source_line_range"] = list(source_line_range)
                if extraction_timestamp:
                    payload["extraction_timestamp"] = extraction_timestamp
                if source_pipeline:
                    payload["source_pipeline"] = source_pipeline

                await self._vector.upsert(
                    memory_id, vector, payload, collection=resolved_collection,
                )
            except EmbeddingUnavailableError:
                embedding_ok = False
                logger.warning(
                    "Embedding unavailable for memory %s, falling back to FTS5-only",
                    memory_id,
                )
            except _VECTOR_ERRORS:
                embedding_ok = False
                logger.error(
                    "Vector store error for memory %s — falling back to FTS5-only",
                    memory_id, exc_info=True,
                )
            except Exception:
                embedding_ok = False
                logger.error(
                    "Unexpected error during vector storage for memory %s",
                    memory_id, exc_info=True,
                )

        # Always write to FTS5
        await self._text.upsert(
            memory_id,
            content,
            source_type="memory",
            tags=",".join(resolved_tags) if resolved_tags else "",
            collection=resolved_collection,
        )

        # Write metadata
        await self._metadata.create(
            memory_id,
            created_at=now_iso,
            collection=resolved_collection,
            confidence=confidence,
            embedding_status="embedded" if embedding_ok else "pending",
            memory_class=resolved_class,
            wing=wing,
            room=room,
        )

        if not embedding_ok and self._pending:
            # Queue for later embedding
            await self._pending.enqueue(
                memory_id,
                content,
                memory_type,
                ",".join(resolved_tags) if resolved_tags else "",
                resolved_collection,
                source=source,
                confidence=confidence,
                source_session_id=source_session_id,
                transcript_path=transcript_path,
                source_line_range=(
                    f"{source_line_range[0]},{source_line_range[1]}"
                    if source_line_range else None
                ),
                extraction_timestamp=extraction_timestamp,
                source_pipeline=source_pipeline,
            )
        elif embedding_ok and auto_link and self._linker and vector is not None:
            await self._linker.auto_link(
                memory_id, vector, collection=resolved_collection,
            )

        return memory_id

    async def delete(self, memory_id: str) -> dict:
        """Delete a memory from all layers. Returns per-layer status."""
        results: dict[str, bool | int] = {}

        results["metadata"] = await self._metadata.delete(memory_id)
        results["fts5"] = await self._text.delete(memory_id)

        # Vector — try all known collections
        for coll in self._collections:
            try:
                await self._vector.delete(memory_id, collection=coll)
                results[f"vector_{coll}"] = True
            except Exception:
                logger.error(
                    "Vector delete failed for %s in %s", memory_id, coll,
                    exc_info=True,
                )
                results[f"vector_{coll}"] = False

        results["links_deleted"] = await self._links.delete_by_memory(memory_id)

        if self._pending:
            results["pending_deleted"] = await self._pending.delete(memory_id)

        return results
