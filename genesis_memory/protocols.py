"""Backend protocols for the genesis-memory storage layer.

Each protocol abstracts a storage concern, allowing swappable backends:
- VectorBackend: Qdrant, ChromaDB, or any vector store
- TextBackend: SQLite FTS5 (or any full-text search)
- MetadataBackend: memory metadata (creation time, classification, etc.)
- LinkBackend: memory-to-memory relationships (knowledge graph edges)
- PendingBackend: queue for memories awaiting embedding (resilience layer)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Abstracts vector similarity search (Qdrant, ChromaDB, etc.)."""

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        collection: str = "default",
        wing: str | None = None,
        room: str | None = None,
    ) -> list[dict]:
        """Search by vector similarity.

        Returns list of {id: str, score: float, payload: dict}.
        Score convention: higher = more similar (backends normalize internally).
        """
        ...

    async def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict,
        *,
        collection: str = "default",
    ) -> None:
        """Insert or update a vector point."""
        ...

    async def update_payload(
        self,
        point_id: str,
        payload: dict,
        *,
        collection: str = "default",
    ) -> None:
        """Update metadata payload for an existing point."""
        ...

    async def delete(self, point_id: str, *, collection: str = "default") -> None:
        """Delete a vector point by ID."""
        ...

    async def scroll_tags(
        self,
        collections: list[str],
    ) -> tuple[list[list[str]], int]:
        """Iterate all points to extract tag metadata for co-occurrence index.

        Returns (tag_lists, total_point_count) where each element in tag_lists
        is the list of tags from one memory point.
        """
        ...


@runtime_checkable
class TextBackend(Protocol):
    """Abstracts full-text search (SQLite FTS5, etc.)."""

    async def search_ranked(
        self,
        query: str,
        *,
        collection: str | None = None,
        limit: int = 30,
        boolean: bool = False,
    ) -> list[dict]:
        """FTS5 ranked search.

        Returns list of {memory_id, content, source_type, collection, rank}.
        """
        ...

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict]:
        """Basic text search. Returns list of {memory_id, content, source_type, collection}."""
        ...

    async def upsert(
        self,
        memory_id: str,
        content: str,
        *,
        source_type: str = "memory",
        tags: str = "",
        collection: str = "default",
    ) -> str:
        """Insert or update a text entry. Returns memory_id."""
        ...

    async def find_exact_duplicate(self, content: str) -> str | None:
        """Check if content already exists. Returns memory_id or None."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Delete a text entry. Returns True if deleted."""
        ...


@runtime_checkable
class MetadataBackend(Protocol):
    """Abstracts memory metadata storage."""

    async def create(
        self,
        memory_id: str,
        *,
        created_at: str,
        collection: str = "default",
        confidence: float | None = None,
        embedding_status: str = "embedded",
        memory_class: str = "fact",
        wing: str | None = None,
        room: str | None = None,
    ) -> str:
        """Create a metadata record. Returns memory_id."""
        ...

    async def get(self, memory_id: str) -> dict | None:
        """Get metadata for a memory. Returns dict or None."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Delete metadata. Returns True if deleted."""
        ...


@runtime_checkable
class LinkBackend(Protocol):
    """Abstracts memory link (knowledge graph edge) storage."""

    async def create(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        strength: float,
        created_at: str,
    ) -> tuple[str, str]:
        """Create a link between two memories. Returns (source_id, target_id)."""
        ...

    async def count_links(self, memory_id: str) -> int:
        """Count outbound links from a memory."""
        ...

    async def delete_by_memory(self, memory_id: str) -> int:
        """Delete all links involving a memory (source or target). Returns count deleted."""
        ...


@runtime_checkable
class PendingBackend(Protocol):
    """Queue for memories awaiting embedding (resilience layer).

    When embedding is unavailable, memories are stored FTS5-only and queued
    here for later vector embedding when the provider recovers.
    """

    async def enqueue(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        tags: str,
        collection: str,
        *,
        source: str = "",
        confidence: float | None = None,
        source_session_id: str | None = None,
        transcript_path: str | None = None,
        source_line_range: str | None = None,
        extraction_timestamp: str | None = None,
        source_pipeline: str | None = None,
    ) -> None:
        """Queue a memory for later embedding."""
        ...

    async def get_pending(self, *, limit: int = 50) -> list[dict]:
        """Get pending memories ordered by creation time."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Remove a memory from the pending queue. Returns True if deleted."""
        ...
