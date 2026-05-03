"""Factory function to wire a complete memory system from config."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from genesis_memory.backends.schema import init_schema
from genesis_memory.backends.sqlite_links import SQLiteLinkBackend
from genesis_memory.backends.sqlite_metadata import SQLiteMetadataBackend
from genesis_memory.backends.sqlite_pending import SQLitePendingBackend
from genesis_memory.backends.sqlite_text import SQLiteTextBackend
from genesis_memory.embeddings import EmbeddingProvider
from genesis_memory.linker import MemoryLinker
from genesis_memory.protocols import VectorBackend
from genesis_memory.retrieval import HybridRetriever
from genesis_memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class MemorySystem:
    """Fully wired memory system — store, retriever, and cleanup."""

    store: MemoryStore
    retriever: HybridRetriever
    db: aiosqlite.Connection
    vector_backend: VectorBackend

    async def close(self) -> None:
        """Clean up resources."""
        await self.db.close()


async def create_memory_system(
    *,
    # Vector backend
    vector_backend: str = "chromadb",
    chromadb_path: str | Path | None = None,
    qdrant_url: str | None = None,
    # SQLite
    db_path: str | Path = ":memory:",
    # Embeddings
    embedding_backends: list | None = None,
    ollama_url: str | None = None,
    deepinfra_key: str | None = None,
    dashscope_key: str | None = None,
    # Collections
    collections: list[str] | None = None,
    vector_dim: int = 1024,
    # Options
    auto_link: bool = True,
    embedding_cache_dir: Path | None = None,
    min_confidence: float | None = None,
) -> MemorySystem:
    """Create a fully wired memory system.

    Quickstart::

        system = await create_memory_system(db_path="memory.db")
        await system.store.store("Hello world", "test")
        results = await system.retriever.recall("hello")
        await system.close()
    """
    resolved_collections = collections or ["default"]

    # 1. SQLite — wrap entire construction in try/except to prevent resource leak
    db = await aiosqlite.connect(str(db_path))
    try:
        db.row_factory = aiosqlite.Row
        await init_schema(db)

        # 2. SQLite backends
        text = SQLiteTextBackend(db)
        metadata = SQLiteMetadataBackend(db)
        links = SQLiteLinkBackend(db)
        pending = SQLitePendingBackend(db)

        # 3. Vector backend
        vec: VectorBackend
        if vector_backend == "chromadb":
            from genesis_memory.backends.chromadb import ChromaDBVectorBackend

            vec = ChromaDBVectorBackend(
                path=chromadb_path,
                collections=resolved_collections,
                vector_dim=vector_dim,
            )
        elif vector_backend == "qdrant":
            from genesis_memory.backends.qdrant import QdrantVectorBackend

            if not qdrant_url:
                msg = "qdrant_url required when vector_backend='qdrant'"
                raise ValueError(msg)
            vec = QdrantVectorBackend(url=qdrant_url, collections=resolved_collections)
        else:
            msg = f"Unknown vector_backend: {vector_backend!r}. Use 'chromadb' or 'qdrant'."
            raise ValueError(msg)

        # 4. Embedding provider
        if embedding_backends:
            backends = embedding_backends
        else:
            backends = []
            if ollama_url:
                from genesis_memory.embedding_backends.ollama import OllamaBackend

                backends.append(OllamaBackend(url=ollama_url))
            if deepinfra_key:
                from genesis_memory.embedding_backends.deepinfra import DeepInfraBackend

                backends.append(DeepInfraBackend(api_key=deepinfra_key))
            if dashscope_key:
                from genesis_memory.embedding_backends.dashscope import DashScopeBackend

                backends.append(DashScopeBackend(api_key=dashscope_key))
            if not backends:
                logger.warning(
                    "No embedding backends configured. Provide ollama_url, deepinfra_key, "
                    "or pass embedding_backends directly."
                )

        embeddings = EmbeddingProvider(backends=backends, cache_dir=embedding_cache_dir)

        # 5. Linker
        linker: MemoryLinker | None = None
        if auto_link:
            linker = MemoryLinker(
                vector_backend=vec,
                text_backend=text,
                link_backend=links,
            )

        # 6. Store
        store = MemoryStore(
            embedding_provider=embeddings,
            vector_backend=vec,
            text_backend=text,
            metadata_backend=metadata,
            link_backend=links,
            pending_backend=pending,
            linker=linker,
            collections=resolved_collections,
            min_confidence=min_confidence,
        )

        # 7. Retriever
        retriever = HybridRetriever(
            embedding_provider=embeddings,
            vector_backend=vec,
            text_backend=text,
            link_backend=links,
        )
    except Exception:
        await db.close()
        raise

    return MemorySystem(
        store=store,
        retriever=retriever,
        db=db,
        vector_backend=vec,
    )
