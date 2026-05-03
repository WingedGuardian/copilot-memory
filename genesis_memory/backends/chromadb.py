"""ChromaDB vector backend — implements VectorBackend protocol.

Zero-infrastructure default: uses embedded ChromaDB with PersistentClient
(or EphemeralClient for testing). Converts ChromaDB distance to similarity
for protocol compliance (higher = more similar).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromaDBVectorBackend:
    """ChromaDB-backed vector search implementing VectorBackend protocol.

    ChromaDB returns cosine distance (lower = more similar). This backend
    converts to similarity (1.0 - distance) so the protocol convention
    (higher = more similar) is maintained.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        collections: list[str] | None = None,
        vector_dim: int = 1024,
    ) -> None:
        import chromadb

        if path is None:
            self._client = chromadb.EphemeralClient()
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))

        self._vector_dim = vector_dim
        self._collections_cache: dict[str, object] = {}

        # Pre-create requested collections
        for name in (collections or ["default"]):
            self._get_or_create_collection(name)

    def _get_or_create_collection(self, name: str):
        if name not in self._collections_cache:
            self._collections_cache[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections_cache[name]

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        collection: str = "default",
        wing: str | None = None,
        room: str | None = None,
    ) -> list[dict]:
        coll = self._get_or_create_collection(collection)

        where = None
        if wing and room:
            where = {"$and": [{"wing": wing}, {"room": room}]}
        elif wing:
            where = {"wing": wing}
        elif room:
            where = {"room": room}

        kwargs: dict = {
            "query_embeddings": [vector],
            "n_results": limit,
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            kwargs["where"] = where

        try:
            result = coll.query(**kwargs)
        except Exception as exc:
            # ChromaDB raises if collection is empty
            if "no results" in str(exc).lower() or not coll.count():
                return []
            raise

        hits: list[dict] = []
        if result["ids"] and result["ids"][0]:
            ids = result["ids"][0]
            distances = result["distances"][0] if result["distances"] else [0.0] * len(ids)
            metadatas = result["metadatas"][0] if result["metadatas"] else [{}] * len(ids)

            for point_id, distance, metadata in zip(ids, distances, metadatas, strict=True):
                # Convert distance → similarity (protocol: higher = more similar)
                # Clamp to [0, 1] — cosine distance can range [0, 2]
                score = max(0.0, 1.0 - distance)
                hits.append({
                    "id": point_id,
                    "score": score,
                    "payload": metadata or {},
                })

        return hits

    async def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict,
        *,
        collection: str = "default",
    ) -> None:
        coll = self._get_or_create_collection(collection)

        # ChromaDB metadata must be flat (str, int, float, bool)
        flat_meta = _flatten_metadata(payload)

        coll.upsert(
            ids=[point_id],
            embeddings=[vector],
            metadatas=[flat_meta],
            documents=[payload.get("content", "")],
        )

    async def update_payload(
        self,
        point_id: str,
        payload: dict,
        *,
        collection: str = "default",
    ) -> None:
        coll = self._get_or_create_collection(collection)

        # Get existing metadata, merge, update
        existing = coll.get(ids=[point_id], include=["metadatas"])
        if existing["ids"]:
            old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
            merged = {**(old_meta or {}), **_flatten_metadata(payload)}
            coll.update(ids=[point_id], metadatas=[merged])

    async def delete(self, point_id: str, *, collection: str = "default") -> None:
        coll = self._get_or_create_collection(collection)
        try:
            coll.delete(ids=[point_id])
        except Exception:
            logger.debug("ChromaDB delete failed for %s in %s", point_id, collection)

    async def scroll_tags(
        self,
        collections: list[str],
    ) -> tuple[list[list[str]], int]:
        """Iterate all points to extract tag metadata for co-occurrence index."""
        tag_lists: list[list[str]] = []
        total_count = 0

        for coll_name in collections:
            coll = self._get_or_create_collection(coll_name)
            count = coll.count()
            total_count += count

            if count == 0:
                continue

            # Paginate through all points
            batch_size = 500
            offset = 0
            while offset < count:
                result = coll.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"],
                )
                if not result["ids"]:
                    break

                for meta in result["metadatas"]:
                    if meta:
                        tags = meta.get("tags")
                        if tags:
                            # Tags stored as comma-separated string in flat metadata
                            if isinstance(tags, str):
                                tag_lists.append(tags.split(","))
                            elif isinstance(tags, list):
                                tag_lists.append(tags)

                offset += len(result["ids"])

        return tag_lists, total_count


def _flatten_metadata(payload: dict) -> dict:
    """Flatten nested payload for ChromaDB (only supports str/int/float/bool).

    Lists are joined with commas. Nested dicts are skipped. None values removed.
    """
    flat: dict = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, list):
            # Join list items as comma-separated string
            flat[key] = ",".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Skip nested dicts — ChromaDB can't store them
            logger.debug("Skipping nested dict key %r in ChromaDB metadata", key)
        else:
            flat[key] = str(value)
    return flat
