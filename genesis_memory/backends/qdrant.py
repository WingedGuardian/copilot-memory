"""Qdrant vector backend — implements VectorBackend protocol.

Requires ``qdrant-client`` (optional dependency, imported at runtime).
Cosine similarity scores from Qdrant are already normalized to [0, 1],
so no distance-to-similarity conversion is needed (unlike ChromaDB).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class QdrantVectorBackend:
    """Qdrant-backed vector search implementing VectorBackend protocol.

    Qdrant returns cosine similarity directly (higher = more similar),
    which already matches the protocol convention.
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collections: list[str] | None = None,
        vector_dim: int = 1024,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._vector_dim = vector_dim

        # Ensure requested collections exist
        existing = {c.name for c in self._client.get_collections().collections}
        for name in collections or ["default"]:
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_dim, distance=Distance.COSINE
                    ),
                )

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        collection: str = "default",
        wing: str | None = None,
        room: str | None = None,
    ) -> list[dict]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions: list = []
        if wing:
            conditions.append(
                FieldCondition(key="wing", match=MatchValue(value=wing))
            )
        if room:
            conditions.append(
                FieldCondition(key="room", match=MatchValue(value=room))
            )
        query_filter = Filter(must=conditions) if conditions else None

        results = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        )
        return [
            {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
            for hit in results.points
        ]

    async def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict,
        *,
        collection: str = "default",
    ) -> None:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def update_payload(
        self,
        point_id: str,
        payload: dict,
        *,
        collection: str = "default",
    ) -> None:
        self._client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[point_id],
        )

    async def delete(self, point_id: str, *, collection: str = "default") -> None:
        from qdrant_client.models import PointIdsList

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=[point_id]),
            )
        except Exception:
            logger.debug("Qdrant delete failed for %s in %s", point_id, collection)

    async def scroll_tags(
        self,
        collections: list[str],
    ) -> tuple[list[list[str]], int]:
        """Iterate all points to extract tag metadata for co-occurrence index.

        Uses qdrant_client.scroll() to paginate through points and extract
        the ``tags`` field from each payload.
        """
        tag_lists: list[list[str]] = []
        total_count = 0

        for coll_name in collections:
            # Get collection point count
            try:
                info = self._client.get_collection(collection_name=coll_name)
                coll_count = info.points_count or 0
            except Exception:
                logger.debug("Qdrant scroll_tags: collection %s not found", coll_name)
                continue

            total_count += coll_count
            if coll_count == 0:
                continue

            # Paginate through all points using scroll
            offset = None
            batch_size = 500
            while True:
                points, next_offset = self._client.scroll(
                    collection_name=coll_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    tags = payload.get("tags")
                    if tags:
                        if isinstance(tags, str):
                            tag_lists.append(tags.split(","))
                        elif isinstance(tags, list):
                            tag_lists.append(tags)

                if next_offset is None:
                    break
                offset = next_offset

        return tag_lists, total_count
