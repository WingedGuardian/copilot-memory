"""SQLite pending embeddings backend — implements PendingBackend protocol."""

from __future__ import annotations

import uuid

import aiosqlite


class SQLitePendingBackend:
    """Queue for memories awaiting embedding (resilience layer)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

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
        from datetime import UTC, datetime

        entry_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            "INSERT INTO pending_embeddings "
            "(id, memory_id, content, memory_type, tags, collection, created_at, "
            "source, confidence, source_session_id, transcript_path, "
            "source_line_range, extraction_timestamp, source_pipeline, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (entry_id, memory_id, content, memory_type, tags, collection, now,
             source, confidence, source_session_id, transcript_path,
             source_line_range, extraction_timestamp, source_pipeline),
        )
        await self._db.commit()

    async def get_pending(self, *, limit: int = 50) -> list[dict]:
        """Get pending memories ordered by creation time."""
        cursor = await self._db.execute(
            "SELECT id, memory_id, content, memory_type, tags, collection, "
            "created_at, source, confidence, source_session_id, transcript_path, "
            "source_line_range, extraction_timestamp, source_pipeline "
            "FROM pending_embeddings "
            "WHERE status = 'pending' "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "memory_id": r[1], "content": r[2],
                "memory_type": r[3], "tags": r[4], "collection": r[5],
                "created_at": r[6], "source": r[7], "confidence": r[8],
                "source_session_id": r[9], "transcript_path": r[10],
                "source_line_range": r[11], "extraction_timestamp": r[12],
                "source_pipeline": r[13],
            }
            for r in rows
        ]

    async def delete(self, memory_id: str) -> bool:
        """Remove a memory from the pending queue. Returns True if deleted."""
        cursor = await self._db.execute(
            "DELETE FROM pending_embeddings WHERE memory_id = ?", (memory_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
