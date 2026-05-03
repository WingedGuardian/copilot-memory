"""SQLite metadata backend — implements MetadataBackend protocol."""

from __future__ import annotations

import aiosqlite


class SQLiteMetadataBackend:
    """Stores memory metadata (creation time, classification, taxonomy)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

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
        """Insert a metadata row. Returns memory_id."""
        await self._db.execute(
            "INSERT OR IGNORE INTO memory_metadata "
            "(memory_id, created_at, collection, confidence, embedding_status, "
            "memory_class, wing, room) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (memory_id, created_at, collection, confidence, embedding_status,
             memory_class, wing, room),
        )
        await self._db.commit()
        return memory_id

    async def get(self, memory_id: str) -> dict | None:
        """Get metadata for a memory."""
        cursor = await self._db.execute(
            "SELECT memory_id, created_at, collection, confidence, "
            "embedding_status, memory_class, wing, room "
            "FROM memory_metadata WHERE memory_id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "memory_id": row[0],
            "created_at": row[1],
            "collection": row[2],
            "confidence": row[3],
            "embedding_status": row[4],
            "memory_class": row[5],
            "wing": row[6],
            "room": row[7],
        }

    async def delete(self, memory_id: str) -> bool:
        """Delete a metadata row. Returns True if deleted."""
        cursor = await self._db.execute(
            "DELETE FROM memory_metadata WHERE memory_id = ?", (memory_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
