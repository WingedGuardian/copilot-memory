"""SQLite links backend — implements LinkBackend protocol."""

from __future__ import annotations

import aiosqlite


class SQLiteLinkBackend:
    """Stores memory-to-memory relationships (knowledge graph edges)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        strength: float,
        created_at: str,
    ) -> tuple[str, str]:
        """Insert a memory link. Returns (source_id, target_id)."""
        await self._db.execute(
            "INSERT INTO memory_links (source_id, target_id, link_type, strength, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, link_type, strength, created_at),
        )
        await self._db.commit()
        return (source_id, target_id)

    async def count_links(self, memory_id: str) -> int:
        """Count links where memory_id is source or target."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM memory_links WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        row = await cursor.fetchone()
        return row[0]

    async def delete_by_memory(self, memory_id: str) -> int:
        """Delete ALL links involving a memory. Returns count deleted."""
        cursor = await self._db.execute(
            "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        await self._db.commit()
        return cursor.rowcount
