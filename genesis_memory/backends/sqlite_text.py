"""SQLite FTS5 text backend — implements TextBackend protocol."""

from __future__ import annotations

import re

import aiosqlite


def _prepare_fts5(query: str, *, boolean: bool = False) -> str | None:
    """Prepare a query string for FTS5 MATCH.

    Default path: lowercases to neutralize accidental FTS5 boolean operators.
    Boolean path: preserves OR/AND keywords for structured queries.

    Returns None if the query is empty after escaping (caller should return []).
    """
    if boolean:
        cleaned = re.sub(r'[^\w\s()]', " ", query, flags=re.UNICODE).strip()
        if cleaned.count("(") != cleaned.count(")"):
            cleaned = cleaned.replace("(", " ").replace(")", " ").strip()
    else:
        cleaned = re.sub(r'[^\w\s]', " ", query.lower(), flags=re.UNICODE).strip()
    return cleaned or None


class SQLiteTextBackend:
    """FTS5-backed text search implementing TextBackend protocol."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def search_ranked(
        self,
        query: str,
        *,
        collection: str | None = None,
        limit: int = 30,
        boolean: bool = False,
    ) -> list[dict]:
        """FTS5 search returning rank scores for RRF fusion."""
        escaped = _prepare_fts5(query, boolean=boolean)
        if not escaped:
            return []
        sql = (
            "SELECT memory_id, content, source_type, collection, rank "
            "FROM memory_fts WHERE memory_fts MATCH ?"
        )
        params: list = [escaped]
        if collection:
            sql += " AND collection = ?"
            params.append(collection)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            {
                "memory_id": r[0], "content": r[1], "source_type": r[2],
                "collection": r[3], "rank": r[4],
            }
            for r in rows
        ]

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict]:
        """Basic FTS5 search."""
        escaped = _prepare_fts5(query)
        if not escaped:
            return []
        sql = (
            "SELECT memory_id, content, source_type, collection "
            "FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?"
        )
        cursor = await self._db.execute(sql, [escaped, limit])
        rows = await cursor.fetchall()
        return [
            {"memory_id": r[0], "content": r[1], "source_type": r[2], "collection": r[3]}
            for r in rows
        ]

    async def upsert(
        self,
        memory_id: str,
        content: str,
        *,
        source_type: str = "memory",
        tags: str = "",
        collection: str = "default",
    ) -> str:
        """Idempotent write: delete-then-insert for FTS5 (no ON CONFLICT)."""
        await self._db.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
        await self._db.execute(
            "INSERT INTO memory_fts (memory_id, content, source_type, tags, collection) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, content, source_type, tags, collection),
        )
        await self._db.commit()
        return memory_id

    async def find_exact_duplicate(self, content: str) -> str | None:
        """Return memory_id if exact content already exists."""
        if not content:
            return None
        prefix = content[:200]
        cursor = await self._db.execute(
            "SELECT memory_id, content FROM memory_fts "
            "WHERE length(content) = ? "
            "AND substr(content, 1, 200) = ? "
            "LIMIT 200",
            (len(content), prefix),
        )
        for row in await cursor.fetchall():
            if row[1] == content:
                return row[0]
        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry from the FTS5 index."""
        cursor = await self._db.execute(
            "DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
