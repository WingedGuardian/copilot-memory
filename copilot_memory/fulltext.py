"""SQLite FTS5 full-text search store for episodic memory.

Complements Qdrant vector search with exact keyword matching.
Uses the existing copilot.db — no new infrastructure required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from loguru import logger


@dataclass
class FTSResult:
    """A single full-text search result."""
    id: int
    text: str
    session_key: str
    timestamp: float
    importance: float
    rank: float  # BM25 relevance score (lower = more relevant in FTS5)


class FullTextStore:
    """SQLite FTS5-backed full-text search for episodic memory."""

    TABLE = "episodic_fts"

    def __init__(self, db_path: str | Path = "data/sqlite/copilot.db"):
        self._db_path = str(db_path)

    async def ensure_table(self) -> None:
        """Create the FTS5 virtual table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.TABLE} USING fts5(
                    text,
                    session_key,
                    timestamp UNINDEXED,
                    importance UNINDEXED,
                    content={self.TABLE}_content,
                    content_rowid=id
                )
            """)
            # Also create a regular table to back the content
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE}_content (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    importance REAL DEFAULT 0.5
                )
            """)
            await db.commit()

    async def store(
        self,
        text: str,
        session_key: str,
        importance: float = 0.5,
        timestamp: float | None = None,
    ) -> int:
        """Store text for full-text search. Returns row ID."""
        ts = timestamp or time.time()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    f"INSERT INTO {self.TABLE}_content (text, session_key, timestamp, importance) "
                    "VALUES (?, ?, ?, ?)",
                    (text, session_key, ts, importance),
                )
                row_id = cursor.lastrowid
                await db.execute(
                    f"INSERT INTO {self.TABLE} (rowid, text, session_key, timestamp, importance) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row_id, text, session_key, ts, importance),
                )
                await db.commit()
                return row_id
        except Exception as e:
            logger.warning(f"FTS store failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "high", f"FTS5 insertion failed: {e}", "fts_store",)))
            return -1

    async def search(
        self,
        query: str,
        limit: int = 10,
        session_key: str | None = None,
    ) -> list[FTSResult]:
        """Full-text search using BM25 ranking.

        Args:
            query: Search query (FTS5 syntax supported).
            limit: Maximum results.
            session_key: Optional filter to a specific session.
        """
        try:
            # Escape special FTS5 characters in query
            safe_query = self._escape_query(query)
            if not safe_query.strip():
                return []

            async with aiosqlite.connect(self._db_path) as db:
                if session_key:
                    cursor = await db.execute(
                        f"""SELECT c.id, c.text, c.session_key, c.timestamp, c.importance,
                                   rank
                            FROM {self.TABLE} f
                            JOIN {self.TABLE}_content c ON c.id = f.rowid
                            WHERE {self.TABLE} MATCH ? AND c.session_key = ?
                            ORDER BY rank
                            LIMIT ?""",
                        (safe_query, session_key, limit),
                    )
                else:
                    cursor = await db.execute(
                        f"""SELECT c.id, c.text, c.session_key, c.timestamp, c.importance,
                                   rank
                            FROM {self.TABLE} f
                            JOIN {self.TABLE}_content c ON c.id = f.rowid
                            WHERE {self.TABLE} MATCH ?
                            ORDER BY rank
                            LIMIT ?""",
                        (safe_query, limit),
                    )

                rows = await cursor.fetchall()
                return [
                    FTSResult(
                        id=row[0],
                        text=row[1],
                        session_key=row[2],
                        timestamp=row[3],
                        importance=row[4],
                        rank=row[5],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.warning(f"FTS search failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"FTS5 search failed: {e}", "fts_search",)))
            return []

    @staticmethod
    def _escape_query(query: str) -> str:
        """Escape FTS5 special characters for safe matching."""
        # Remove FTS5 operators that could cause syntax errors
        # Keep alphanumeric, spaces, hyphens, underscores
        import re
        # Split into tokens, wrap each in quotes for exact matching
        tokens = re.findall(r'[\w\-]+', query)
        if not tokens:
            return ""
        return " OR ".join(f'"{t}"' for t in tokens)

    async def count(self) -> int:
        """Total number of FTS entries."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    f"SELECT COUNT(*) FROM {self.TABLE}_content"
                )
                row = await cursor.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0


def reciprocal_rank_fusion(
    vector_results: list,
    fts_results: list[FTSResult],
    k: int = 60,
) -> list[dict]:
    """Combine vector and FTS results using Reciprocal Rank Fusion.

    Args:
        vector_results: List of Episode objects from Qdrant.
        fts_results: List of FTSResult objects from SQLite FTS5.
        k: RRF constant (default 60, standard in literature).

    Returns:
        List of dicts with 'text', 'score', 'source' keys, sorted by fused score.
    """
    scores: dict[str, dict] = {}

    # Score vector results by rank position
    for rank, ep in enumerate(vector_results):
        text_key = ep.text[:200]  # Use text prefix as dedup key
        rrf_score = 1.0 / (k + rank + 1)
        if text_key in scores:
            scores[text_key]["score"] += rrf_score
            scores[text_key]["source"] = "both"
        else:
            scores[text_key] = {
                "text": ep.text,
                "score": rrf_score,
                "source": "vector",
                "episode": ep,
            }

    # Score FTS results by rank position
    for rank, fts in enumerate(fts_results):
        text_key = fts.text[:200]
        rrf_score = 1.0 / (k + rank + 1)
        if text_key in scores:
            scores[text_key]["score"] += rrf_score
            scores[text_key]["source"] = "both"
        else:
            scores[text_key] = {
                "text": fts.text,
                "score": rrf_score,
                "source": "fts",
                "fts_result": fts,
            }

    # Sort by fused score
    combined = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return combined
