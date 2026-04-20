"""Memory manager: orchestrates Qdrant (episodic) + SQLite (structured + FTS5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

from copilot_memory.embedder import Embedder
from copilot_memory.episodic import Episode, EpisodicStore
from copilot_memory.fulltext import FullTextStore


class MemoryManager:
    """Orchestrates Qdrant episodic store and SQLite structured items + FTS5."""

    def __init__(
        self,
        embedder: Embedder,
        qdrant_url: str = "http://localhost:6333",
        db_path: str | Path = "data/sqlite/copilot.db",
        dimensions: int = 768,
    ):
        self._embedder = embedder
        self._episodic = EpisodicStore(embedder, qdrant_url, dimensions=dimensions)
        self._fts = FullTextStore(db_path)
        self._db_path = str(db_path)
        self._slm_queue: Any = None  # Set by commands.py for deferred re-embedding

    async def initialize(self) -> None:
        """Connect to all backends."""
        try:
            await self._episodic._ensure_client()
        except Exception as e:
            logger.warning(f"Qdrant init failed (degraded mode): {e}")
            logger.warning(f"Alert: " + str(("memory", "high", f"Qdrant init failed: {e}", "qdrant_init",)))
        try:
            await self._fts.ensure_table()
        except Exception as e:
            logger.warning(f"FTS5 init failed (degraded mode): {e}")
            logger.warning(f"Alert: " + str(("memory", "high", f"FTS5 init failed: {e}", "fts_init",)))

    async def remember_exchange(
        self, user_msg: str, assistant_msg: str, session_key: str
    ) -> str:
        """Store a full user/assistant exchange as episodic memory."""
        combined = f"User: {user_msg}\nAssistant: {assistant_msg}"
        try:
            point_id = await self._episodic.store(
                text=combined, session_key=session_key, role="exchange",
            )
        except Exception as e:
            logger.warning(f"Exchange storage failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"Exchange storage failed: {e}", "remember_exchange",)))
            point_id = ""

        # Queue re-embedding only on double failure (local AND cloud both down)
        if self._slm_queue and self._embedder._last_zero:
            try:
                await self._slm_queue.enqueue_embedding(
                    text=combined, session_key=session_key, role="exchange",
                )
            except Exception:
                pass

        # Also write to FTS5 for keyword search
        try:
            await self._fts.store(combined, session_key)
        except Exception as e:
            logger.warning(f"FTS exchange storage failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"FTS exchange storage failed: {e}", "fts_store",)))

        return point_id

    async def remember_extractions(
        self, extractions: dict[str, Any], session_key: str,
        conversation_ts: float | None = None,
    ) -> list[str]:
        """Store extractions to Qdrant + FTS5 + SQLite structured items."""
        tags = extractions.get("tags", [])
        ids = []
        try:
            ids = await self._episodic.store_extractions(
                extractions, session_key, conversation_ts=conversation_ts,
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Extraction storage failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"Extraction storage failed: {e}", "extraction_store",)))

        # Queue re-embedding only on double failure (local AND cloud both down)
        if self._slm_queue and self._embedder._last_zero:
            for key in ("facts", "decisions", "constraints", "entities"):
                for item in extractions.get(key, []):
                    try:
                        await self._slm_queue.enqueue_embedding(
                            text=item, session_key=session_key,
                            role={"facts": "fact", "decisions": "decision", "constraints": "constraint", "entities": "entity"}[key], importance=0.8,
                            conversation_ts=conversation_ts,
                        )
                    except Exception:
                        pass

        # Also write extractions to FTS5
        fts_failures = 0
        for key in ("facts", "decisions", "constraints", "entities"):
            for item in extractions.get(key, []):
                try:
                    await self._fts.store(item, session_key, importance=0.8)
                except Exception:
                    fts_failures += 1
        if fts_failures:
            logger.warning(f"FTS extraction storage: {fts_failures} item(s) failed")
            logger.warning(f"Alert: " + str(("memory", "medium", f"FTS extraction storage: {fts_failures} item(s) failed", "fts_extraction",)))

        # Also upsert into SQLite structured items
        for key in ("facts", "decisions", "entities"):
            for item in extractions.get(key, []):
                category = {"facts": "fact", "decisions": "decision", "entities": "entity"}[key]
                await self._upsert_item(
                    category=category,
                    key=item[:100],
                    value=item,
                    session_key=session_key,
                    source="extraction",
                    tags=tags,
                )
        return ids

    async def recall(
        self, query: str, session_key: str, limit: int = 5
    ) -> list[Episode]:
        """Hybrid search across Qdrant + FTS5."""
        return await self._episodic.recall_hybrid(
            query, self._fts, limit=limit, session_key=session_key
        )

    async def proactive_recall(
        self, current_message: str, session_key: str, limit: int = 3
    ) -> str:
        """Anticipate needed context and format as injection block.

        Searches across all sessions using hybrid search.
        """
        episodes = await self._episodic.recall_hybrid(
            current_message, self._fts, limit=limit, session_key=None
        )
        if not episodes:
            return ""

        return self._format_for_injection(episodes, budget_tokens=200)

    @staticmethod
    def _format_for_injection(episodes: list[Episode], budget_tokens: int = 800) -> str:
        """Format recalled episodes as context injection block."""
        if not episodes:
            return ""

        lines = ["## Recalled Memories"]
        total_chars = 0
        for ep in episodes:
            text = ep.text
            # Rough budget: 4 chars per token
            if total_chars + len(text) > budget_tokens * 4:
                text = text[:200] + "..."
            score_pct = int(ep.score * 100)
            lines.append(f"- [{score_pct}%] {text}")
            total_chars += len(text)
            if total_chars > budget_tokens * 4:
                break

        return "\n".join(lines)

    async def _upsert_item(
        self,
        category: str,
        key: str,
        value: str,
        session_key: str,
        source: str,
        tier: str = "domain",
        tags: list[str] | None = None,
        confidence: float | None = None,
    ) -> None:
        """SQLite INSERT ON CONFLICT with confidence boost."""
        import json as _json
        tags_json = _json.dumps(tags or [])
        try:
            async with aiosqlite.connect(self._db_path) as db:
                if confidence is not None:
                    await db.execute(
                        """INSERT INTO memory_items (category, key, value, session_key, source, tier, tags, confidence)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(category, key) DO UPDATE SET
                               value = excluded.value,
                               confidence = MAX(confidence, excluded.confidence),
                               access_count = access_count + 1,
                               tier = CASE WHEN excluded.tier = 'domain' THEN tier ELSE excluded.tier END,
                               tags = CASE WHEN excluded.tags = '[]' THEN tags ELSE excluded.tags END,
                               updated_at = CURRENT_TIMESTAMP""",
                        (category, key, value, session_key, source, tier, tags_json, confidence),
                    )
                else:
                    await db.execute(
                        """INSERT INTO memory_items (category, key, value, session_key, source, tier, tags)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(category, key) DO UPDATE SET
                               value = excluded.value,
                               confidence = MIN(confidence + 0.1, 1.0),
                               access_count = access_count + 1,
                               tier = CASE WHEN excluded.tier = 'domain' THEN tier ELSE excluded.tier END,
                               tags = CASE WHEN excluded.tags = '[]' THEN tags ELSE excluded.tags END,
                               updated_at = CURRENT_TIMESTAMP""",
                        (category, key, value, session_key, source, tier, tags_json),
                    )
                await db.commit()
        except Exception as e:
            logger.warning(f"Memory item upsert failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"Structured memory write failed: {e}", "upsert_item",)))

    async def get_high_confidence_items(
        self, min_confidence: float = 0.6, limit: int = 20, tier: str | None = None,
    ) -> list[dict]:
        """Fetch high-confidence structured memory items."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                if tier:
                    cursor = await db.execute(
                        """SELECT category, key, value, confidence, access_count
                           FROM memory_items WHERE confidence >= ? AND tier = ?
                           ORDER BY confidence DESC LIMIT ?""",
                        (min_confidence, tier, limit),
                    )
                else:
                    cursor = await db.execute(
                        """SELECT category, key, value, confidence, access_count
                           FROM memory_items WHERE confidence >= ?
                           ORDER BY confidence DESC LIMIT ?""",
                        (min_confidence, limit),
                    )
                rows = await cursor.fetchall()
                cols = [d[0] for d in cursor.description]
                return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.warning(f"Memory items query failed: {e}")
            return []

    async def store_fact(
        self,
        content: str,
        category: str,
        session_key: str,
        tier: str = "domain",
        tags: list[str] | None = None,
    ) -> str:
        """Store an explicit fact to all backends (searchable immediately).

        Agent-deliberate stores get importance=0.9 (outranks passive extraction at 0.8)
        and confidence=0.7 (above extraction default of 0.5).
        """
        # 1. SQLite memory_items (structured, confidence-tracked)
        await self._upsert_item(
            category=category, key=content[:100], value=content,
            session_key=session_key, source="agent",
            tier=tier, tags=tags, confidence=0.7,
        )
        # 2. Qdrant (semantic search) — importance=0.9 for agent-explicit
        point_id = ""
        try:
            point_id = await self._episodic.store(
                text=f"[{category}] {content}",
                session_key=session_key, role=category,
                importance=0.9,
                metadata={"tags": tags or []},
            )
        except Exception as e:
            logger.warning(f"Episodic store_fact failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"store_fact Qdrant failed: {e}", "store_fact_qdrant",)))
        # 3. FTS5 (keyword search)
        try:
            await self._fts.store(
                f"[{category}] {content}",
                session_key, importance=0.9,
            )
        except Exception as e:
            logger.warning(f"FTS store_fact failed: {e}")
            logger.warning(f"Alert: " + str(("memory", "medium", f"store_fact FTS failed: {e}", "store_fact_fts",)))
        return point_id

    async def get_core_facts_block(self, budget_tokens: int = 200) -> str:
        """Format high-confidence core-tier items as a system prompt block."""
        items = await self.get_high_confidence_items(
            min_confidence=0.8, limit=10, tier="core",
        )
        if not items:
            return ""
        lines = ["## Core Facts"]
        total_chars = 0
        for item in items:
            line = f"- [{item['category']}] {item['value']}"
            if total_chars + len(line) > budget_tokens * 4:
                break
            lines.append(line)
            total_chars += len(line)
        return "\n".join(lines)

    async def health(self) -> dict[str, bool]:
        """Check health of all memory backends."""
        qdrant_ok = False
        try:
            await self._episodic.count()
            qdrant_ok = True
        except Exception:
            pass
        return {"qdrant": qdrant_ok}
