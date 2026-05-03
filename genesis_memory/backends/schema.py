"""DDL for the genesis-memory SQLite tables.

Creates FTS5, metadata, links, and pending_embeddings tables.
Call ``init_schema(db)`` once on startup.
"""

from __future__ import annotations

import aiosqlite

_DDL = [
    # FTS5 virtual table for full-text search
    """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        memory_id UNINDEXED,
        content,
        source_type,
        tags,
        collection UNINDEXED,
        tokenize='porter ascii'
    )""",
    # Companion metadata table
    """CREATE TABLE IF NOT EXISTS memory_metadata (
        memory_id        TEXT PRIMARY KEY,
        created_at       TEXT NOT NULL,
        collection       TEXT NOT NULL DEFAULT 'default',
        confidence       REAL,
        embedding_status TEXT NOT NULL DEFAULT 'embedded',
        memory_class     TEXT DEFAULT 'fact',
        wing             TEXT,
        room             TEXT
    )""",
    # Knowledge graph edges
    """CREATE TABLE IF NOT EXISTS memory_links (
        source_id   TEXT NOT NULL,
        target_id   TEXT NOT NULL,
        link_type   TEXT NOT NULL CHECK (
            link_type IN (
                'supports','contradicts','extends','elaborates',
                'discussed_in','evaluated_for','decided',
                'action_item_for','categorized_as','related_to',
                'succeeded_by','preceded_by'
            )
        ),
        strength    REAL NOT NULL DEFAULT 0.5,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (source_id, target_id)
    )""",
    # Pending embeddings queue (resilience layer)
    """CREATE TABLE IF NOT EXISTS pending_embeddings (
        id                   TEXT PRIMARY KEY,
        memory_id            TEXT NOT NULL,
        content              TEXT NOT NULL,
        memory_type          TEXT NOT NULL DEFAULT 'episodic',
        tags                 TEXT DEFAULT '',
        collection           TEXT DEFAULT 'default',
        created_at           TEXT NOT NULL,
        source               TEXT DEFAULT '',
        confidence           REAL,
        source_session_id    TEXT,
        transcript_path      TEXT,
        source_line_range    TEXT,
        extraction_timestamp TEXT,
        source_pipeline      TEXT,
        status               TEXT DEFAULT 'pending'
    )""",
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_metadata_created ON memory_metadata(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_memory_metadata_collection ON memory_metadata(collection)",
    "CREATE INDEX IF NOT EXISTS idx_memory_metadata_wing ON memory_metadata(wing)",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links(target_id)",
    "CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_embeddings(status)",
]


async def init_schema(db: aiosqlite.Connection) -> None:
    """Create all memory tables and indexes."""
    for ddl in _DDL:
        await db.execute(ddl)
    for idx in _INDEXES:
        await db.execute(idx)
    await db.commit()
