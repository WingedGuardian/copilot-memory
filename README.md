# copilot-memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*Extracted from [GENesis-AGI](https://github.com/WingedGuardian/GENesis-AGI). Genesis's 4-layer hybrid memory (Qdrant + FTS5 + KG + relational) scaled this pattern to production; this is the standalone 2-layer core (Qdrant + FTS5 fused via RRF).*

**Hybrid AI memory system**: Qdrant vector search + SQLite FTS5 full-text search, fused via Reciprocal Rank Fusion with multi-factor scoring.

## Features

- **Dual retrieval**: Semantic search (Qdrant vectors) + exact keyword search (SQLite FTS5)
- **Multi-factor scoring**: Results scored by recency, relevance, importance, and access frequency
- **Reciprocal Rank Fusion**: Combines results from both engines into a single ranked list
- **Local-first embeddings**: Uses local LM Studio by default, falls back to cloud (OpenAI-compatible)
- **Episodic memory**: Stores conversation exchanges with session tracking and role metadata
- **Structured extraction**: Store facts, decisions, entities, and preferences with confidence scores
- **Proactive recall**: Cross-session context injection based on current conversation

## Installation

```bash
pip install copilot-memory
```

### Prerequisites

- [Qdrant](https://qdrant.tech/) running locally (`localhost:6333`)
- An embedding model (local via LM Studio or cloud via OpenAI-compatible API)

## Quick Start

```python
import asyncio
from copilot_memory import Embedder, MemoryManager

async def main():
    embedder = Embedder(
        api_base="http://localhost:1234/v1",  # LM Studio
        model="text-embedding-nomic-embed-text-v1.5",
    )
    
    memory = MemoryManager(
        embedder=embedder,
        qdrant_url="http://localhost:6333",
        db_path="memory.db",
    )
    await memory.initialize()
    
    # Store a conversation exchange
    await memory.remember_exchange(
        user_msg="What's the capital of France?",
        assistant_msg="The capital of France is Paris.",
        session_key="session_001",
    )
    
    # Recall relevant memories
    episodes = await memory.recall("French geography", limit=5)
    for ep in episodes:
        print(f"[{ep.score:.2f}] {ep.text[:100]}")
    
    # Store structured facts
    await memory.remember_extractions([
        {"category": "fact", "content": "Capital of France is Paris", "confidence": 0.95}
    ], session_key="session_001")
    
    # Get high-confidence core facts
    facts = await memory.get_high_confidence_items(min_confidence=0.8)

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              MemoryManager                   │
│  (orchestrator — hybrid recall via RRF)      │
├──────────────────┬──────────────────────────┤
│   EpisodicStore  │     FullTextStore         │
│   (Qdrant)       │     (SQLite FTS5)         │
│                  │                           │
│ • Vector search  │ • BM25 keyword search     │
│ • Multi-factor   │ • Exact phrase matching    │
│   scoring        │ • Fast prefix queries      │
│ • 768-dim embed  │ • Importance weighting     │
├──────────────────┴──────────────────────────┤
│               Embedder                       │
│  (local LM Studio → cloud OpenAI fallback)   │
└─────────────────────────────────────────────┘
```

### Multi-Factor Scoring

Each recalled memory is scored across multiple factors:

| Factor | Weight | Source |
|--------|--------|--------|
| **Relevance** | Primary | Qdrant cosine similarity / FTS5 BM25 rank |
| **Recency** | Decay curve | Timestamp-based exponential decay |
| **Importance** | Metadata | Explicitly set or derived from confidence |
| **Access frequency** | Boost | How often this memory is recalled |

Results from Qdrant and FTS5 are combined using **Reciprocal Rank Fusion (RRF)**, which merges ranked lists without needing to normalize scores across different engines.

## API Reference

### MemoryManager

| Method | Description |
|--------|-------------|
| `initialize()` | Set up Qdrant collection and FTS5 tables |
| `remember_exchange(user_msg, assistant_msg, session_key)` | Store a conversation turn |
| `remember_extractions(extractions, session_key)` | Store structured facts |
| `recall(query, limit=5)` | Hybrid search returning ranked episodes |
| `proactive_recall(text, limit=3)` | Cross-session context injection |
| `get_high_confidence_items(min_confidence)` | Core facts above threshold |
| `store_fact(text, category, importance, session_key)` | Store a single fact |
| `stats()` | Collection counts and health info |

### Embedder

| Method | Description |
|--------|-------------|
| `embed(text)` | Generate embedding vector (local → cloud fallback) |
| `embed_batch(texts)` | Batch embedding |

## License

MIT
