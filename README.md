# genesis-memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Production-grade memory system with 12-step RRF hybrid retrieval, extracted from the [Genesis autonomous agent](https://github.com/WingedGuardian/GENesis-AGI). This isn't a tutorial — it's the real retrieval engine that runs 24/7 in production, handling thousands of memories with sub-second recall.

## What Makes This Different

Most memory libraries do vector search. Some add keyword search. Genesis Memory does both — and fuses them with **Reciprocal Rank Fusion**, activation scoring, intent classification, and a knowledge graph. The result: retrieval that understands *what you're asking*, not just *what's similar*.

### The 12-Step Retrieval Pipeline

```
Query → Embed → Vector Search → Intent Classification → Query Expansion
  → FTS5 Search → Union Candidates → Activation Scoring
  → Build Ranked Lists → RRF Fusion → Filter → Results
```

Every memory goes through this pipeline on recall. Vector search finds semantically similar memories. FTS5 catches exact keyword matches that vectors miss. Activation scoring prioritizes recent, frequently-accessed, well-connected memories. Intent classification biases results toward the *type* of answer you need (WHY queries surface reflections, HOW queries surface procedures). RRF fuses all four ranked lists into a single, well-ordered result.

## Features

### Retrieval Intelligence
- **12-step RRF pipeline** — vector + FTS5 + activation + intent, fused via Reciprocal Rank Fusion
- **Intent-aware retrieval** — classifies queries as WHAT/WHY/HOW/WHEN/WHERE/STATUS and biases results accordingly
- **Category-aware activation decay** — memories decay at different rates based on their source (conversation extractions last longer than routine observations)
- **Entity recognition** — proper noun tags (capitalized) get 2x half-life because "AgentMail is an email service" doesn't become less true over time
- **Query expansion** — tag co-occurrence index broadens FTS5 recall for oblique references

### Memory Classification
- **Rule / Fact / Reference** — automatic classification with activation weight multipliers (rules get 1.3x boost — they prevent mistakes)
- **Wing/Room taxonomy** — organize memories into domains (infrastructure, learning, routing, etc.) with automatic classification

### Knowledge Graph
- **Auto-linking** — new memories automatically link to similar existing memories (similarity threshold, typed relationships)
- **12 link types** — supports, contradicts, extends, elaborates, decided, action_item_for, and more
- **Recursive CTE traversal** — traverse relationships, find clusters, query by link type
- **Bidirectional clustering** — find the full connected component around any memory

### Resilience
- **FTS5-only fallback** — if embedding is unavailable, memories store to FTS5 and queue for later vector embedding
- **Pending embeddings queue** — recovery worker can drain the queue when the embedding provider comes back
- **Multi-backend embedding chain** — Ollama → DeepInfra → DashScope, with automatic fallback
- **Two-level embedding cache** — L1 in-process dict (24h TTL) + L2 diskcache on disk (7 days, shared across processes)
- **Deduplication** — exact content match prevents duplicate storage

### Zero Infrastructure
- **ChromaDB default** — `pip install genesis-memory` gives you a working system with no external services
- **Qdrant optional** — `pip install genesis-memory[qdrant]` for production deployments
- **SQLite FTS5** — built into Python's sqlite3 module, no setup needed
- **Pluggable everything** — 5 backend protocols (vector, text, metadata, links, pending) for custom implementations

## Installation

```bash
pip install genesis-memory
```

For production with Qdrant:
```bash
pip install genesis-memory[qdrant]
```

## Quick Start

```python
import asyncio
from genesis_memory import create_memory_system

async def main():
    # Create a fully wired memory system (ChromaDB + SQLite, zero config)
    system = await create_memory_system(
        db_path="memory.db",
        chromadb_path="./vectors",
        ollama_url="http://localhost:11434",  # or pass deepinfra_key="sk-..."
    )

    # Store memories — classification, taxonomy, and linking happen automatically
    await system.store.store(
        "Circuit breakers protect against cascading failures by isolating failing providers",
        "documentation",
        tags=["routing", "resilience"],
    )
    await system.store.store(
        "You MUST NEVER push directly to main without a pull request",
        "team_rules",  # auto-classified as "rule" → gets 1.3x activation boost
    )

    # Recall with full RRF pipeline
    results = await system.retriever.recall(
        "how do we handle provider failures?",
        limit=5,
    )

    for r in results:
        print(f"[{r.memory_class}] (score={r.score:.4f}) {r.content[:80]}")
        print(f"  vector_rank={r.vector_rank}, fts_rank={r.fts_rank}")
        print(f"  activation={r.activation_score:.4f}, intent={r.query_intent}")

    await system.close()

asyncio.run(main())
```

## Architecture

### Backend Protocols

Every storage concern is abstracted behind a protocol, making backends swappable:

| Protocol | Default Backend | Purpose |
|----------|----------------|---------|
| `VectorBackend` | ChromaDB | Semantic similarity search |
| `TextBackend` | SQLite FTS5 | Full-text keyword search |
| `MetadataBackend` | SQLite | Memory metadata (timestamps, classification) |
| `LinkBackend` | SQLite | Knowledge graph edges |
| `PendingBackend` | SQLite | Embedding recovery queue |

### Activation Scoring

```
final = confidence × recency × (0.5 + 0.3×access + 0.2×connectivity) × class_weight
```

- **Recency**: Exponential decay with category-aware half-lives (30-60 days)
- **Access frequency**: Log-scaled retrieval count (saturates at ~20 accesses)
- **Connectivity**: Log-scaled link count (saturates at ~10 links)
- **Class weight**: rule=1.3, fact=1.0, reference=0.7

### RRF Fusion

RRF is deliberately score-agnostic — it only uses rank positions, never absolute scores. This means ChromaDB distances and Qdrant similarities produce identical fusion results for the same data:

```python
score[id] += 1.0 / (k + rank)  # k=60, summed across all ranked lists
```

Up to 4 ranked lists are fused: vector, FTS5, activation, and intent-biased.

## Embedding Backends

The embedding provider chains multiple backends with automatic fallback:

| Backend | Type | Use Case |
|---------|------|----------|
| `OllamaBackend` | Local | Cost-free, GPU-accelerated (qwen3-embedding) |
| `DeepInfraBackend` | Cloud | Low-latency cloud inference |
| `DashScopeBackend` | Cloud | Alibaba's text-embedding-v4 |
| `OpenAICompatBackend` | Any | Works with any OpenAI-compatible API |

```python
from genesis_memory.embedding_backends.ollama import OllamaBackend
from genesis_memory.embedding_backends.deepinfra import DeepInfraBackend
from genesis_memory import EmbeddingProvider

# Storage path: local first (cost-optimized)
provider = EmbeddingProvider(backends=[
    OllamaBackend(url="http://localhost:11434"),
    DeepInfraBackend(api_key="sk-..."),
])
```

## Advanced Usage

### Graph Traversal

```python
from genesis_memory.graph import traverse, get_cluster, find_connected_by_type

# Traverse from a memory, following links up to 3 hops
result = await traverse(db, root_id="mem-abc", max_depth=3, min_strength=0.5)
for node in result.nodes:
    print(f"  depth={node.depth} {node.link_type} → {node.memory_id}")

# Find everything connected (bidirectional)
cluster = await get_cluster(db, root_id="mem-abc")

# Find all decisions related to a memory
decisions = await find_connected_by_type(db, "mem-abc", "decided")
```

### Custom Backends

Implement any protocol to plug in your own storage:

```python
from genesis_memory import VectorBackend

class PineconeBackend:
    """Custom Pinecone implementation."""

    async def search(self, vector, *, limit=10, collection="default", **kw):
        # Your Pinecone search logic
        return [{"id": "...", "score": 0.95, "payload": {...}}]

    async def upsert(self, point_id, vector, payload, *, collection="default"):
        # Your Pinecone upsert logic
        ...
    # ... implement remaining protocol methods
```

## Extracted From Genesis

This package contains the real retrieval algorithms from [Genesis](https://github.com/WingedGuardian/GENesis-AGI), an autonomous AI agent that remembers everything, learns from every interaction, and thinks between conversations. The RRF pipeline, activation scoring, intent classification, and knowledge graph traversal are production code running 24/7 — not a tutorial or reference implementation.

What was removed: Genesis-specific orchestrators (session extraction, user model synthesis, essential knowledge generation) and database coupling (replaced with pluggable backend protocols). What was preserved: every algorithm, every scoring formula, every heuristic.

## License

MIT
