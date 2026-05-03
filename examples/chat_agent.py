#!/usr/bin/env python3
"""Interactive chat agent demonstrating genesis-memory's full RRF retrieval pipeline.

No API keys required -- uses a deterministic mock embedding backend that generates
consistent vectors from text hashing. This means semantically similar text (sharing
words) will produce similar vectors, giving a realistic demo of the hybrid pipeline.

Usage:
    python examples/chat_agent.py

Commands:
    <any text>              Search memories via the 12-step RRF pipeline
    /store <content>        Store a new memory
    /graph <memory_id>      Traverse the knowledge graph from a memory
    /stats                  Show embedding cache and system stats
    /memories               List all stored memories
    /help                   Show available commands
    /quit                   Exit

Requires: pip install genesis-memory[demo]
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
from datetime import UTC, datetime, timedelta

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Mock embedding backend -- deterministic vectors from text, no API keys
# ---------------------------------------------------------------------------

_VECTOR_DIM = 128  # Smaller than production (1024) for speed


class MockEmbeddingBackend:
    """Deterministic embedding backend that produces vectors from text hashing.

    Generates reproducible vectors where texts sharing words produce similar
    vectors (via additive word hashing). Good enough to demonstrate vector
    search ranking without any external service.
    """

    @property
    def name(self) -> str:
        return "mock-hasher"

    async def embed(self, text: str) -> list[float]:
        return _text_to_vector(text)

    async def is_available(self) -> bool:
        return True


def _word_hash_vector(word: str) -> list[float]:
    """Hash a single word into a unit vector of _VECTOR_DIM dimensions.

    Uses iterative SHA-256 hashing to fill the full vector dimension,
    since a single digest only gives 32 bytes.
    """
    raw: list[float] = []
    counter = 0
    while len(raw) < _VECTOR_DIM:
        digest = hashlib.sha256(f"{word.lower()}:{counter}".encode()).digest()
        raw.extend((b / 255.0) * 2.0 - 1.0 for b in digest)
        counter += 1
    raw = raw[:_VECTOR_DIM]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _text_to_vector(text: str) -> list[float]:
    """Convert text to a vector by averaging word hash vectors.

    Texts that share words will produce similar vectors -- this gives
    the demo realistic vector search behavior without a real model.
    """
    words = text.lower().split()
    if not words:
        return [0.0] * _VECTOR_DIM

    # Accumulate word vectors with position decay (earlier words matter more)
    acc = [0.0] * _VECTOR_DIM
    for i, word in enumerate(words):
        # Strip punctuation
        cleaned = "".join(c for c in word if c.isalnum())
        if not cleaned or len(cleaned) < 2:
            continue
        weight = 1.0 / (1.0 + i * 0.1)  # Gentle position decay
        wv = _word_hash_vector(cleaned)
        for j in range(_VECTOR_DIM):
            acc[j] += wv[j] * weight

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in acc)) or 1.0
    return [x / norm for x in acc]


# ---------------------------------------------------------------------------
# Seed memories -- a mix of types to showcase classification and retrieval
# ---------------------------------------------------------------------------

SEED_MEMORIES: list[dict] = [
    # Rules (auto-classified via content heuristics)
    {
        "content": "You MUST NEVER push directly to main without a pull request. "
        "All changes require code review before merging.",
        "source": "team_rules",
        "tags": ["git", "workflow", "review"],
        "confidence": 0.95,
        "memory_type": "knowledge",
        "wing": "infrastructure",
    },
    {
        "content": "ALWAYS run the test suite before deploying to production. "
        "Skipping tests has caused two outages this quarter.",
        "source": "incident_review",
        "tags": ["testing", "deployment", "resilience"],
        "confidence": 0.90,
        "memory_type": "knowledge",
        "wing": "infrastructure",
    },
    # Facts -- technical knowledge
    {
        "content": "Circuit breakers protect against cascading failures by isolating "
        "failing providers. When error rate exceeds 50% over a 30-second window, "
        "the breaker opens and routes traffic to healthy backends.",
        "source": "documentation",
        "tags": ["routing", "resilience", "circuit-breaker"],
        "confidence": 0.85,
        "memory_type": "knowledge",
        "wing": "routing",
    },
    {
        "content": "The embedding provider chains multiple backends with automatic "
        "fallback: Ollama (local, free) -> DeepInfra (cloud, fast) -> DashScope "
        "(cloud, backup). If all fail, memories store to FTS5 and queue for later.",
        "source": "session_extraction",
        "tags": ["embeddings", "resilience", "fallback"],
        "confidence": 0.80,
        "memory_type": "episodic",
        "wing": "infrastructure",
    },
    {
        "content": "RRF (Reciprocal Rank Fusion) is deliberately score-agnostic. "
        "It only uses rank positions, never absolute scores. This means different "
        "vector backends (ChromaDB distances vs Qdrant similarities) produce "
        "identical fusion results for the same data.",
        "source": "deep_reflection",
        "tags": ["retrieval", "rrf", "algorithm"],
        "confidence": 0.90,
        "memory_type": "knowledge",
        "wing": "memory",
    },
    # Episodic -- session events
    {
        "content": "Deployed v2.3 of the routing subsystem. Key changes: added "
        "provider health scoring, reduced cold-start latency by 40%, and fixed "
        "the edge case where stale circuit breaker state survived restarts.",
        "source": "session_extraction",
        "tags": ["routing", "deployment", "performance"],
        "confidence": 0.75,
        "memory_type": "episodic",
        "wing": "routing",
    },
    {
        "content": "Investigated memory leak in the background session dispatcher. "
        "Root cause: asyncio tasks were not being awaited on shutdown, leaving "
        "dangling references. Fixed by adding a task registry with graceful drain.",
        "source": "session_extraction",
        "tags": ["debugging", "memory-leak", "asyncio"],
        "confidence": 0.85,
        "memory_type": "episodic",
        "wing": "infrastructure",
    },
    # References
    {
        "content": "Qdrant documentation for filtered search is at "
        "https://qdrant.tech/documentation/concepts/filtering/ -- supports "
        "nested conditions, geo filters, and payload indexing.",
        "source": "documentation",
        "tags": ["qdrant", "reference", "vector-search"],
        "confidence": 0.70,
        "memory_type": "knowledge",
        "wing": "infrastructure",
    },
    # Decision records
    {
        "content": "Decided to use SQLite FTS5 instead of Elasticsearch for text "
        "search. Rationale: zero infrastructure, built into Python, good enough "
        "for thousands of memories. Elasticsearch would add operational burden "
        "without proportional benefit at our scale.",
        "source": "deep_reflection",
        "tags": ["decision", "architecture", "fts5"],
        "confidence": 0.90,
        "memory_type": "knowledge",
        "wing": "memory",
    },
    # Observation with entity
    {
        "content": "AgentMail is an email service designed for AI agents. It provides "
        "programmatic inbox management, webhook delivery, and structured parsing "
        "of incoming messages. Useful for agent-to-agent communication.",
        "source": "session_extraction",
        "tags": ["AgentMail", "email", "agent-communication"],
        "confidence": 0.80,
        "memory_type": "knowledge",
        "wing": "channels",
    },
    # Procedure
    {
        "content": "To add a new embedding backend: 1) Create a class implementing "
        "EmbeddingBackend protocol (name property, embed method, is_available method). "
        "2) Add it to the backends list in EmbeddingProvider. 3) The provider will "
        "automatically chain it with fallback behavior.",
        "source": "auto_memory_harvest",
        "tags": ["procedure", "embeddings", "extensibility"],
        "confidence": 0.85,
        "memory_type": "knowledge",
        "wing": "learning",
    },
    # Recent status update
    {
        "content": "Memory system migration to genesis-memory standalone package is "
        "complete. All 12 retrieval pipeline steps preserved. Backend protocols "
        "extracted. ChromaDB default backend working. Test coverage at 87%.",
        "source": "retrospective",
        "tags": ["status", "migration", "genesis-memory"],
        "confidence": 0.90,
        "memory_type": "episodic",
        "wing": "memory",
    },
]

# Memories that get backdated to show activation decay
AGED_MEMORIES: list[dict] = [
    {
        "content": "Considered using Pinecone for vector search but rejected due to "
        "vendor lock-in concerns. The protocol-based architecture allows swapping "
        "backends without changing retrieval logic.",
        "source": "deep_reflection",
        "tags": ["decision", "Pinecone", "architecture"],
        "confidence": 0.85,
        "memory_type": "knowledge",
        "wing": "memory",
        "age_days": 45,
    },
    {
        "content": "First successful end-to-end test of the RRF pipeline with real "
        "Qdrant data. Vector search and FTS5 results fused correctly. Activation "
        "scoring properly boosted recent memories.",
        "source": "session_extraction",
        "tags": ["milestone", "rrf", "testing"],
        "confidence": 0.75,
        "memory_type": "episodic",
        "wing": "memory",
        "age_days": 90,
    },
]

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

console = Console()


def print_banner() -> None:
    banner = Text.from_markup(
        "[bold cyan]genesis-memory[/] [dim]interactive demo[/]\n\n"
        "[bold]12-Step RRF Hybrid Retrieval Pipeline[/]\n"
        "[dim]Vector Search + FTS5 + Activation Scoring + Intent Classification[/]\n"
        "[dim]Fused via Reciprocal Rank Fusion[/]\n\n"
        "Type a query to search memories, or use /help for commands.\n"
        "No API keys needed -- uses deterministic mock embeddings."
    )
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def print_help() -> None:
    table = Table(title="Commands", border_style="dim", show_header=True)
    table.add_column("Command", style="bold green", min_width=24)
    table.add_column("Description")
    table.add_row("<query>", "Search memories via the full RRF pipeline")
    table.add_row("/store <content>", "Store a new memory (auto-classified)")
    table.add_row("/graph <id>", "Traverse knowledge graph from a memory ID")
    table.add_row("/stats", "Show cache stats and system metrics")
    table.add_row("/memories", "List all stored memories with their IDs")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit the demo")
    console.print(table)


def format_results(results: list) -> None:
    """Pretty-print retrieval results with full pipeline details."""
    if not results:
        console.print("[yellow]No results found.[/]")
        return

    # Show intent classification
    if results:
        r0 = results[0]
        intent_style = "bold magenta" if r0.query_intent != "GENERAL" else "dim"
        console.print(
            f"  [{intent_style}]Intent: {r0.query_intent}[/] "
            f"[dim](confidence: {r0.intent_confidence:.0%})[/]"
        )
        console.print()

    for i, r in enumerate(results, 1):
        # Classification badge
        class_colors = {"rule": "red", "fact": "blue", "reference": "yellow"}
        cls_color = class_colors.get(r.memory_class, "white")

        # Score bar (visual indicator)
        max_score = results[0].score if results else 1.0
        bar_width = 20
        filled = int((r.score / max_score) * bar_width) if max_score > 0 else 0
        score_bar = "[green]" + "#" * filled + "[/][dim]" + "." * (bar_width - filled) + "[/]"

        # Memory ID (first 8 chars for readability)
        short_id = r.memory_id[:8]

        # Build the result panel
        header = Text.from_markup(
            f"[bold]#{i}[/]  "
            f"[{cls_color}][{r.memory_class.upper()}][/]  "
            f"[dim]{short_id}...[/]"
        )

        detail_lines = [
            f"  {score_bar}  [bold]RRF Score: {r.score:.4f}[/]",
            "",
            f"  [cyan]Vector Rank:[/]  {r.vector_rank or '-':>4}    "
            f"[cyan]FTS5 Rank:[/]  {r.fts_rank or '-':>4}",
            f"  [cyan]Activation:[/]  {r.activation_score:.4f}    "
            f"[cyan]Source:[/]  {r.source}",
        ]

        # Content (truncated for display)
        content_display = r.content
        if len(content_display) > 200:
            content_display = content_display[:197] + "..."

        console.print(header)
        for line in detail_lines:
            console.print(line)
        console.print()
        console.print(f"  [italic]{escape(content_display)}[/]")
        console.print()


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

async def seed_memories(system, memory_ids: dict[str, str]) -> None:
    """Pre-seed the demo with interesting memories."""
    console.print("[dim]Seeding memories...[/]")

    for mem in SEED_MEMORIES:
        mid = await system.store.store(
            mem["content"],
            mem["source"],
            tags=mem.get("tags"),
            confidence=mem.get("confidence", 0.5),
            memory_type=mem.get("memory_type", "episodic"),
            wing=mem.get("wing"),
        )
        # Track for display
        short = mem["content"][:60].replace("\n", " ")
        memory_ids[mid] = short

    # Store aged memories with backdated timestamps via direct vector payload update
    for mem in AGED_MEMORIES:
        age_days = mem.get("age_days", 0)
        mid = await system.store.store(
            mem["content"],
            mem["source"],
            tags=mem.get("tags"),
            confidence=mem.get("confidence", 0.5),
            memory_type=mem.get("memory_type", "episodic"),
            wing=mem.get("wing"),
        )
        memory_ids[mid] = mem["content"][:60].replace("\n", " ")

        # Backdate the created_at timestamp in vector store to show activation decay
        if age_days > 0:
            old_date = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
            is_knowledge = mem.get("memory_type") == "knowledge"
            collection = "knowledge_base" if is_knowledge else "episodic_memory"
            with contextlib.suppress(Exception):
                await system.vector_backend.update_payload(
                    mid,
                    {"created_at": old_date},
                    collection=collection,
                )

    console.print(f"  [green]Stored {len(memory_ids)} memories[/] "
                  f"({len(SEED_MEMORIES)} current + {len(AGED_MEMORIES)} aged)")


async def handle_store(system, content: str, memory_ids: dict[str, str]) -> None:
    """Handle /store command."""
    if not content.strip():
        console.print("[red]Usage: /store <content>[/]")
        return

    # Auto-detect source from content
    source = "user_input"

    mid = await system.store.store(
        content.strip(),
        source,
        confidence=0.80,
    )
    memory_ids[mid] = content.strip()[:60]

    # Show what happened
    from genesis_memory.classification import classify_memory

    mem_class = classify_memory(content.strip())

    table = Table(border_style="green", show_header=False, padding=(0, 1))
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Memory ID", mid)
    table.add_row("Classification", f"[bold]{mem_class}[/]")
    table.add_row("Stored to", "Vector + FTS5")

    # Check if it got auto-linked
    link_count = await system.store.linker.count_links(mid) if system.store.linker else 0
    if link_count > 0:
        table.add_row("Auto-linked to", f"{link_count} similar memories")

    console.print(Panel(table, title="[green]Memory Stored[/]", border_style="green"))


async def handle_graph(system, memory_id_input: str, memory_ids: dict[str, str]) -> None:
    """Handle /graph command."""
    if not memory_id_input.strip():
        console.print("[red]Usage: /graph <memory_id>[/]")
        console.print("[dim]Tip: use /memories to see available IDs[/]")
        return

    from genesis_memory.graph import get_cluster, traverse

    query_id = memory_id_input.strip()

    # Allow prefix matching for convenience
    matched_id = None
    for mid in memory_ids:
        if mid.startswith(query_id):
            matched_id = mid
            break

    if not matched_id:
        console.print(f"[red]Memory ID '{escape(query_id)}' not found.[/]")
        console.print("[dim]Tip: use /memories to see available IDs[/]")
        return

    # Traverse outgoing links
    result = await traverse(system.db, matched_id, max_depth=3, min_strength=0.0)

    # Get bidirectional cluster
    cluster = await get_cluster(system.db, matched_id, max_depth=2, min_strength=0.0)

    # Display
    short_content = memory_ids.get(matched_id, "???")
    console.print(
        Panel(
            f"[bold]{escape(short_content)}[/]\n"
            f"[dim]ID: {matched_id}[/]",
            title="[cyan]Graph Root[/]",
            border_style="cyan",
        )
    )

    if result.nodes:
        table = Table(
            title=f"Outgoing Links ({len(result.nodes)} nodes, {result.query_ms:.1f}ms)",
            border_style="cyan",
        )
        table.add_column("Depth", justify="center", style="bold")
        table.add_column("Link Type", style="magenta")
        table.add_column("Strength", justify="right")
        table.add_column("Target ID", style="dim")
        table.add_column("Content Preview")

        for node in result.nodes:
            preview = memory_ids.get(node.memory_id, "[dim]unknown[/]")
            strength_color = (
                "green" if node.strength >= 0.8
                else "yellow" if node.strength >= 0.5
                else "red"
            )
            table.add_row(
                str(node.depth),
                node.link_type,
                f"[{strength_color}]{node.strength:.2f}[/]",
                node.memory_id[:8] + "...",
                escape(str(preview)[:50]),
            )
        console.print(table)
    else:
        console.print("[dim]No outgoing links from this memory.[/]")

    if cluster:
        console.print(
            f"\n[cyan]Cluster size:[/] {len(cluster) + 1} memories "
            f"(root + {len(cluster)} connected)"
        )
    else:
        console.print("\n[dim]This memory is not connected to any cluster.[/]")


def handle_stats(system) -> None:
    """Handle /stats command."""
    cache = system.retriever._embeddings.cache_stats()

    table = Table(title="Embedding Cache Statistics", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    total_requests = cache["l1_hits"] + cache["l2_hits"] + cache["misses"]
    hit_rate = (
        (cache["l1_hits"] + cache["l2_hits"]) / total_requests * 100
        if total_requests > 0
        else 0
    )

    table.add_row("L1 Cache (in-memory)", f"{cache['l1_size']} entries")
    table.add_row("L2 Cache (disk)", f"{cache['l2_size']} entries")
    table.add_row("L1 Hits", str(cache["l1_hits"]))
    table.add_row("L2 Hits", str(cache["l2_hits"]))
    table.add_row("Cache Misses", str(cache["misses"]))
    table.add_row("Total Requests", str(total_requests))
    table.add_row("Hit Rate", f"{hit_rate:.1f}%")
    table.add_row("Remote Calls", str(cache["remote_calls"]))
    console.print(table)

    # Pipeline info
    info = Table(title="Pipeline Configuration", border_style="dim")
    info.add_column("Component", style="bold")
    info.add_column("Value")
    info.add_row("Vector Backend", "ChromaDB (ephemeral)")
    info.add_row("Text Backend", "SQLite FTS5")
    info.add_row("Embedding Backend", "Mock Hasher (deterministic)")
    info.add_row("Vector Dimensions", str(_VECTOR_DIM))
    info.add_row("RRF k", "60")
    info.add_row("Fusion Lists", "Vector + FTS5 + Activation + Intent")
    info.add_row("Auto-linking", "Enabled (similarity threshold: 0.75)")
    console.print(info)


def handle_memories(memory_ids: dict[str, str]) -> None:
    """Handle /memories command."""
    if not memory_ids:
        console.print("[yellow]No memories stored yet.[/]")
        return

    table = Table(title=f"Stored Memories ({len(memory_ids)})", border_style="dim")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("ID (prefix)", style="cyan", min_width=10)
    table.add_column("Content Preview")

    for i, (mid, preview) in enumerate(memory_ids.items(), 1):
        table.add_row(str(i), mid[:12] + "...", escape(preview))

    console.print(table)
    console.print("[dim]Use the ID prefix with /graph to explore connections.[/]")


async def main() -> None:
    from genesis_memory import create_memory_system

    print_banner()
    console.print("[dim]Initializing memory system...[/]")

    # Create the mock embedding backend
    mock_backend = MockEmbeddingBackend()

    # Create the memory system with mock embeddings -- no API keys needed
    system = await create_memory_system(
        db_path="demo_memory.db",
        vector_backend="chromadb",  # Ephemeral (no path = in-memory)
        embedding_backends=[mock_backend],
        vector_dim=_VECTOR_DIM,
        collections=["episodic_memory", "knowledge_base"],
        embedding_cache_dir=None,  # Skip disk cache for demo
    )

    console.print("[green]Memory system ready.[/]\n")

    # Track memory IDs for /memories and /graph commands
    memory_ids: dict[str, str] = {}

    # Seed with demo memories
    await seed_memories(system, memory_ids)
    console.print()

    # Interactive loop
    try:
        while True:
            try:
                query = console.input("[bold green]>[/] ").strip()
            except EOFError:
                break

            if not query:
                continue

            # Commands
            if query.lower() in ("/quit", "/exit", "/q"):
                break
            elif query.lower() == "/help":
                print_help()
                continue
            elif query.lower() == "/stats":
                handle_stats(system)
                continue
            elif query.lower() == "/memories":
                handle_memories(memory_ids)
                continue
            elif query.lower().startswith("/store "):
                content = query[7:]
                await handle_store(system, content, memory_ids)
                continue
            elif query.lower().startswith("/graph "):
                memory_id_input = query[7:]
                await handle_graph(system, memory_id_input, memory_ids)
                continue
            elif query.startswith("/"):
                console.print(f"[red]Unknown command: {escape(query.split()[0])}[/]")
                console.print("[dim]Type /help for available commands.[/]")
                continue

            # Retrieval query
            console.print()
            console.print(
                f"[bold cyan]Searching:[/] [italic]{escape(query)}[/]"
            )
            console.print("[dim]Running 12-step pipeline: "
                          "Embed -> Vector -> Intent -> Expand -> FTS5 -> "
                          "Union -> Activation -> Rank -> RRF -> Filter -> "
                          "Sort -> Results[/]")
            console.print()

            results = await system.retriever.recall(
                query,
                source="both",
                limit=5,
            )

            format_results(results)

            # Show fusion summary
            if results:
                vector_only = sum(1 for r in results if r.vector_rank and not r.fts_rank)
                fts_only = sum(1 for r in results if r.fts_rank and not r.vector_rank)
                both = sum(1 for r in results if r.vector_rank and r.fts_rank)
                console.print(
                    f"[dim]Fusion: {both} matched both, "
                    f"{vector_only} vector-only, {fts_only} FTS5-only[/]"
                )
            console.print()

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
    finally:
        await system.close()
        console.print("[dim]Goodbye.[/]")

        # Clean up demo database
        import os
        with contextlib.suppress(FileNotFoundError):
            os.unlink("demo_memory.db")


if __name__ == "__main__":
    asyncio.run(main())
