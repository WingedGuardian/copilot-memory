"""copilot-memory: Hybrid AI memory with Qdrant vectors + SQLite FTS5."""

from copilot_memory.embedder import Embedder
from copilot_memory.episodic import Episode, EpisodicStore
from copilot_memory.fulltext import FTSResult, FullTextStore
from copilot_memory.manager import MemoryManager

__all__ = [
    "Embedder",
    "Episode",
    "EpisodicStore",
    "FTSResult",
    "FullTextStore",
    "MemoryManager",
]
