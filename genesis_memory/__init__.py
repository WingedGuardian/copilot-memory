"""genesis-memory — Production-grade memory system with RRF hybrid retrieval."""

from genesis_memory.activation import compute_activation
from genesis_memory.classification import CLASS_WEIGHTS, classify_memory
from genesis_memory.embeddings import EmbeddingProvider
from genesis_memory.factory import MemorySystem, create_memory_system
from genesis_memory.protocols import (
    LinkBackend,
    MetadataBackend,
    PendingBackend,
    TextBackend,
    VectorBackend,
)
from genesis_memory.retrieval import HybridRetriever
from genesis_memory.store import MemoryStore
from genesis_memory.types import (
    ActivationScore,
    LinkRecord,
    MemoryRecord,
    RetrievalResult,
)

__all__ = [
    # Core classes
    "MemoryStore",
    "HybridRetriever",
    "EmbeddingProvider",
    "MemorySystem",
    # Factory
    "create_memory_system",
    # Types
    "ActivationScore",
    "LinkRecord",
    "MemoryRecord",
    "RetrievalResult",
    # Protocols
    "LinkBackend",
    "MetadataBackend",
    "PendingBackend",
    "TextBackend",
    "VectorBackend",
    # Functions
    "classify_memory",
    "compute_activation",
    "CLASS_WEIGHTS",
]
