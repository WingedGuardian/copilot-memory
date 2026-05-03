"""Embedding provider with configurable backend chains and two-level cache.

Two chain configurations for split read/write paths:
  Storage (writes): local → cloud (cost-optimized)
  Recall (reads):   cloud → local (latency-optimized)

Two-level cache: L1 in-process dict (fast, per-process) backed by
L2 diskcache on disk (shared across processes).
Embeddings are deterministic for a given model+text, so long TTLs are safe.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".genesis_memory" / "embedding_cache"

_HTTPX_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.HTTPStatusError,
)


class EmbeddingUnavailableError(Exception):
    """Raised when all embedding backends are unavailable."""


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends in the provider chain."""

    @property
    def name(self) -> str: ...
    async def embed(self, text: str) -> list[float]: ...
    async def is_available(self) -> bool: ...


class EmbeddingProvider:
    """Embedding provider with backend chain and two-level cache.

    Backend chain order is user-configurable. If all backends fail,
    raises EmbeddingUnavailableError. Caller (MemoryStore) falls to
    FTS5-only and queues for later embedding.
    """

    def __init__(
        self,
        *,
        backends: list[EmbeddingBackend],
        cache_dir: Path | None = _DEFAULT_CACHE_DIR,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self._backends = backends
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._cache_ttl: float = 86400.0  # 24 hours
        self._cache_max: int = 2048
        self._on_event = on_event

        # Observability counters
        self._l1_hits: int = 0
        self._l2_hits: int = 0
        self._misses: int = 0
        self._remote_calls: int = 0
        self._consecutive_backend_failures: dict[str, int] = {}

        # L2 shared disk cache
        self._disk_cache = None
        if cache_dir is not None:
            try:
                import diskcache
                cache_dir.mkdir(parents=True, exist_ok=True)
                self._disk_cache = diskcache.Cache(
                    str(cache_dir), size_limit=100_000_000,  # 100 MB
                )
            except Exception:
                logger.warning(
                    "Failed to initialize diskcache at %s, using L1-only",
                    cache_dir, exc_info=True,
                )

        backend_names = [b.name for b in self._backends]
        self._cache_prefix = "+".join(backend_names) if backend_names else "none"
        logger.info("Embedding provider initialized: chain=%s", backend_names)

    # -- Cache layer --

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self._cache_prefix}:{text}".encode()).hexdigest()

    def _cache_get(self, text: str) -> list[float] | None:
        key = self._cache_key(text)

        # L1: in-process dict
        entry = self._cache.get(key)
        if entry is not None:
            vec, ts = entry
            if time.monotonic() - ts <= self._cache_ttl:
                self._l1_hits += 1
                return vec
            del self._cache[key]

        # L2: shared diskcache
        if self._disk_cache is not None:
            try:
                vec = self._disk_cache.get(key)
                if vec is not None:
                    self._l2_hits += 1
                    self._l1_put(key, vec)
                    return vec
            except Exception:
                logger.debug("diskcache get failed for key %s", key[:12], exc_info=True)

        self._misses += 1
        return None

    def _l1_put(self, key: str, vec: list[float]) -> None:
        if len(self._cache) >= self._cache_max:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (vec, time.monotonic())

    def _cache_put(self, text: str, vec: list[float]) -> None:
        key = self._cache_key(text)
        self._l1_put(key, vec)
        if self._disk_cache is not None:
            try:
                self._disk_cache.set(key, vec, expire=604800)
            except Exception:
                logger.debug("diskcache set failed for key %s", key[:12], exc_info=True)

    def cache_stats(self) -> dict:
        return {
            "l1_size": len(self._cache),
            "l2_size": len(self._disk_cache) if self._disk_cache is not None else 0,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "remote_calls": self._remote_calls,
        }

    # -- Public API --

    async def is_available(self) -> bool:
        """Check if at least one embedding backend is reachable."""
        for backend in self._backends:
            try:
                if await backend.is_available():
                    return True
            except Exception:
                continue
        return False

    async def embed(self, text: str) -> list[float]:
        """Embed single text, returns vector."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        vec = await self._embed_remote(text)
        self._cache_put(text, vec)
        return vec

    async def _embed_remote(self, text: str) -> list[float]:
        """Try each backend in chain order. First success wins."""
        self._remote_calls += 1
        if self._remote_calls % 100 == 0:
            stats = self.cache_stats()
            logger.debug(
                "Embedding cache: L1=%d/%d L2=%d hits=%d+%d misses=%d remote=%d",
                stats["l1_size"], self._cache_max, stats["l2_size"],
                stats["l1_hits"], stats["l2_hits"], stats["misses"],
                stats["remote_calls"],
            )

        errors: list[tuple[str, Exception]] = []

        for backend in self._backends:
            t0 = time.monotonic()
            try:
                vec = await backend.embed(text)
                latency = (time.monotonic() - t0) * 1000
                # Reset failure counter on success
                self._consecutive_backend_failures[backend.name] = 0
                if errors:
                    failed_names = [name for name, _ in errors]
                    details = "; ".join(
                        f"{n}: {type(e).__name__}" + (f" ({e})" if str(e) else "")
                        for n, e in errors
                    )
                    self._emit(
                        "embedding.fallback",
                        f"{'→'.join(failed_names)}→{backend.name} fallback ({details})",
                    )
                logger.debug(
                    "Embedding via %s: %.0fms", backend.name, latency,
                )
                return vec
            except Exception as exc:
                fails = self._consecutive_backend_failures.get(backend.name, 0) + 1
                self._consecutive_backend_failures[backend.name] = fails
                exc_desc = (
                    f"{type(exc).__name__}" + (f": {exc}" if str(exc) else " (no details)")
                )
                if fails <= 3:
                    logger.warning(
                        "Embedding backend '%s' failed (%d consecutive): %s",
                        backend.name, fails, exc_desc, exc_info=True,
                    )
                elif fails % 50 == 0:
                    logger.warning(
                        "Embedding backend '%s' still failing (%d consecutive): %s",
                        backend.name, fails, exc_desc,
                    )
                else:
                    logger.debug(
                        "Embedding backend '%s' failed (%d consecutive): %s",
                        backend.name, fails, exc_desc,
                    )
                errors.append((backend.name, exc))

        # All backends failed
        failed_names = [name for name, _ in errors]
        self._emit(
            "embedding.failed",
            f"All embedding backends failed: {', '.join(failed_names)}",
        )
        msg = f"All embedding backends failed: {failed_names}"
        raise EmbeddingUnavailableError(msg)

    def _emit(self, event_type: str, message: str) -> None:
        if self._on_event:
            try:
                self._on_event(event_type, message)
            except Exception:
                logger.debug("Failed to emit event", exc_info=True)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [await self.embed(t) for t in texts]

    @staticmethod
    def enrich(content: str, memory_type: str, tags: list[str]) -> str:
        """Contextual enrichment: prepend type and tags before embedding."""
        if tags:
            return f"{memory_type}: {' '.join(tags)}: {content}"
        return f"{memory_type}: {content}"
