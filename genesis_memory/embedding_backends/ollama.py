"""Ollama embedding backend (local, qwen3-embedding)."""

from __future__ import annotations

import time

import httpx

_HTTPX_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.HTTPStatusError,
)


class OllamaBackend:
    """Local Ollama embedding backend.

    Uses a 60s timeout (Ollama can be slow under GPU contention or cold model
    loading) and retries once on ReadTimeout before propagating the failure.
    """

    _AVAIL_TTL = 120.0

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:0.6b-fp16",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=60.0)
        self._avail_cache: bool | None = None
        self._avail_cache_at: float = 0.0

    @property
    def name(self) -> str:
        return "ollama_embedding"

    async def embed(self, text: str) -> list[float]:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = await self._client.post(
                    f"{self._url.rstrip('/')}/api/embed",
                    json={"model": self._model, "input": text, "keep_alive": -1},
                )
                resp.raise_for_status()
                return resp.json()["embeddings"][0]
            except httpx.ReadTimeout as exc:
                last_exc = exc
                if attempt == 0:
                    import asyncio
                    await asyncio.sleep(1.0)
                    continue
                raise
            except Exception:
                raise
        raise last_exc  # type: ignore[misc]

    async def is_available(self) -> bool:
        now = time.monotonic()
        if (
            self._avail_cache is not None
            and (now - self._avail_cache_at) < self._AVAIL_TTL
        ):
            return self._avail_cache
        try:
            resp = await self._client.get(
                f"{self._url.rstrip('/')}/api/tags", timeout=5.0,
            )
            result = resp.status_code == 200
        except _HTTPX_ERRORS:
            result = False
        self._avail_cache = result
        self._avail_cache_at = now
        return result
