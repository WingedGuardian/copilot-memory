"""Alibaba DashScope cloud embedding backend (OpenAI-compatible API)."""

from __future__ import annotations

import httpx


class DashScopeBackend:
    """DashScope cloud embedding backend.

    Uses text-embedding-v4 with explicit dimensions for vector space
    compatibility.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client = client or httpx.AsyncClient(timeout=30.0)

    @property
    def name(self) -> str:
        return "dashscope_embedding"

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": [text],
                "dimensions": self._dimensions,
            },
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def is_available(self) -> bool:
        return True
