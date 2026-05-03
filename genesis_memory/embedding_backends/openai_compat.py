"""Generic OpenAI-compatible embedding backend.

Works with any provider that implements the OpenAI /v1/embeddings endpoint
(e.g., Together, Fireworks, local vLLM, etc.).
"""

from __future__ import annotations

import httpx


class OpenAICompatBackend:
    """Generic OpenAI-compatible embedding backend."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int | None = None,
        name: str = "openai_compat_embedding",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._name = name
        self._client = client or httpx.AsyncClient(timeout=30.0)

    @property
    def name(self) -> str:
        return self._name

    async def embed(self, text: str) -> list[float]:
        body: dict = {"model": self._model, "input": [text]}
        if self._dimensions is not None:
            body["dimensions"] = self._dimensions
        resp = await self._client.post(
            f"{self._base_url}/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def is_available(self) -> bool:
        return True
