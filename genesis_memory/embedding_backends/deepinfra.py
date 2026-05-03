"""DeepInfra cloud embedding backend (OpenAI-compatible API)."""

from __future__ import annotations

import httpx


class DeepInfraBackend:
    """DeepInfra cloud embedding backend."""

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=30.0)

    @property
    def name(self) -> str:
        return "deepinfra_embedding"

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            "https://api.deepinfra.com/v1/openai/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": [text]},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def is_available(self) -> bool:
        return True
