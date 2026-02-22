"""Embedding client with local-first + cloud fallback."""

from __future__ import annotations

from loguru import logger


class Embedder:
    """Generates text embeddings via local LM Studio, falling back to cloud.

    Tries local endpoint first (fast, free). If that fails, falls back to a
    cloud OpenAI-compatible endpoint (e.g. OpenAI text-embedding-3-small via
    OpenRouter or direct). Zero-vector returned only if both fail.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:1234/v1",
        model: str = "text-embedding-nomic-embed-text-v1.5",
        dimensions: int = 768,
        cloud_api_key: str | None = None,
        cloud_api_base: str | None = None,
        cloud_model: str = "text-embedding-3-small",
        cloud_dimensions: int | None = None,
        secrets: "SecretsProvider | None" = None,
    ):
        self._api_base = api_base
        self._model = model
        self._dimensions = dimensions

        # Cloud fallback config — prefer SecretsProvider if available
        if not cloud_api_key and secrets:
            cloud_api_key = secrets.get("OPENAI_API_KEY")
        self._cloud_api_key = cloud_api_key
        self._cloud_api_base = cloud_api_base  # None = default OpenAI endpoint
        self._cloud_model = cloud_model
        self._cloud_dimensions = cloud_dimensions or dimensions

        self._local_client = None
        self._cloud_client = None
        self._local_available = True  # Optimistic; flips on failure
        self._last_zero = False  # True when last embed() returned zero-vector

    def _ensure_local(self):
        if self._local_client is None:
            from openai import AsyncOpenAI
            self._local_client = AsyncOpenAI(
                api_key="lm-studio",
                base_url=self._api_base,
            )

    def _ensure_cloud(self):
        if self._cloud_client is None and self._cloud_api_key:
            from openai import AsyncOpenAI
            kwargs = {"api_key": self._cloud_api_key}
            if self._cloud_api_base:
                kwargs["base_url"] = self._cloud_api_base
            self._cloud_client = AsyncOpenAI(**kwargs)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text. Local → cloud → zero-vector.

        Zero-vector only on double failure (both local and cloud down).
        Caller can check ``_last_zero`` to decide whether to enqueue for retry.
        """
        text = text[:8000]
        self._last_zero = False

        # 1. Try local (free, fast, private)
        if self._local_available:
            try:
                self._ensure_local()
                response = await self._local_client.embeddings.create(
                    input=text, model=self._model,
                )
                self._local_available = True
                return self._pad_or_trim(response.data[0].embedding)
            except Exception as e:
                logger.info(f"Local embedding unavailable: {e}")
                self._local_available = False

        # 2. Try cloud (Jina/OpenAI — cheap, reliable)
        cloud_vec = await self._embed_cloud(text)
        if cloud_vec is not None:
            return cloud_vec

        # 3. Both down — zero-vector (enqueued for retry)
        logger.warning("Both local and cloud embedding failed — zero-vector stored")
        self._last_zero = True
        return [0.0] * self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding. Local → cloud per-item → zero-vectors."""
        truncated = [t[:8000] for t in texts]
        self._last_zero = False

        # Try local batch
        if self._local_available:
            try:
                self._ensure_local()
                response = await self._local_client.embeddings.create(
                    input=truncated, model=self._model,
                )
                self._local_available = True
                return [self._pad_or_trim(d.embedding) for d in response.data]
            except Exception as e:
                logger.info(f"Local batch embedding unavailable: {e}")
                self._local_available = False

        # Fall back to cloud per-item
        results = []
        any_zero = False
        for t in truncated:
            cloud_vec = await self._embed_cloud(t)
            if cloud_vec is not None:
                results.append(cloud_vec)
            else:
                results.append([0.0] * self._dimensions)
                any_zero = True
        self._last_zero = any_zero
        return results

    async def embed_cloud(self, text: str) -> list[float]:
        """Embed via cloud (public API for drainer/callers that need cloud explicitly)."""
        result = await self._embed_cloud(text)
        if result is not None:
            return result
        raise RuntimeError("Cloud embedding failed")

    async def _embed_cloud(self, text: str) -> list[float] | None:
        """Embed via cloud endpoint. Returns None on failure."""
        if not self._cloud_api_key:
            return None
        try:
            self._ensure_cloud()
            kwargs = {"input": text, "model": self._cloud_model}
            # OpenAI supports dimensions param for ada-3/text-embedding-3
            if self._cloud_dimensions and "embedding-3" in self._cloud_model:
                kwargs["dimensions"] = self._cloud_dimensions
            response = await self._cloud_client.embeddings.create(**kwargs)
            vec = response.data[0].embedding
            return self._pad_or_trim(vec)
        except Exception as e:
            logger.warning(f"Cloud embedding failed: {e}")
            return None

    def _pad_or_trim(self, vec: list[float]) -> list[float]:
        """Ensure vector matches expected dimensions."""
        if len(vec) == self._dimensions:
            return vec
        if len(vec) > self._dimensions:
            return vec[:self._dimensions]
        return vec + [0.0] * (self._dimensions - len(vec))
