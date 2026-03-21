"""RouteLLM Matrix Factorization Router - numpy-only inference.

Supports two embedding backends (auto-detected from checkpoint shape):
  - OpenAI: text-embedding-3-small (1536-dim)
  - Gemini: gemini-embedding-001 (3072-dim)

Based on: https://github.com/lm-sys/RouteLLM
Paper: "RouteLLM: Learning to Route LLMs with Preference Data"

Architecture:
  1. Embed prompt via embedding API (1536 or 3072-dim)
  2. Project to hidden space: text_proj @ embedding -> (128,)
  3. Get strong/weak model embeddings from P, L2-normalize
  4. Element-wise multiply: embed * prompt_proj
  5. Classify: classifier @ product -> logit
  6. Sigmoid(logit_strong - logit_weak) -> win_rate
  7. win_rate > threshold -> route to cloud (strong)

Usage::

    router = MFRouter(
        checkpoint_path="models/mf-router/model_gemini.safetensors",
        gemini_api_key="AIzaSy...",
        threshold=0.5,
    )
    use_cloud, win_rate = await router.should_route_to_cloud("Explain quicksort")
    # use_cloud=False, win_rate=0.32  -> local can handle this

    use_cloud, win_rate = await router.should_route_to_cloud(
        "Design a distributed consensus algorithm with Byzantine fault tolerance"
    )
    # use_cloud=True, win_rate=0.87  -> route to cloud
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("omlx.cloud.mf_router")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 128
_CACHE_KEY_PREFIX_LEN = 512  # First N chars used as cache key
_LOGIT_CLIP = 20.0  # Clip sigmoid input to prevent overflow
_API_TIMEOUT_SECONDS = 5

# OpenAI backend
_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_OPENAI_EMBEDDING_DIM = 1536
_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
_OPENAI_MAX_INPUT_CHARS = 8192

# Gemini backend
_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
_GEMINI_EMBEDDING_DIM = 3072
_GEMINI_EMBEDDINGS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-embedding-001:embedContent"
)
_GEMINI_MAX_INPUT_CHARS = 10000


# ---------------------------------------------------------------------------
# MFRouter
# ---------------------------------------------------------------------------

class MFRouter:
    """Matrix Factorization Router for strong/weak model routing.

    Uses pre-trained weights from RouteLLM's Chatbot Arena preference data
    to predict whether a prompt benefits from a strong (cloud) or weak (local)
    model.  The forward pass is pure numpy (<0.1ms); the dominant cost is
    the OpenAI embedding API call (~100-300ms).

    Attributes:
        MODEL_IDS: Mapping of arena model names to embedding matrix indices.
        DEFAULT_STRONG_ID: GPT-4-1106-preview (ID 21) - proxy for cloud premium.
        DEFAULT_WEAK_ID: Mixtral-8x7B (ID 36) - proxy for local 35B model.
    """

    # Full RouteLLM MODEL_IDS registry (from the original source)
    MODEL_IDS: Dict[str, int] = {
        "gpt-4-1106-preview": 21,
        "gpt-4-0613": 20,
        "gpt-4-0314": 19,
        "gpt-4-0125-preview": 18,
        "gpt-4-turbo-2024-04-09": 22,
        "gpt-3.5-turbo-0613": 15,
        "gpt-3.5-turbo-0125": 13,
        "claude-3-opus-20240229": 2,
        "claude-3-sonnet-20240229": 3,
        "claude-3-haiku-20240307": 1,
        "claude-2.1": 0,
        "claude-2.0": 4,
        "command-r-plus": 6,
        "command-r": 5,
        "dbrx-instruct": 7,
        "gemini-pro": 8,
        "gemini-pro-dev-api": 9,
        "gemma-1.1-7b-it": 10,
        "gemma-7b-it": 11,
        "llama-2-13b-chat": 25,
        "llama-2-70b-chat": 26,
        "llama-2-7b-chat": 27,
        "mixtral-8x7b-instruct-v0.1": 36,
        "mistral-7b-instruct-v0.2": 35,
        "mistral-7b-instruct": 34,
        "mistral-large-2402": 37,
        "mistral-medium": 38,
        "phi-3-mini-4k-instruct": 42,
        "phi-3-small-8k-instruct": 43,
        "qwen1.5-72b-chat": 46,
        "qwen1.5-7b-chat": 48,
        "qwen1.5-110b-chat": 44,
        "yi-34b-chat": 61,
        "vicuna-13b": 55,
        "vicuna-33b": 56,
        "vicuna-7b": 57,
        "zephyr-7b-beta": 63,
        "codellama-70b-instruct": 24,
        "deepseek-llm-67b-chat": 23,
        "starling-lm-7b-beta": 52,
        "tulu-2-dpo-70b": 53,
        "openchat-3.5-0106": 40,
        "wizardlm-70b": 60,
        "wizardlm-13b": 59,
        "chatglm3-6b": 12,
        "llama-3-70b-instruct": 30,
        "llama-3-8b-instruct": 31,
        "gemma-1.1-2b-it": 32,
        "olmo-7b-instruct": 41,
        "pplx-70b-online": 39,
        "gpt-4o-2024-05-13": 16,
        "gemini-1.5-pro-api-0409-preview": 17,
        "phi-3-medium-4k-instruct": 33,
        "snowflake-arctic-instruct": 51,
        "reka-flash-21b-20240226-online": 49,
        "qwen1.5-14b-chat": 45,
        "qwen1.5-32b-chat": 47,
        "mistral-7b-instruct-v0.1": 29,
        "snorkel-mistral-pairrm-dpo": 50,
        "llama-2-chat": 28,
        "yi-1.5-34b-chat": 62,
        "llama-3-70b-instruct-nitro": 54,
        "reka-flash": 58,
    }

    DEFAULT_STRONG_ID: int = 21  # gpt-4-1106-preview
    DEFAULT_WEAK_ID: int = 36    # mixtral-8x7b-instruct-v0.1

    def __init__(
        self,
        checkpoint_path: str,
        openai_api_key: str = "",
        gemini_api_key: str = "",
        threshold: float = 0.5,
        strong_model_id: int = DEFAULT_STRONG_ID,
        weak_model_id: int = DEFAULT_WEAK_ID,
        embedding_cache_size: int = 256,
    ) -> None:
        """Initialize the MFRouter with pre-trained weights.

        The embedding backend (OpenAI or Gemini) is auto-detected from the
        text_proj weight shape: (128, 1536) = OpenAI, (128, 3072) = Gemini.

        Args:
            checkpoint_path: Path to the safetensors checkpoint file.
            openai_api_key: OpenAI API key (for 1536-dim checkpoint).
            gemini_api_key: Gemini API key (for 3072-dim checkpoint).
            threshold: Win-rate threshold; above routes to cloud. Range [0, 1].
            strong_model_id: Index into P.weight for the strong model proxy.
            weak_model_id: Index into P.weight for the weak model proxy.
            embedding_cache_size: Max number of cached prompt embeddings.

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
            KeyError: If expected tensor names are missing from checkpoint.
            ValueError: If tensor shapes are incompatible.
        """
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"MF Router checkpoint not found: {checkpoint_path}"
            )

        # Load weights with safetensors (numpy-only, no PyTorch dependency)
        from safetensors.numpy import load_file

        tensors = load_file(str(checkpoint))

        # Validate expected tensors
        required_keys = {"P.weight", "text_proj.0.weight", "classifier.0.weight"}
        missing = required_keys - set(tensors.keys())
        if missing:
            raise KeyError(
                f"Missing tensors in checkpoint: {missing}. "
                f"Found: {list(tensors.keys())}"
            )

        self._P: np.ndarray = tensors["P.weight"]                     # (64, 128)
        self._text_proj: np.ndarray = tensors["text_proj.0.weight"]    # (128, D)
        self._classifier: np.ndarray = tensors["classifier.0.weight"]  # (1, 128)

        # Validate shapes
        if self._P.ndim != 2 or self._P.shape[1] != _HIDDEN_DIM:
            raise ValueError(
                f"P.weight shape {self._P.shape} incompatible, "
                f"expected (N, {_HIDDEN_DIM})"
            )
        if self._classifier.shape != (1, _HIDDEN_DIM):
            raise ValueError(
                f"classifier.0.weight shape {self._classifier.shape} incompatible, "
                f"expected (1, {_HIDDEN_DIM})"
            )

        # Auto-detect embedding backend from text_proj shape
        embed_dim = self._text_proj.shape[1]
        if self._text_proj.shape[0] != _HIDDEN_DIM:
            raise ValueError(
                f"text_proj.0.weight shape {self._text_proj.shape} incompatible, "
                f"expected ({_HIDDEN_DIM}, embed_dim)"
            )

        if embed_dim == _OPENAI_EMBEDDING_DIM:
            self._backend = "openai"
            self._embedding_dim = _OPENAI_EMBEDDING_DIM
        elif embed_dim == _GEMINI_EMBEDDING_DIM:
            self._backend = "gemini"
            self._embedding_dim = _GEMINI_EMBEDDING_DIM
        else:
            raise ValueError(
                f"text_proj embed_dim={embed_dim} does not match "
                f"OpenAI ({_OPENAI_EMBEDDING_DIM}) or Gemini ({_GEMINI_EMBEDDING_DIM})"
            )

        # Validate model IDs are within P matrix bounds
        num_models = self._P.shape[0]
        if strong_model_id >= num_models:
            raise ValueError(
                f"strong_model_id={strong_model_id} out of bounds "
                f"(P has {num_models} models)"
            )
        if weak_model_id >= num_models:
            raise ValueError(
                f"weak_model_id={weak_model_id} out of bounds "
                f"(P has {num_models} models)"
            )

        self._openai_api_key: str = openai_api_key
        self._gemini_api_key: str = gemini_api_key
        self._threshold: float = max(0.0, min(1.0, threshold))
        self._strong_id: int = strong_model_id
        self._weak_id: int = weak_model_id

        # Pre-compute and L2-normalize model embeddings for fast inference
        self._embed_strong: np.ndarray = self._l2_normalize(
            self._P[self._strong_id].copy()
        )
        self._embed_weak: np.ndarray = self._l2_normalize(
            self._P[self._weak_id].copy()
        )

        # LRU embedding cache to avoid redundant API calls
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_max: int = max(1, embedding_cache_size)

        # Statistics
        self._total_predictions: int = 0
        self._cloud_predictions: int = 0
        self._avg_latency_ms: float = 0.0
        self._api_errors: int = 0

        # Router is available when matching API key is present
        if self._backend == "gemini":
            self._available = bool(gemini_api_key)
        else:
            self._available = bool(openai_api_key)

        logger.info(
            "[__init__] MFRouter ready | checkpoint=%s | backend=%s | "
            "embed_dim=%d | threshold=%.2f | strong_id=%d | weak_id=%d | "
            "P_shape=%s | api_key=%s",
            checkpoint_path,
            self._backend,
            self._embedding_dim,
            self._threshold,
            strong_model_id,
            weak_model_id,
            self._P.shape,
            "set" if self._available else "NOT SET (disabled)",
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def default_checkpoint_path(cls) -> str:
        """Return the default checkpoint path if found, else empty string.

        Prefers Gemini checkpoint (model_gemini.safetensors) over OpenAI
        checkpoint (model.safetensors).  Searches common locations relative
        to the package and the user's home directory.

        Returns:
            Absolute path string to the best checkpoint, or "" if not found.
        """
        base_dirs = [
            Path(__file__).parent.parent.parent.parent / "models" / "mf-router",
            Path.home() / "ThunderOMLX" / "models" / "mf-router",
        ]
        # Prefer Gemini-trained checkpoint, then fall back to OpenAI
        filenames = ["model_gemini.safetensors", "model.safetensors"]
        for base in base_dirs:
            for fname in filenames:
                candidate = base / fname
                if candidate.exists():
                    return str(candidate.resolve())
        return ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the MF Router is available (has matching API key)."""
        return self._available

    @property
    def backend(self) -> str:
        """Embedding backend: 'openai' or 'gemini'."""
        return self._backend

    @property
    def threshold(self) -> float:
        """Current routing threshold."""
        return self._threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_threshold(self, value: float) -> None:
        """Update the routing threshold.

        Higher threshold = more requests stay local.
        Lower threshold = more requests route to cloud.

        Args:
            value: New threshold, clamped to [0.0, 1.0].
        """
        self._threshold = max(0.0, min(1.0, value))
        logger.info("[set_threshold] threshold updated to %.3f", self._threshold)

    async def predict(self, prompt: str) -> float:
        """Predict win rate of the strong model over the weak model.

        The win rate indicates how much a prompt would benefit from using
        a strong (cloud) model versus a weak (local) model.

        Args:
            prompt: User message text.

        Returns:
            Float in [0.0, 1.0]. Higher means the prompt likely needs the
            strong (cloud) model.  Returns 0.5 (uncertain) on any error
            or when the router is not available.
        """
        if not self._available:
            return 0.5

        if not prompt or not prompt.strip():
            return 0.5

        t0 = time.monotonic()

        # Step 1: Get prompt embedding (from cache or OpenAI API)
        embedding = await self._get_embedding(prompt)
        if embedding is None:
            self._api_errors += 1
            return 0.5

        # Step 2: Forward pass (pure numpy, <0.1ms)
        # Project: (128, D) @ (D,) -> (128,)  where D=1536(openai) or 3072(gemini)
        prompt_proj = self._text_proj @ embedding

        # Element-wise multiply with model embeddings + classify
        # classifier is (1, 128), product is (128,) -> result is (1,)
        logit_strong = float(
            (self._classifier @ (self._embed_strong * prompt_proj)).item()
        )
        logit_weak = float(
            (self._classifier @ (self._embed_weak * prompt_proj)).item()
        )

        # Sigmoid of difference -> win probability
        diff = logit_strong - logit_weak
        diff = float(np.clip(diff, -_LOGIT_CLIP, _LOGIT_CLIP))
        win_rate = 1.0 / (1.0 + np.exp(-diff))

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Update statistics
        self._total_predictions += 1
        if win_rate > self._threshold:
            self._cloud_predictions += 1
        # Exponential moving average for latency tracking
        alpha = 0.1
        self._avg_latency_ms = (
            alpha * elapsed_ms + (1 - alpha) * self._avg_latency_ms
        )

        logger.debug(
            "[predict] win_rate=%.4f threshold=%.2f -> %s | "
            "logit_s=%.4f logit_w=%.4f diff=%.4f | latency=%.1fms",
            win_rate,
            self._threshold,
            "CLOUD" if win_rate > self._threshold else "LOCAL",
            logit_strong,
            logit_weak,
            diff,
            elapsed_ms,
        )

        return float(win_rate)

    async def should_route_to_cloud(self, prompt: str) -> Tuple[bool, float]:
        """Determine if a prompt should be routed to cloud.

        Convenience method that calls :meth:`predict` and compares against
        the current threshold.

        Args:
            prompt: User message text.

        Returns:
            ``(should_use_cloud, win_rate)`` tuple.
        """
        win_rate = await self.predict(prompt)
        return win_rate > self._threshold, win_rate

    def get_stats(self) -> Dict[str, Any]:
        """Return MF Router statistics.

        Returns:
            Dict with prediction counts, cloud percentage, average latency,
            API error count, threshold, cache size, and availability status.
        """
        cloud_pct = (
            (self._cloud_predictions / self._total_predictions * 100)
            if self._total_predictions > 0
            else 0.0
        )
        return {
            "total_predictions": self._total_predictions,
            "cloud_predictions": self._cloud_predictions,
            "local_predictions": self._total_predictions - self._cloud_predictions,
            "cloud_percentage": round(cloud_pct, 1),
            "avg_latency_ms": round(self._avg_latency_ms, 1),
            "api_errors": self._api_errors,
            "threshold": self._threshold,
            "backend": self._backend,
            "embedding_dim": self._embedding_dim,
            "cache_size": len(self._embedding_cache),
            "cache_max": self._cache_max,
            "available": self._available,
        }

    # ------------------------------------------------------------------
    # Embedding retrieval (OpenAI / Gemini API + LRU cache)
    # ------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get text embedding from the configured API with LRU caching.

        Auto-dispatches to OpenAI or Gemini based on ``self._backend``.

        Args:
            text: Raw prompt text.

        Returns:
            1-D numpy array of shape ``(embed_dim,)`` with dtype float32,
            or ``None`` on any API error or timeout.
        """
        # Cache lookup (use first N chars as key)
        cache_key = text[:_CACHE_KEY_PREFIX_LEN]
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if self._backend == "gemini":
            embedding = await self._get_gemini_embedding(text)
        else:
            embedding = await self._get_openai_embedding(text)

        if embedding is not None:
            self._cache_put(cache_key, embedding)
        return embedding

    async def _get_openai_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from OpenAI text-embedding-3-small API."""
        import aiohttp

        api_input = text[:_OPENAI_MAX_INPUT_CHARS]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _OPENAI_EMBEDDINGS_URL,
                    headers={
                        "Authorization": f"Bearer {self._openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _OPENAI_EMBEDDING_MODEL,
                        "input": api_input,
                    },
                    timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT_SECONDS),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data["data"][0]["embedding"]
                        embedding = np.array(raw, dtype=np.float32)
                        if embedding.shape != (_OPENAI_EMBEDDING_DIM,):
                            logger.warning(
                                "[openai] unexpected dim: %s (expected %d)",
                                embedding.shape, _OPENAI_EMBEDDING_DIM,
                            )
                            return None
                        return embedding

                    body = await resp.text()
                    logger.warning(
                        "[openai] API error: HTTP %d %s", resp.status, body[:200],
                    )
                    return None

        except asyncio.TimeoutError:
            logger.warning("[openai] timeout (%ds)", _API_TIMEOUT_SECONDS)
            return None
        except Exception as exc:
            logger.warning("[openai] error: %s", exc)
            return None

    async def _get_gemini_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Gemini gemini-embedding-001 API."""
        import urllib.request
        import json as _json

        api_input = text[:_GEMINI_MAX_INPUT_CHARS]

        url = f"{_GEMINI_EMBEDDINGS_URL}?key={self._gemini_api_key}"
        payload = _json.dumps({
            "content": {"parts": [{"text": api_input or " "}]},
        }).encode()

        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            # Run blocking call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            resp_bytes = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=_API_TIMEOUT_SECONDS).read(),
            )
            result = _json.loads(resp_bytes)
            values = result.get("embedding", {}).get("values", [])
            if len(values) != _GEMINI_EMBEDDING_DIM:
                logger.warning(
                    "[gemini] unexpected dim: %d (expected %d)",
                    len(values), _GEMINI_EMBEDDING_DIM,
                )
                return None

            return np.array(values, dtype=np.float32)

        except Exception as exc:
            logger.warning("[gemini] error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # LRU cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve an entry from the LRU cache, promoting it to most-recent.

        Args:
            key: Cache key string.

        Returns:
            The cached numpy embedding, or ``None`` if not found.
        """
        if key not in self._embedding_cache:
            return None
        # Promote to most-recent
        self._cache_order.remove(key)
        self._cache_order.append(key)
        return self._embedding_cache[key]

    def _cache_put(self, key: str, value: np.ndarray) -> None:
        """Insert an entry into the LRU cache, evicting oldest if full.

        Args:
            key: Cache key string.
            value: Numpy embedding array to cache.
        """
        if key in self._embedding_cache:
            # Already present, just promote
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return

        if len(self._embedding_cache) >= self._cache_max:
            # Evict oldest entry
            oldest = self._cache_order.pop(0)
            del self._embedding_cache[oldest]

        self._embedding_cache[key] = value
        self._cache_order.append(key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector in-place, handling zero-norm gracefully.

        Args:
            vec: 1-D numpy array to normalize.

        Returns:
            The normalized vector (same object, modified in-place).
        """
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
