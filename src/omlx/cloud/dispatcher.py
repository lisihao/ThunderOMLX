"""CloudDispatcher - Retry + Exponential Backoff + Provider Fallback + Circuit Breaker

Unified cloud request dispatcher that handles:
- Retry with exponential backoff for transient errors (5xx, timeout, network)
- Provider fallback chains (e.g., glm fails -> try deepseek)
- Circuit breaker per backend (CLOSED -> OPEN after N failures, HALF_OPEN after timeout)
- Model name mapping (friendly name -> API name)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, AsyncGenerator, List

import httpx

from omlx.cloud.backends.base import GenerationRequest, GenerationResponse

logger = logging.getLogger("omlx.cloud.dispatcher")


# ========== Circuit Breaker ==========


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class BackendHealth:
    """Per-backend circuit breaker state"""
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    failure_threshold: int = 3       # consecutive failures -> OPEN
    recovery_timeout: float = 60.0   # seconds before HALF_OPEN
    last_failure_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0

    def record_success(self):
        self.total_requests += 1
        self.total_successes += 1
        self.consecutive_failures = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"[CircuitBreaker] HALF_OPEN -> CLOSED (recovery success)")

    def record_failure(self):
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(
                    f"[CircuitBreaker] -> OPEN (consecutive_failures={self.consecutive_failures})"
                )
            self.state = CircuitState.OPEN

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[CircuitBreaker] OPEN -> HALF_OPEN (elapsed={elapsed:.0f}s)")
                return True
            return False
        # HALF_OPEN: allow one request to test
        return True

    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "success_rate": (
                self.total_successes / self.total_requests
                if self.total_requests > 0 else 1.0
            ),
        }


# ========== Retry Logic ==========


def _is_retryable(exc: Exception) -> bool:
    """Determine if an exception is retryable (transient)"""
    # httpx timeouts and connection errors
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    # HTTP 5xx server errors and 429 Too Many Requests
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    # Generic connection errors
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    return False


def _get_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After delay from a 429 response, or None if not applicable"""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                return None
    return None


# ========== Model Routing ==========


# Friendly model name -> (primary_backend, API model name)
MODEL_BACKEND_MAP: Dict[str, Tuple[str, str]] = {
    # GLM models
    "glm-4-flash": ("glm", "glm-4-flash"),
    "glm-4-plus": ("glm", "glm-4-plus"),
    "glm-5": ("glm", "glm-5"),
    # DeepSeek models
    "deepseek-v3": ("deepseek", "deepseek-chat"),
    "deepseek-r1": ("deepseek", "deepseek-reasoner"),
    # ChatGPT subscription models
    "gpt-5.2": ("chatgpt", "gpt-5.2"),
    "gpt-5.1": ("chatgpt", "gpt-5.1"),
    # OpenAI API models
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    # Gemini models
    "gemini-2.5-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-2.5-pro": ("gemini", "gemini-2.5-pro"),
    "gemini-2-flash": ("gemini", "gemini-2.0-flash"),
    "gemini-2-pro": ("gemini", "gemini-2.0-pro"),
}

# Fallback chains: if primary backend fails, try alternatives
FALLBACK_CHAINS: Dict[str, List[str]] = {
    "glm": ["glm", "deepseek"],
    "deepseek": ["deepseek", "glm"],
    "chatgpt": ["chatgpt", "openai"],
    "openai": ["openai", "chatgpt"],
    "gemini": ["gemini"],  # no compatible fallback
}

# When falling back to a different backend, remap the model name
FALLBACK_MODEL_MAP: Dict[Tuple[str, str], str] = {
    # glm model on deepseek backend
    ("glm-4-flash", "deepseek"): "deepseek-chat",
    ("glm-4-plus", "deepseek"): "deepseek-chat",
    ("glm-5", "deepseek"): "deepseek-chat",
    # deepseek model on glm backend
    ("deepseek-v3", "glm"): "glm-5",
    ("deepseek-r1", "glm"): "glm-5",
    # gpt models cross-backend
    ("gpt-5.2", "openai"): "gpt-4o",
    ("gpt-5.1", "openai"): "gpt-4o",
    ("gpt-4o", "chatgpt"): "gpt-5.2",
    ("gpt-4o-mini", "chatgpt"): "gpt-5.1",
}


# ========== CloudDispatcher ==========


class CloudDispatcher:
    """Unified cloud request dispatcher with retry, fallback, and circuit breaker"""

    def __init__(
        self,
        backends: Dict,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 8.0,
    ):
        self.backends = backends
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

        # Per-backend health tracking
        self._health: Dict[str, BackendHealth] = {
            name: BackendHealth() for name in backends
        }

        # Per-backend in-flight request counting
        self._in_flight: Dict[str, int] = {name: 0 for name in backends}

    def _resolve_model(self, model: str) -> Tuple[str, str]:
        """Resolve friendly model name to (primary_backend, api_model_name)"""
        if model in MODEL_BACKEND_MAP:
            return MODEL_BACKEND_MAP[model]

        # Prefix-based fallback
        if model.startswith("glm"):
            return "glm", model
        elif model.startswith("gpt-5"):
            return "chatgpt", model
        elif model.startswith("gpt"):
            return "openai", model
        elif model.startswith("deepseek"):
            return "deepseek", model
        elif model.startswith("gemini"):
            return "gemini", model

        return "openai", model  # default

    def _get_fallback_chain(self, primary_backend: str) -> List[str]:
        """Get ordered list of backends to try"""
        chain = FALLBACK_CHAINS.get(primary_backend, [primary_backend])
        # Filter to only available backends
        return [b for b in chain if b in self.backends]

    def _get_api_model(self, model: str, backend_name: str, primary_backend: str) -> str:
        """Get the API model name for a given backend"""
        if backend_name == primary_backend:
            # Primary backend: use the mapped API name
            if model in MODEL_BACKEND_MAP:
                return MODEL_BACKEND_MAP[model][1]
            return model
        else:
            # Fallback backend: remap model name
            return FALLBACK_MODEL_MAP.get((model, backend_name), model)

    async def _call_with_retry(
        self,
        backend,
        backend_name: str,
        request: GenerationRequest,
        api_model: str,
    ) -> GenerationResponse:
        """Call a single backend with retry + exponential backoff"""
        last_exc = None
        self._in_flight[backend_name] = self._in_flight.get(backend_name, 0) + 1

        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await backend.generate(request, model=api_model)
                    self._health[backend_name].record_success()
                    return response
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable(exc) or attempt == self.max_retries:
                        self._health[backend_name].record_failure()
                        raise
                    # Use Retry-After header for 429 responses, else exponential backoff
                    retry_after = _get_retry_after(exc)
                    if retry_after is not None:
                        delay = min(retry_after, self.backoff_max)
                    else:
                        delay = min(
                            self.backoff_base * (2 ** (attempt - 1)),
                            self.backoff_max,
                        )
                    logger.warning(
                        f"[Dispatcher] {backend_name} attempt {attempt}/{self.max_retries} "
                        f"failed: {type(exc).__name__}: {exc} | retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            self._health[backend_name].record_failure()
            raise last_exc
        finally:
            self._in_flight[backend_name] = max(0, self._in_flight.get(backend_name, 1) - 1)

    async def dispatch(
        self, request: GenerationRequest, model: str
    ) -> Tuple[GenerationResponse, str]:
        """Dispatch a non-streaming request with retry + fallback

        Returns:
            (response, backend_name) - the response and which backend served it
        """
        primary_backend, _ = self._resolve_model(model)
        chain = self._get_fallback_chain(primary_backend)

        if not chain:
            raise RuntimeError(
                f"No available backend for model '{model}' "
                f"(primary={primary_backend}, available={list(self.backends.keys())})"
            )

        last_exc = None
        for backend_name in chain:
            health = self._health[backend_name]
            if not health.is_available():
                logger.info(
                    f"[Dispatcher] Skipping {backend_name} "
                    f"(circuit={health.state.value})"
                )
                continue

            backend = self.backends[backend_name]
            api_model = self._get_api_model(model, backend_name, primary_backend)

            logger.info(
                f"[Dispatcher] Trying {backend_name} | "
                f"model={model} -> api_model={api_model}"
            )

            try:
                response = await self._call_with_retry(
                    backend, backend_name, request, api_model
                )
                if backend_name != primary_backend:
                    logger.info(
                        f"[Dispatcher] Fallback success: "
                        f"{primary_backend} -> {backend_name}"
                    )
                return response, backend_name
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[Dispatcher] {backend_name} exhausted retries: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

        raise RuntimeError(
            f"All backends exhausted for model '{model}': {last_exc}"
        ) from last_exc

    async def dispatch_stream(
        self, request: GenerationRequest, model: str
    ) -> Tuple[AsyncGenerator[str, None], str]:
        """Dispatch a streaming request with fallback (no retry for streams)

        Returns:
            (stream_generator, backend_name)
        """
        primary_backend, _ = self._resolve_model(model)
        chain = self._get_fallback_chain(primary_backend)

        if not chain:
            raise RuntimeError(f"No available backend for model '{model}'")

        last_exc = None
        for backend_name in chain:
            health = self._health[backend_name]
            if not health.is_available():
                continue

            backend = self.backends[backend_name]
            api_model = self._get_api_model(model, backend_name, primary_backend)

            try:
                stream = backend.generate_stream(request, model=api_model)
                # We can't fully validate a stream before returning it,
                # but record intent. Success/failure tracked by caller.
                return stream, backend_name
            except Exception as exc:
                last_exc = exc
                self._health[backend_name].record_failure()
                logger.warning(
                    f"[Dispatcher] Stream init failed on {backend_name}: {exc}"
                )
                continue

        raise RuntimeError(
            f"All backends exhausted for streaming model '{model}': {last_exc}"
        ) from last_exc

    def record_stream_success(self, backend_name: str):
        """Call after a stream completes successfully"""
        if backend_name in self._health:
            self._health[backend_name].record_success()

    def record_stream_failure(self, backend_name: str):
        """Call after a stream fails"""
        if backend_name in self._health:
            self._health[backend_name].record_failure()

    def get_in_flight(self) -> Dict[str, int]:
        """Get per-backend in-flight request counts"""
        return dict(self._in_flight)

    def get_health(self) -> Dict[str, Dict]:
        """Get health status for all backends (includes in-flight)"""
        result = {}
        for name, health in self._health.items():
            d = health.to_dict()
            d["in_flight"] = self._in_flight.get(name, 0)
            result[name] = d
        return result
