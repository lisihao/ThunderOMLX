"""ML-assisted task classifier using a small 0.8B local model.

When the rule-based TaskClassifier has low confidence, this module
calls a small local model (e.g. qwen3.5-0.8b) via the server's own
``/v1/chat/completions`` endpoint for more accurate classification.

Typical latency: <200ms for the 0.8B model generating ~50 tokens.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger("omlx.cloud.ml_classifier")

# Valid task types the 0.8B model may emit
_VALID_TASK_TYPES = frozenset({
    "completion", "fix", "docs", "debug",
    "architecture", "security", "refactor", "explanation",
})

# Map 0.8B model output -> CodingSubtask enum values used by the router
_TASK_TYPE_MAP: Dict[str, str] = {
    "completion": "completion",
    "fix": "simple_fix",
    "docs": "documentation",
    "debug": "debugging",
    "architecture": "architecture",
    "security": "security",
    "refactor": "refactor",
    "explanation": "explanation",
}

_VALID_COMPLEXITIES = frozenset({"low", "medium", "high"})

_CLASSIFY_SYSTEM_PROMPT = (
    "你是路由分类器。分析用户请求，返回 JSON:\n"
    "{\n"
    '  "task_type": "completion|fix|docs|debug|architecture|security|refactor|explanation",\n'
    '  "complexity": "low|medium|high",\n'
    '  "needs_cloud": true/false,\n'
    '  "reason": "一句话理由"\n'
    "}\n"
    "只输出 JSON，不要其他内容。不要用 markdown 代码块。"
)

_MAX_INPUT_CHARS = 2000

# Regex to extract the first JSON object from potentially noisy output
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class MLClassifier:
    """0.8B model-assisted task classifier.

    Calls the local server's own ``/v1/chat/completions`` endpoint with a
    small model for quick task classification.  Designed to supplement the
    rule-based ``TaskClassifier`` when keyword-matching confidence is low.

    All errors are swallowed and logged — ``classify()`` returns ``None``
    on any failure so the caller can fall back to rule-based results.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5-0.8b-opus-distilled",
        base_url: str = "http://localhost:8082",
        api_key: str = "",
        timeout: float = 3.0,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            "[__init__] MLClassifier ready | model=%s base_url=%s timeout=%.1fs",
            model_name, self._base_url, timeout,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create the aiohttp session on first use."""
        if self._session is None or self._session.closed:
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=headers,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("[close] MLClassifier session closed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(self, message: str) -> Optional[Dict[str, Any]]:
        """Classify a user message using the 0.8B model.

        Args:
            message: The last user message text.

        Returns:
            A dict with ``task_type`` (mapped to CodingSubtask values),
            ``complexity``, ``needs_cloud``, and ``reason``.
            Returns ``None`` on any error or timeout.
        """
        if not message or not message.strip():
            return None

        # Truncate to avoid overwhelming the small model
        truncated = message[:_MAX_INPUT_CHARS]

        try:
            session = self._get_session()
            payload = {
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": truncated},
                ],
                "temperature": 0.1,
                "max_tokens": 100,
                "stream": False,
            }

            url = f"{self._base_url}/v1/chat/completions"
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "[classify] HTTP %d from %s: %s",
                        resp.status, url, body[:200],
                    )
                    return None

                data = await resp.json()

            # Extract the assistant reply
            raw_text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            return self._parse_response(raw_text)

        except asyncio.TimeoutError:
            logger.warning("[classify] timeout after %.1fs", self._timeout.total)
            return None
        except aiohttp.ClientError as exc:
            logger.warning("[classify] HTTP error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("[classify] unexpected error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Parsing & validation
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse and validate the 0.8B model's JSON response.

        Attempts direct ``json.loads`` first, then falls back to regex
        extraction if the model wrapped the JSON in extra text.
        """
        if not raw:
            return None

        parsed = self._try_parse_json(raw)
        if parsed is None:
            return None

        return self._validate(parsed)

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        """Try to parse JSON, with regex fallback."""
        text = text.strip()

        # Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Regex fallback: find first {...}
        match = _JSON_RE.search(text)
        if match:
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        logger.warning("[_try_parse_json] failed to parse: %s", text[:100])
        return None

    @staticmethod
    def _validate(data: Dict) -> Optional[Dict[str, Any]]:
        """Validate parsed JSON fields and map task_type."""
        raw_type = str(data.get("task_type", "")).lower().strip()
        complexity = str(data.get("complexity", "")).lower().strip()
        needs_cloud = data.get("needs_cloud")
        reason = str(data.get("reason", ""))

        # Validate task_type
        if raw_type not in _VALID_TASK_TYPES:
            logger.warning("[_validate] invalid task_type: %s", raw_type)
            return None

        # Validate complexity
        if complexity not in _VALID_COMPLEXITIES:
            complexity = "medium"  # Safe default

        # Validate needs_cloud
        if not isinstance(needs_cloud, bool):
            needs_cloud = raw_type in ("architecture", "security")

        # Map to CodingSubtask enum values
        mapped_type = _TASK_TYPE_MAP.get(raw_type, raw_type)

        result = {
            "task_type": mapped_type,
            "complexity": complexity,
            "needs_cloud": needs_cloud,
            "reason": reason[:200],  # Cap reason length
        }

        logger.info(
            "[classify] 0.8B result: type=%s complexity=%s cloud=%s reason=%s",
            mapped_type, complexity, needs_cloud, reason[:60],
        )
        return result


# Required for async timeout handling
import asyncio  # noqa: E402
