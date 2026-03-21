# SPDX-License-Identifier: Apache-2.0
"""CloudRouter - Main entry point for edge-cloud collaborative routing.

Wires together backends, dispatcher, and budget checker based on
CloudSettingsV2 configuration. Provides a clean interface for server.py
to intercept cloud model requests.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

from omlx.cloud.backends.base import GenerationRequest, GenerationResponse
from omlx.cloud.budget import BudgetChecker
from omlx.cloud.classifier import TaskClassifier
from omlx.cloud.context_pilot import ContextPilotOptimizer
from omlx.cloud.conversation_store import ConversationDB, ConversationStore, TopicSegment
from omlx.cloud.dispatcher import CloudDispatcher
from omlx.cloud.intelligent_router import IntelligentRouter
from omlx.cloud.selector import ModelSelector
from omlx.cloud.semantic_cache import SemanticCache, SemanticCacheDB

logger = logging.getLogger("omlx.cloud.router")


class CloudRouter:
    """Routes requests to cloud backends when the model is a cloud model.

    Initializes available backends based on which API keys are present
    in CloudSettingsV2, creates the dispatcher and budget checker,
    and provides generate/generate_stream methods for server integration.
    """

    # Approximate pricing per 1K tokens (USD)
    _MODEL_PRICING = {
        # DeepSeek
        "deepseek-chat": {"input": 0.00027, "output": 0.0011},
        "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},
        # Gemini
        "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-2-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-2-pro": {"input": 0.00125, "output": 0.005},
        # OpenAI
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        # GLM
        "glm-4-plus": {"input": 0.001, "output": 0.001},
        "glm-4-flash": {"input": 0.0001, "output": 0.0001},
        "glm-5": {"input": 0.001, "output": 0.001},
    }
    _DEFAULT_PRICING = {"input": 0.001, "output": 0.002}

    def __init__(self, settings) -> None:
        """Initialize CloudRouter from CloudSettingsV2.

        Args:
            settings: A CloudSettingsV2 instance with API keys and config.
        """
        self._settings = settings
        self._cloud_models: set[str] = set(settings.cloud_models)

        # Initialize backends based on available API keys
        backends: Dict[str, Any] = {}
        self._init_backends(settings, backends)

        if not backends:
            logger.warning(
                "[CloudRouter] No cloud backends available "
                "(no API keys configured)"
            )

        # Create dispatcher
        self._dispatcher = CloudDispatcher(backends=backends)

        # Create budget checker
        db_path = self._resolve_budget_db_path(settings)
        self._budget = BudgetChecker(
            db_path=db_path,
            config={
                "daily_limit": settings.daily_budget,
                "monthly_limit": settings.monthly_budget,
            },
        )

        # Phase 2: Context optimization
        self._semantic_cache: Optional[SemanticCache] = None
        self._context_pilot: Optional[ContextPilotOptimizer] = None
        self._conversation_store: Optional[ConversationStore] = None

        if getattr(settings, "semantic_cache_enabled", False):
            try:
                cache_db = SemanticCacheDB()
                self._semantic_cache = SemanticCache(
                    db=cache_db,
                    threshold=getattr(
                        settings, "semantic_cache_threshold", 0.85
                    ),
                    ttl_hours=getattr(
                        settings, "semantic_cache_ttl", 14400
                    )
                    // 3600,
                )
                logger.info("[CloudRouter] SemanticCache initialized")
            except Exception as exc:
                logger.warning(
                    "[CloudRouter] SemanticCache init failed: %s", exc
                )

        if getattr(settings, "context_pilot_enabled", False):
            try:
                self._context_pilot = ContextPilotOptimizer(enabled=True)
                logger.info("[CloudRouter] ContextPilot initialized")
            except Exception as exc:
                logger.warning(
                    "[CloudRouter] ContextPilot init failed: %s", exc
                )

        if getattr(settings, "conversation_store_enabled", False):
            try:
                db_path = getattr(
                    settings, "conversation_db_path", None
                )
                conv_db = ConversationDB(db_path=db_path) if db_path else ConversationDB()
                self._conversation_store = ConversationStore(db=conv_db)
                logger.info("[CloudRouter] ConversationStore initialized")
            except Exception as exc:
                logger.warning(
                    "[CloudRouter] ConversationStore init failed: %s", exc
                )

        self._backends = backends

        # Phase: Intelligent Router (model="auto")
        self._intelligent_router: Optional[IntelligentRouter] = None
        if getattr(settings, "intelligent_routing_enabled", False):
            try:
                self._intelligent_router = IntelligentRouter(
                    classifier=TaskClassifier(),
                    selector=ModelSelector(),
                    settings=settings,
                    budget_checker=self._budget,
                )
                logger.info("[CloudRouter] IntelligentRouter initialized")
            except Exception as exc:
                logger.warning(
                    "[CloudRouter] IntelligentRouter init failed: %s", exc
                )

        logger.info(
            "[CloudRouter] Initialized with %d backends: %s",
            len(backends),
            list(backends.keys()),
        )

    @staticmethod
    def _init_backends(settings, backends: Dict[str, Any]) -> None:
        """Try to initialize each backend; skip those without API keys."""
        # DeepSeek
        if settings.deepseek_api_key:
            try:
                from omlx.cloud.backends.deepseek import DeepSeekBackend

                backends["deepseek"] = DeepSeekBackend(
                    api_key=settings.deepseek_api_key,
                )
                logger.info("[CloudRouter] DeepSeek backend registered")
            except Exception as exc:
                logger.warning("[CloudRouter] DeepSeek init failed: %s", exc)

        # GLM
        if settings.glm_api_key:
            try:
                from omlx.cloud.backends.glm import GLMBackend

                backends["glm"] = GLMBackend(api_key=settings.glm_api_key)
                logger.info("[CloudRouter] GLM backend registered")
            except Exception as exc:
                logger.warning("[CloudRouter] GLM init failed: %s", exc)

        # OpenAI
        if settings.openai_api_key:
            try:
                from omlx.cloud.backends.openai_backend import OpenAIBackend

                backends["openai"] = OpenAIBackend(
                    api_key=settings.openai_api_key,
                )
                logger.info("[CloudRouter] OpenAI backend registered")
            except Exception as exc:
                logger.warning("[CloudRouter] OpenAI init failed: %s", exc)

        # Gemini
        if settings.gemini_api_key:
            try:
                from omlx.cloud.backends.gemini import GeminiBackend

                backends["gemini"] = GeminiBackend(
                    api_key=settings.gemini_api_key,
                )
                logger.info("[CloudRouter] Gemini backend registered")
            except Exception as exc:
                logger.warning("[CloudRouter] Gemini init failed: %s", exc)

        # ChatGPT (subscription-based, uses access_token)
        if settings.chatgpt_access_token:
            try:
                from omlx.cloud.backends.chatgpt import ChatGPTBackend

                backends["chatgpt"] = ChatGPTBackend(
                    access_token=settings.chatgpt_access_token,
                )
                logger.info("[CloudRouter] ChatGPT backend registered")
            except Exception as exc:
                logger.warning("[CloudRouter] ChatGPT init failed: %s", exc)

    @staticmethod
    def _resolve_budget_db_path(settings) -> str:
        """Resolve the budget database path from settings."""
        if settings.conversation_db_path:
            return settings.conversation_db_path
        # Default: ~/.omlx/cloud_budget.db
        default_path = Path.home() / ".omlx" / "cloud_budget.db"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        return str(default_path)

    @property
    def intelligent_router(self) -> Optional[IntelligentRouter]:
        """Return the IntelligentRouter instance, or None if disabled."""
        return self._intelligent_router

    async def is_cloud_model(self, model: Optional[str]) -> bool:
        """Check if a model should be routed to the cloud.

        Returns True only if cloud routing is enabled AND the model
        is in the configured cloud_models list.
        """
        if not model or not self._settings.enabled:
            return False
        return model in self._cloud_models

    async def generate(self, request: Any, model: str) -> Dict:
        """Generate a non-streaming cloud response.

        Flow: budget -> cache lookup -> context optimize -> dispatch
              -> store -> return

        Args:
            request: The incoming ChatCompletionRequest (OpenAI format).
            model: The model name to route to.

        Returns:
            OpenAI ChatCompletion-format dict.

        Raises:
            HTTPException-compatible error if budget exceeded or dispatch fails.
        """
        # 1. Budget check
        budget_status = self._budget.check()
        if not budget_status["allowed"]:
            return {
                "error": {
                    "message": budget_status["reason"],
                    "type": "budget_exceeded",
                    "code": "budget_exceeded",
                }
            }

        # 2. Convert to GenerationRequest
        gen_request = self._to_generation_request(request)

        # 3. Semantic cache lookup (non-streaming only)
        last_user_msg = self._extract_last_user_message(gen_request.messages)
        if self._semantic_cache and last_user_msg:
            cached = self._semantic_cache.lookup(last_user_msg, model)
            if cached and "response" in cached:
                logger.info(
                    "[CloudRouter] SemanticCache HIT for model=%s", model
                )
                return cached["response"]

        # 4. ContextPilot optimization
        if self._context_pilot and gen_request.messages:
            optimized_msgs, cp_meta = self._context_pilot.optimize(
                gen_request.messages
            )
            if cp_meta.get("optimized"):
                gen_request.messages = optimized_msgs
                logger.info(
                    "[CloudRouter] ContextPilot: %s (%s)",
                    cp_meta.get("method", "unknown"),
                    cp_meta,
                )

        # 5. Dispatch
        try:
            response, backend_name = await self._dispatcher.dispatch(
                gen_request, model
            )
        except RuntimeError as exc:
            logger.error("[CloudRouter] Dispatch failed for %s: %s", model, exc)
            return {
                "error": {
                    "message": str(exc),
                    "type": "dispatch_error",
                    "code": "all_backends_exhausted",
                }
            }
        logger.info(
            "[CloudRouter] %s served by %s (%d input, %d output tokens)",
            model,
            backend_name,
            response.input_tokens,
            response.output_tokens,
        )

        # 5b. Record cost to budget DB
        self._record_cost(model, response.input_tokens, response.output_tokens, response.total_time)

        # 6. Convert to OpenAI ChatCompletion format
        result = self._to_chat_completion(response, model)

        # 7. Store in SemanticCache
        if self._semantic_cache and last_user_msg:
            try:
                self._semantic_cache.store(last_user_msg, model, result)
            except Exception as exc:
                logger.warning(
                    "[CloudRouter] SemanticCache store failed: %s", exc
                )

        # 8. Store in ConversationStore
        if self._conversation_store and gen_request.messages:
            try:
                conv_id = self._conversation_store.derive_conversation_id(
                    gen_request.messages
                )
                segment = TopicSegment(
                    start=0,
                    end=len(gen_request.messages),
                    topic_type="work",
                    confidence=0.8,
                )
                segment.messages = gen_request.messages + [
                    {"role": "assistant", "content": response.content}
                ]
                self._conversation_store.store_segments(conv_id, [segment])
            except Exception as exc:
                logger.warning(
                    "[CloudRouter] ConversationStore save failed: %s", exc
                )

        return result

    async def generate_stream(
        self, request: Any, model: str
    ) -> AsyncIterator[str]:
        """Generate a streaming cloud response as SSE chunks.

        Args:
            request: The incoming ChatCompletionRequest (OpenAI format).
            model: The model name to route to.

        Yields:
            SSE-formatted strings (data: {...}\\n\\n).
        """
        # Budget check
        budget_status = self._budget.check()
        if not budget_status["allowed"]:
            error_data = json.dumps(
                {
                    "error": {
                        "message": budget_status["reason"],
                        "type": "budget_exceeded",
                    }
                }
            )
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
            return

        gen_request = self._to_generation_request(request)
        gen_request.stream = True

        # ContextPilot optimization
        if self._context_pilot and gen_request.messages:
            optimized_msgs, cp_meta = self._context_pilot.optimize(
                gen_request.messages
            )
            if cp_meta.get("optimized"):
                gen_request.messages = optimized_msgs
                logger.info(
                    "[CloudRouter] Stream ContextPilot: %s",
                    cp_meta.get("method"),
                )

        try:
            stream, backend_name = await self._dispatcher.dispatch_stream(
                gen_request, model
            )
        except RuntimeError as exc:
            error_data = json.dumps(
                {"error": {"message": str(exc), "type": "dispatch_error"}}
            )
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"
            return

        chunk_id = f"chatcmpl-{uuid4().hex[:12]}"
        created = int(time.time())
        stream_start = time.time()
        total_output_chars = 0

        # Estimate input tokens from messages (rough: 4 chars per token)
        input_chars = sum(
            len(m.get("content", "")) for m in gen_request.messages
        )
        est_input_tokens = max(input_chars // 4, 1)

        try:
            async for token in stream:
                total_output_chars += len(token)
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            self._dispatcher.record_stream_success(backend_name)

            # Record streaming cost (estimated tokens)
            est_output_tokens = max(total_output_chars // 4, 1)
            self._record_cost(
                model, est_input_tokens, est_output_tokens,
                time.time() - stream_start,
            )
        except Exception as exc:
            logger.error(
                "[CloudRouter] Stream error on %s: %s", backend_name, exc
            )
            self._dispatcher.record_stream_failure(backend_name)
            error_data = json.dumps(
                {"error": {"message": str(exc), "type": "stream_error"}}
            )
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    def get_health(self) -> Dict:
        """Return health status for all cloud backends plus budget info."""
        result = {
            "enabled": self._settings.enabled,
            "backends": self._dispatcher.get_health(),
            "budget": self._budget.get_budget_info(),
            "registered_backends": list(self._backends.keys()),
            "cloud_models": sorted(self._cloud_models),
        }
        # Phase 2 status
        if self._semantic_cache:
            result["semantic_cache"] = {"enabled": True}
        if self._context_pilot:
            result["context_pilot"] = self._context_pilot.get_stats()
        if self._conversation_store:
            result["conversation_store"] = {"enabled": True}
        return result

    def _record_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_time: float,
    ) -> None:
        """Calculate and record request cost to budget DB."""
        pricing = self._MODEL_PRICING.get(model, self._DEFAULT_PRICING)
        cost = (
            (input_tokens / 1000) * pricing["input"]
            + (output_tokens / 1000) * pricing["output"]
        )
        latency_ms = total_time * 1000

        try:
            self._budget.db_helper.record_request(
                model=model,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            self._budget.invalidate_cache()
            logger.debug(
                "[CloudRouter] Recorded cost $%.6f for %s (%d in, %d out)",
                cost,
                model,
                input_tokens,
                output_tokens,
            )
        except Exception as exc:
            logger.warning("[CloudRouter] Failed to record cost: %s", exc)

    @staticmethod
    def _extract_last_user_message(messages: list) -> str:
        """Extract the last user message content from a message list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @staticmethod
    def _to_generation_request(request: Any) -> GenerationRequest:
        """Convert an OpenAI ChatCompletionRequest to GenerationRequest."""
        messages = []
        if hasattr(request, "messages") and request.messages:
            for msg in request.messages:
                m = {"role": msg.role}
                if hasattr(msg, "content") and msg.content is not None:
                    m["content"] = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                messages.append(m)

        return GenerationRequest(
            messages=messages,
            max_tokens=getattr(request, "max_tokens", None),
            temperature=getattr(request, "temperature", None),
            top_p=getattr(request, "top_p", None),
            stream=getattr(request, "stream", False) or False,
            # KB-019: Pass through parameters previously dropped
            tools=getattr(request, "tools", None),
            tool_choice=getattr(request, "tool_choice", None),
            response_format=getattr(request, "response_format", None),
            stop=getattr(request, "stop", None),
            stream_options=getattr(request, "stream_options", None),
            presence_penalty=getattr(request, "presence_penalty", None),
            frequency_penalty=getattr(request, "frequency_penalty", None),
            min_p=getattr(request, "min_p", None),
            n=getattr(request, "n", None),
            chat_template_kwargs=getattr(request, "chat_template_kwargs", None),
        )

    @staticmethod
    def _to_chat_completion(
        response: GenerationResponse, model: str
    ) -> Dict:
        """Convert GenerationResponse to OpenAI ChatCompletion format."""
        return {
            "id": f"chatcmpl-{uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
            },
        }
