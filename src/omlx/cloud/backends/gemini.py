"""Google Gemini API Backend"""

import os
import httpx
from typing import List, Dict, AsyncGenerator, Optional
import time
import logging

from .base import BaseEngine, GenerationRequest, GenerationResponse

logger = logging.getLogger("omlx.cloud.backends.gemini")

# KB-033: Thinking/reasoning models that need higher token budgets.
# These models use internal chain-of-thought that consumes the max_tokens budget,
# leaving 0 tokens for visible output if the budget is too low.
THINKING_MODELS = {"gemini-2.5-pro", "gemini-2-pro"}


class GeminiBackend(BaseEngine):
    """Google Gemini API 后端 (OpenAI 兼容接口)

    支持模型:
    - gemini-2.5-pro
    - gemini-2.5-flash
    - gemini-2-pro
    - gemini-2-flash

    特点: 内容审查相对宽松，适合处理敏感话题
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # Google AI Studio OpenAI 兼容端点
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found")

        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, request: GenerationRequest, model: str = "gemini-2.5-flash") -> GenerationResponse:
        """非流式生成"""
        start_time = time.time()
        logger.debug(f"[Gemini] POST {self.base_url}/chat/completions | model={model} temp={request.temperature} max_tokens={request.max_tokens}")

        # KB-033: Thinking models need higher token budget to leave room for output
        max_tokens = request.max_tokens or 2048
        if any(t in model for t in THINKING_MODELS):
            max_tokens = max(max_tokens * 4, 8192)

        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        # KB-019: Pass through additional parameters
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.n is not None:
            payload["n"] = request.n

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            logger.debug(f"[Gemini] HTTP {response.status_code} | 耗时={time.time()-start_time:.2f}s")
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[Gemini] HTTP 错误: {e.response.status_code} | body={e.response.text[:200]}")
            raise
        except Exception as e:
            logger.error(f"[Gemini] 请求异常: {type(e).__name__}: {e}")
            raise

        data = response.json()

        choice = data["choices"][0]
        content = choice["message"].get("content", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        end_time = time.time()

        return GenerationResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft=end_time - start_time,
            total_time=end_time - start_time,
        )

    async def generate_stream(
        self, request: GenerationRequest, model: str = "gemini-2.5-flash"
    ) -> AsyncGenerator[str, None]:
        """流式生成"""
        # KB-033: Thinking models need higher token budget to leave room for output
        max_tokens = request.max_tokens or 2048
        if any(t in model for t in THINKING_MODELS):
            max_tokens = max(max_tokens * 4, 8192)

        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        # KB-019: Pass through additional parameters
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.stream_options is not None:
            payload["stream_options"] = request.stream_options
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.n is not None:
            payload["n"] = request.n

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                import json

                try:
                    data = json.loads(data_str)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        yield content
                except Exception:
                    continue

    def get_stats(self) -> Dict:
        """获取引擎统计信息"""
        return {
            "engine_type": "gemini_backend",
            "base_url": self.base_url,
            "authenticated": bool(self.api_key),
        }

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
