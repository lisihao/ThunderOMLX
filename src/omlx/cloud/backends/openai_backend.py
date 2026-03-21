"""OpenAI API Backend"""

import os
import httpx
from typing import List, Dict, AsyncGenerator, Optional
import time

from .base import BaseEngine, GenerationRequest, GenerationResponse


class OpenAIBackend(BaseEngine):
    """OpenAI API 后端

    支持模型:
    - gpt-5.2 (最新)
    - gpt-5.1
    - gpt-4o (已废弃)

    注意：用户使用订阅账户，可能需要特殊访问方式
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, request: GenerationRequest, model: str = "gpt-4o") -> GenerationResponse:
        """非流式生成"""
        start_time = time.time()

        # 构建请求
        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_tokens": request.max_tokens or 2048,
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

        # 调用 API
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

        # 解析响应
        choice = data["choices"][0]
        content = choice["message"].get("content", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        end_time = time.time()

        return GenerationResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft=end_time - start_time,
            total_time=end_time - start_time,
        )

    async def generate_stream(
        self, request: GenerationRequest, model: str = "gpt-4o"
    ) -> AsyncGenerator[str, None]:
        """流式生成"""
        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "max_tokens": request.max_tokens or 2048,
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
            "engine_type": "openai_backend",
            "base_url": self.base_url,
            "authenticated": bool(self.api_key),
        }

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
