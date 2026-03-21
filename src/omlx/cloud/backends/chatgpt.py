"""ChatGPT Backend API - 访问订阅账户配额

使用 ChatGPT 的 Codex API（OpenClaw 方式）。
认证方式：JWT Token + Account ID
"""

import os
import httpx
from typing import List, Dict, AsyncGenerator, Optional
import time
import json
import base64

from .base import BaseEngine, GenerationRequest, GenerationResponse


class ChatGPTBackend(BaseEngine):
    """ChatGPT 后端 API（订阅账户 - OpenClaw 方式）

    支持模型:
    - gpt-5.2
    - gpt-5.1
    - gpt-5.1-codex-max
    - gpt-5.1-codex-mini

    认证：使用 JWT Token（从 OpenClaw 配置中获取）
    """

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or os.getenv("CHATGPT_ACCESS_TOKEN")
        self.base_url = "https://chatgpt.com/backend-api"

        if not self.access_token:
            raise ValueError(
                "CHATGPT_ACCESS_TOKEN not found. "
                "需要从 OpenClaw 获取 JWT Token"
            )

        # 从 JWT token 中提取 accountId
        self.account_id = self._extract_account_id(self.access_token)
        self.client = httpx.AsyncClient(timeout=60.0)

    def _extract_account_id(self, token: str) -> str:
        """从 JWT token 中提取 chatgpt_account_id"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise ValueError("Invalid JWT token format")

            # 解码 payload（base64）
            payload_b64 = parts[1]
            # 添加 padding 如果需要
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload = json.loads(base64.b64decode(payload_b64))

            # 提取 chatgpt_account_id
            auth_claim = payload.get("https://api.openai.com/auth", {})
            account_id = auth_claim.get("chatgpt_account_id")

            if not account_id:
                raise ValueError("No chatgpt_account_id in token")

            return account_id
        except Exception as e:
            raise ValueError(f"Failed to extract accountId from token: {e}")

    async def generate(self, request: GenerationRequest, model: str = "gpt-5.2") -> GenerationResponse:
        """非流式生成"""
        start_time = time.time()

        # 提取系统提示词和用户消息（OpenClaw 格式）
        system_prompt = ""
        user_messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                user_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg["content"]}]
                })

        # 构建 Codex Responses API 请求（OpenClaw 格式）
        # 注意：Codex API 要求必须使用 stream=True
        # Codex API 不支持 temperature 参数
        payload = {
            "model": model,
            "store": False,
            "stream": True,  # Codex API 强制要求流式
            "instructions": system_prompt,
            "input": user_messages,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        # KB-019: Pass through tools if provided
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        # 使用流式调用，但在非流式方法中手动合并所有块
        content = ""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/codex/responses",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "chatgpt-account-id": self.account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "pi",
                "User-Agent": "pi (darwin arm64)",
                "Accept": "text/event-stream",
                "Content-Type": "application/json"
            },
            json=payload
        ) as response:
            response.raise_for_status()

            # 读取流式响应，合并所有内容
            # Codex API 使用事件流格式：
            # - response.output_text.delta: 包含增量文本 (delta字段)
            # - response.output_text.done: 包含完整文本 (text字段)
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")

                        # 收集增量文本
                        if event_type == "response.output_text.delta":
                            content += data.get("delta", "")

                        # 或者直接使用完整文本
                        elif event_type == "response.output_text.done":
                            content = data.get("text", content)
                    except:
                        continue

        latency = time.time() - start_time

        return GenerationResponse(
            content=content,
            model=model,
            input_tokens=sum(len(msg["content"].split()) for msg in request.messages),
            output_tokens=len(content.split()),
            total_time=latency,
            ttft=latency
        )

    async def generate_stream(self, request: GenerationRequest, model: str = "gpt-5.2") -> AsyncGenerator[str, None]:
        """流式生成"""
        # 提取系统提示词和用户消息（OpenClaw 格式）
        system_prompt = ""
        user_messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                user_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg["content"]}]
                })

        # 构建请求（OpenClaw 格式）
        # Codex API 不支持 temperature 参数
        payload = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": user_messages,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        # KB-019: Pass through tools if provided
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        # 流式调用（OpenClaw 方式）
        async with self.client.stream(
            "POST",
            f"{self.base_url}/codex/responses",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "chatgpt-account-id": self.account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "pi",
                "User-Agent": "pi (darwin arm64)",
                "Accept": "text/event-stream",
                "Content-Type": "application/json"
            },
            json=payload
        ) as response:
            response.raise_for_status()

            # 流式输出：只发送增量文本
            # Codex API 事件流格式：response.output_text.delta 包含增量
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")

                        # 发送文本增量
                        if event_type == "response.output_text.delta":
                            delta = data.get("delta", "")
                            if delta:
                                yield delta
                    except:
                        continue

    def get_stats(self) -> Dict:
        """获取引擎统计信息"""
        return {
            "engine_type": "chatgpt_backend",
            "base_url": self.base_url,
            "authenticated": bool(self.access_token),
        }

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
