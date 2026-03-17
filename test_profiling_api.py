#!/usr/bin/env python3
"""
测试 Profiling 功能（通过 API）

运行前先启动服务器：
    export OMLX_ENABLE_PROFILING=true
    python -m omlx.server
"""

import asyncio
import json
from openai import AsyncOpenAI


async def test_profiling():
    """测试带 profiling 的 Prefill"""

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy"  # oMLX doesn't require real API key
    )

    print("=" * 80)
    print("测试 Prefill Profiling")
    print("=" * 80)

    # 生成一个长 prompt（~8K tokens）
    long_prompt = """
    Please provide a detailed analysis of the following topics:
    1. The evolution of machine learning from traditional algorithms to deep learning
    2. Key breakthroughs in natural language processing over the past decade
    3. The architecture and training process of large language models
    4. Applications of AI in various industries
    5. Ethical considerations and challenges in AI development
    6. Future trends in artificial intelligence research

    For each topic, please include:
    - Historical background and context
    - Technical details and methodologies
    - Real-world applications and case studies
    - Current challenges and limitations
    - Future directions and potential breakthroughs
    """ * 100  # 重复以达到 ~8K tokens

    print(f"Prompt length: ~{len(long_prompt)} chars\n")
    print("发送请求...")

    # 发送请求（只生成1个token，主要测试Prefill）
    response = await client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": long_prompt}],
        max_tokens=1,
        temperature=0.0
    )

    print("✅ 请求完成")
    print(f"生成: {response.choices[0].message.content}\n")

    # 获取 profiling 统计（如果有 admin API）
    print("=" * 80)
    print("Profiling 数据（需要在服务器日志中查看）")
    print("=" * 80)
    print("查看服务器日志以获取详细的 profiling 统计")


if __name__ == "__main__":
    asyncio.run(test_profiling())
