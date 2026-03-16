#!/usr/bin/env python3
"""
Test with real prompts to verify model output quality.
"""

import requests
import time

def test_real_prompt(port=8000):
    """Test with meaningful prompt."""

    # 真实场景的 prompt
    prompts = [
        {
            "name": "短提示 - 代码生成",
            "text": "用Python写一个函数，计算斐波那契数列的第n项："
        },
        {
            "name": "中等提示 - 问答",
            "text": "请解释什么是机器学习中的过拟合（overfitting），并给出3个避免过拟合的方法。" * 50  # ~1000 tokens
        },
        {
            "name": "长提示 - 摘要",
            "text": "以下是一篇关于人工智能发展的文章，请总结其主要观点：\n\n" + "人工智能技术近年来取得了突破性进展。深度学习、强化学习、大语言模型等技术不断涌现。这些技术在图像识别、自然语言处理、游戏博弈等领域展现出超越人类的能力。然而，人工智能也面临着诸多挑战，包括数据隐私、算法偏见、可解释性等问题。" * 100  # ~2000 tokens
        }
    ]

    url = f"http://127.0.0.1:{port}/v1/completions"

    print("🧪 真实 Prompt 测试")
    print("="*60)

    for prompt_info in prompts:
        print(f"\n📝 {prompt_info['name']}")
        print(f"   提示长度: {len(prompt_info['text'])} 字符")

        data = {
            "model": "Qwen3.5-35B-A3B-6bit",
            "prompt": prompt_info['text'],
            "max_tokens": 100,
            "temperature": 0.7
        }

        start = time.time()
        try:
            response = requests.post(url, json=data, timeout=60)
            latency = time.time() - start

            if response.status_code == 200:
                result = response.json()
                text = result.get('choices', [{}])[0].get('text', '')
                print(f"   ✅ 延迟: {latency:.3f}s")
                print(f"   📄 生成:\n{text[:200]}{'...' if len(text) > 200 else ''}")
            else:
                print(f"   ❌ HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ 错误: {e}")

        time.sleep(1)

if __name__ == "__main__":
    test_real_prompt()
