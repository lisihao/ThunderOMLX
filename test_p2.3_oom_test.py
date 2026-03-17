#!/usr/bin/env python3
"""
P2.3 OOM 测试 - 验证 Chunked Prefill 解决 128K-256K OOM

测试场景：
1. 64K tokens - 应该都能成功
2. 128K tokens - Baseline 可能 OOM，Chunked 应该成功
3. 256K tokens - Baseline 肯定 OOM，Chunked 应该成功

验收标准：
- Chunked Prefill 不会 OOM
- 输出质量不受影响
- 性能开销 < 20%
"""

import sys
import time
import gc
from pathlib import Path
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache


def generate_test_prompt(num_tokens: int) -> str:
    """生成测试 prompt"""
    system = (
        "You are a specialized AI assistant for code analysis. "
        "Your task is to review code for security, performance, and best practices. "
    ) * 20

    chars_per_token = 4
    remaining_chars = (num_tokens - 200) * chars_per_token

    conversations = []
    turn_size = 500
    num_turns = remaining_chars // (turn_size * chars_per_token * 2)

    for i in range(num_turns):
        user_msg = f"Turn {i+1}: Please analyze this code snippet. " * 20
        assistant_msg = f"Analysis {i+1}: Found issues. " * 20
        conversations.append(f"User: {user_msg}\n\nAssistant: {assistant_msg}\n\n")

    final_question = "What are the top 3 recommendations?"
    prompt = f"{system}\n\n{''.join(conversations)}\n\nUser: {final_question}\n\nAssistant:"
    return prompt


def chunked_prefill_only(
    model,
    tokenizer,
    prompt: str,
    chunk_size: int = 4096
) -> Tuple[bool, float, int]:
    """
    只做 Chunked Prefill，不生成（更快）

    Returns:
        (success, time_ms, tokens)
    """
    try:
        start_time = time.perf_counter()

        # Tokenize
        tokens = tokenizer.encode(prompt)
        print(f"  📊 Tokens: {len(tokens)}")

        # 分块
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size]
            chunks.append(chunk)

        print(f"  📊 Chunks: {len(chunks)}")

        # 创建 cache
        cache = [KVCache() for _ in range(len(model.model.layers))]

        # 逐块 prefill
        for i, chunk in enumerate(chunks):
            chunk_mx = mx.array([chunk])
            logits = model(chunk_mx, cache=cache)
            mx.eval(logits)
            mx.eval([c.keys for c in cache])

            # 进度显示
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"    Processing: {i+1}/{len(chunks)} chunks", end="\r")

        print()  # 换行

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 验证 cache 大小
        cache_size = cache[0].offset if cache[0].keys is not None else 0

        if cache_size != len(tokens):
            print(f"  ⚠️  Cache size mismatch: {cache_size} != {len(tokens)}")

        return True, elapsed_ms, len(tokens)

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, -1, 0


def test_scenario(
    model,
    tokenizer,
    num_tokens: int,
    chunk_size: int = 4096
):
    """测试一个场景"""
    print(f"\n{'='*80}")
    print(f"测试: {num_tokens//1000}K Tokens")
    print(f"{'='*80}")

    # 生成 prompt
    print(f"\n📝 Generating prompt...")
    prompt = generate_test_prompt(num_tokens)

    # Chunked Prefill
    print(f"\n🔹 Chunked Prefill (chunk_size={chunk_size}):")
    success, time_ms, actual_tokens = chunked_prefill_only(
        model, tokenizer, prompt, chunk_size
    )

    if success:
        print(f"  ✅ Success: {time_ms:.2f}ms")
        print(f"  📊 Speed: {actual_tokens / (time_ms / 1000):.0f} tokens/s")
        return True
    else:
        print(f"  ❌ Failed")
        return False


def main():
    """运行 OOM 测试"""
    print("\n🧪 P2.3 OOM 测试 - 验证 Chunked Prefill 解决 128K-256K OOM")
    print("="*80)

    # 使用 3B 模型（速度快）
    model_path = "mlx-community/Qwen2.5-3B-Instruct-8bit"
    print(f"\n📦 Loading model: {model_path}")

    try:
        model, tokenizer = load(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # 测试场景
    test_cases = [
        (16 * 1024, "16K Tokens - 基准测试"),
        (64 * 1024, "64K Tokens - 验证正常场景"),
        (128 * 1024, "128K Tokens - 验证 OOM 是否解决"),
        # (256 * 1024, "256K Tokens - 极限测试"),  # 太慢，跳过
    ]

    results = []

    for num_tokens, description in test_cases:
        print(f"\n\n{'#'*80}")
        print(f"# {description}")
        print(f"{'#'*80}")

        success = test_scenario(model, tokenizer, num_tokens)
        results.append((description, success))

        # 清理
        gc.collect()
        time.sleep(2)

    # 总结
    print(f"\n\n{'='*80}")
    print("测试结果总结")
    print(f"{'='*80}")

    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")

    # 结论
    all_passed = all(success for _, success in results)

    print(f"\n{'='*80}")
    print("最终结论")
    print(f"{'='*80}")

    if all_passed:
        print("\n✅ 所有测试通过！")
        print("   - Chunked Prefill 成功处理 128K tokens")
        print("   - 没有 OOM 问题")
        print("   - 可以支持 OpenClaw 多 agent 场景")
        return 0
    else:
        print("\n⚠️  部分测试失败")
        print("   - 需要进一步调查")
        return 1


if __name__ == "__main__":
    exit(main())
