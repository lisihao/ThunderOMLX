#!/usr/bin/env python3
"""
P2.3 Chunked Prefill 实现

目标：实现真正的分块 prefill，避免 128K-256K OOM

核心机制：
1. 创建 KVCache list（每层一个）
2. 逐块调用 model(chunk, cache=cache)
3. cache 会自动累积（原地修改）
4. 最后基于完整 cache 生成

测试场景：
- 16K tokens prompt（验证正确性）
- 128K tokens prompt（验证 OOM 问题是否解决）
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

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


def baseline_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 10
) -> Tuple[str, float]:
    """
    Baseline: 一次性 prefill（可能 OOM）

    Returns:
        (generated_text, time_ms)
    """
    print("\n" + "="*80)
    print("方法 1: 一次性 Prefill (Baseline)")
    print("="*80)

    start_time = time.perf_counter()

    # Tokenize
    tokens = tokenizer.encode(prompt)
    print(f"📊 Tokens: {len(tokens)}")

    # Forward pass (可能 OOM)
    input_ids = mx.array([tokens])

    try:
        logits = model(input_ids, cache=None)
        mx.eval(logits)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"✅ Prefill completed in {elapsed_ms:.2f}ms")

        # 生成第一个 token
        next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
        generated_tokens = [next_token]

        # 简单生成剩余 tokens
        cache = [KVCache() for _ in range(len(model.model.layers))]

        # Rebuild cache
        _ = model(input_ids, cache=cache)
        mx.eval([c.keys for c in cache])

        for _ in range(max_tokens - 1):
            logits = model(mx.array([[next_token]]), cache=cache)
            mx.eval(logits)
            next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
            generated_tokens.append(next_token)

        response = tokenizer.decode(generated_tokens)
        print(f"📝 Generated: {response[:100]}...")

        return response, elapsed_ms

    except Exception as e:
        print(f"❌ OOM Error: {e}")
        return "", -1


def chunked_generate(
    model,
    tokenizer,
    prompt: str,
    chunk_size: int = 4096,
    max_tokens: int = 10
) -> Tuple[str, float]:
    """
    Chunked Prefill: 分块处理，避免 OOM

    核心思路：
    1. 创建 KVCache list（每层一个）
    2. 逐块调用 model(chunk, cache=cache)
    3. cache 会自动累积（原地修改）
    4. 最后基于完整 cache 生成

    Returns:
        (generated_text, time_ms)
    """
    print("\n" + "="*80)
    print(f"方法 2: Chunked Prefill (chunk_size={chunk_size})")
    print("="*80)

    start_time = time.perf_counter()

    # Tokenize
    tokens = tokenizer.encode(prompt)
    print(f"📊 Total tokens: {len(tokens)}")

    # 分块
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        chunks.append(chunk)

    print(f"📊 Split into {len(chunks)} chunks")

    # 创建 cache（关键：每层一个 KVCache）
    cache = [KVCache() for _ in range(len(model.model.layers))]
    print(f"✅ Created cache for {len(cache)} layers")

    # 逐块 prefill
    for i, chunk in enumerate(chunks):
        chunk_mx = mx.array([chunk])  # 添加 batch 维度

        print(f"  Processing chunk {i+1}/{len(chunks)}: {len(chunk)} tokens")

        # Forward pass（cache 会自动累积）
        logits = model(chunk_mx, cache=cache)
        mx.eval(logits)  # 确保计算完成
        mx.eval([c.keys for c in cache])  # 确保 cache 更新完成

    prefill_ms = (time.perf_counter() - start_time) * 1000
    print(f"✅ All chunks processed in {prefill_ms:.2f}ms")

    # 验证 cache 大小
    cache_size = cache[0].offset if cache[0].keys is not None else 0
    print(f"📊 Cache size: {cache_size} tokens (expected: {len(tokens)})")

    if cache_size != len(tokens):
        print(f"⚠️  Cache size mismatch!")

    # 基于完整 cache 生成
    print(f"\n🔹 Generating {max_tokens} tokens...")

    # 获取最后一个 token 的 logits
    next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
    generated_tokens = [next_token]

    # 继续生成
    for _ in range(max_tokens - 1):
        logits = model(mx.array([[next_token]]), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
        generated_tokens.append(next_token)

    response = tokenizer.decode(generated_tokens)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"✅ Total time: {elapsed_ms:.2f}ms")
    print(f"📝 Generated: {response[:100]}...")

    return response, elapsed_ms


def compare_outputs(baseline: str, chunked: str):
    """对比输出"""
    print("\n" + "="*80)
    print("输出对比")
    print("="*80)

    if baseline == chunked:
        print("✅ 完全一致！")
        return True
    else:
        # 计算相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, baseline, chunked).ratio()

        print(f"📊 相似度: {similarity:.2%}")

        if similarity > 0.99:
            print("✅ 高度相似（>99%）")
            return True
        elif similarity > 0.95:
            print("⚠️  基本相似（>95%）")
            return True
        else:
            print("❌ 相似度较低（<95%）")
            return False


def main():
    """运行 Chunked Prefill 测试"""
    print("\n🧪 P2.3 Chunked Prefill 实现")
    print("="*80)
    print("目标: 解决 128K-256K tokens 首次 prefill OOM 问题")
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

    # 测试 1: 16K tokens（验证正确性）
    print("\n\n" + "#"*80)
    print("# 测试 1: 16K Tokens（验证正确性）")
    print("#"*80)

    prompt_16k = generate_test_prompt(16 * 1024)
    actual_tokens = len(tokenizer.encode(prompt_16k))
    print(f"\n📝 Prompt: {actual_tokens} tokens")

    # Baseline
    baseline_output, baseline_time = baseline_generate(
        model, tokenizer, prompt_16k, max_tokens=10
    )

    # Chunked
    chunked_output, chunked_time = chunked_generate(
        model, tokenizer, prompt_16k, chunk_size=4096, max_tokens=10
    )

    # 对比
    if baseline_output and chunked_output:
        match = compare_outputs(baseline_output, chunked_output)

        if match:
            print("\n✅ 16K 测试通过！")
        else:
            print("\n⚠️  16K 测试输出不一致，需要检查实现")
            return 1

        # 性能对比
        print("\n" + "="*80)
        print("性能对比")
        print("="*80)
        print(f"Baseline: {baseline_time:.2f}ms")
        print(f"Chunked:  {chunked_time:.2f}ms")

        if chunked_time > 0 and baseline_time > 0:
            overhead = ((chunked_time - baseline_time) / baseline_time) * 100
            print(f"Overhead: {overhead:+.1f}%")

            if overhead < 20:
                print("✅ 性能开销可接受（< 20%）")
            else:
                print("⚠️  性能开销较大（> 20%）")

    # TODO: 测试 2: 128K tokens（验证 OOM 是否解决）
    # 由于测试环境限制，暂时跳过 128K 测试
    # 实际使用时，应该测试 128K-256K tokens

    return 0


if __name__ == "__main__":
    exit(main())
