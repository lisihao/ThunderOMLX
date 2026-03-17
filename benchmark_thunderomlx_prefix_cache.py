#!/usr/bin/env python3
"""
ThunderOMLX Prefix Caching Benchmark

正确的测试方式：使用 ThunderOMLX 的 BlockAwarePrefixCache + ContextPilotAdapter

与之前错误的测试对比：
- ❌ 之前：使用 mlx-lm 的 make_prompt_cache(model) → 结果：-9.6% (变慢)
- ✅ 现在：使用 ThunderOMLX 的 BlockAwarePrefixCache + PagedCacheManager → 期望：-50% ~ -80%
"""

import time
from pathlib import Path
from mlx_lm import load, stream_generate

# ThunderOMLX 的缓存组件
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.contextpilot import ContextPilotAdapter, ContextIndex


def benchmark_thunderomlx_cache(
    model_path: str,
    system_prompt_length: int = 800,
    num_queries: int = 10,
):
    """
    测试 ThunderOMLX 的 Prefix Caching 系统。

    架构：
    - PagedCacheManager: 块级内存管理 (256 tokens/块)
    - BlockAwarePrefixCache: 前缀去重 (hash-based)
    - ContextPilotAdapter: 请求重排序 (system prompt 前置)
    """

    print("=" * 80)
    print("🚀 ThunderOMLX Prefix Caching Benchmark")
    print("=" * 80)

    print(f"\n📊 Configuration:")
    print(f"   System Prompt Length: {system_prompt_length} tokens")
    print(f"   Num Queries: {num_queries}")

    # 1. Load model
    print(f"\n⏳ Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded")

    # 2. Create ThunderOMLX caching components
    print("\n⏳ Initializing ThunderOMLX cache components...")

    # PagedCacheManager (块级内存管理)
    paged_cache = PagedCacheManager(
        block_size=256,  # ThunderLLAMA 的标准块大小
        max_blocks=1000,
        model_name="qwen3.5-35b",
        initial_blocks=100,
    )
    print(f"  ✓ PagedCacheManager: block_size={paged_cache.block_size}, max_blocks={paged_cache.max_blocks}")

    # BlockAwarePrefixCache (前缀去重)
    prefix_cache = BlockAwarePrefixCache(
        model=model,
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=None,  # Phase 1: 不启用 SSD
    )
    print(f"  ✓ BlockAwarePrefixCache: initialized")

    # ContextPilotAdapter (请求优化)
    context_index = ContextIndex()
    context_pilot = ContextPilotAdapter(
        context_index=context_index,
        tokenizer=tokenizer,
    )
    print(f"  ✓ ContextPilotAdapter: initialized")

    print("✅ All cache components ready")

    # 3. Create fixed system prompt (800 tokens)
    system_prompt = (
        "You are a helpful AI assistant. Your role is to provide accurate, "
        "detailed, and well-structured responses to user queries. "
        "Always be polite, professional, and clear in your communication. "
    ) * 50  # Repeat to reach ~800 tokens

    tokens = tokenizer.encode(system_prompt)
    if len(tokens) > system_prompt_length:
        tokens = tokens[:system_prompt_length]
    system_prompt = tokenizer.decode(tokens)
    system_prompt_tokens = len(tokens)

    print(f"\n✅ System Prompt: {system_prompt_tokens} tokens")

    # 4. Generate different user queries (OpenClaw 场景)
    user_queries = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "Explain the theory of relativity.",
        "What is machine learning?",
        "How do vaccines work?",
        "What causes climate change?",
        "Explain blockchain technology.",
    ]

    # 5. Warmup
    print("\n🔥 Warmup...")
    test_prompt = system_prompt + "\\n\\nUser: Test query\\nAssistant:"
    _ = "".join([r.text for r in stream_generate(model, tokenizer, test_prompt, max_tokens=8)])
    print("   ✓")

    # 6. Test 1: Cold start (第一次请求，cache miss)
    print(f"\n📊 Test 1: Cold Start (no cache)")
    print("-" * 80)

    cold_ttfts = []
    for i, query in enumerate(user_queries[:3]):  # 只测试前 3 个
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # ContextPilot 优化请求
        optimized = context_pilot.optimize_request(messages, previous_requests=[])

        # 构建 prompt
        full_prompt = system_prompt + f"\\n\\nUser: {query}\\nAssistant:"
        prompt_tokens = tokenizer.encode(full_prompt)

        # 检查 cache (第一次应该 miss)
        block_table, remaining_tokens = prefix_cache.fetch_cache(f"req-cold-{i}", prompt_tokens)

        if block_table is not None:
            print(f"  Query {i+1}: Cache hit! (unexpected)")

        # 测量 TTFT
        start_time = time.perf_counter()
        first_token_time = None
        for idx, r in enumerate(stream_generate(model, tokenizer, full_prompt, max_tokens=16)):
            if idx == 0:
                first_token_time = time.perf_counter()
                break

        ttft = first_token_time - start_time
        cold_ttfts.append(ttft)
        print(f"  Query {i+1}: TTFT = {ttft*1000:.1f}ms (cache miss)")

        # 存入 cache (模拟生成后的 cache)
        # 注意：这里简化了，实际 Scheduler 会在 generation 过程中自动存储
        # 我们这里手动模拟 cache 写入

    cold_avg_ttft = sum(cold_ttfts) / len(cold_ttfts)
    print(f"\n⭐ Cold Start Avg TTFT: {cold_avg_ttft*1000:.1f}ms")

    # 7. Test 2: Warm cache (相同 system prompt, 不同 user query)
    print(f"\n📊 Test 2: Warm Cache (with BlockAwarePrefixCache)")
    print("-" * 80)

    # 注意：由于我们没有实际运行 generation 来填充 cache，
    # 这里需要手动模拟 cache 存储
    # 实际使用中，Scheduler 会在 generation 过程中自动调用 store_cache

    print("  ⚠️  注意：这个 benchmark 简化了 cache 写入流程")
    print("  ⚠️  完整测试需要使用 ThunderOMLX Server API")

    print(f"\n📈 Expected Results (基于 ThunderLLAMA 经验):")
    print(f"   Cold Start: ~1000ms (full prefill)")
    print(f"   Warm Cache: ~200ms (只处理 user query)")
    print(f"   Expected TTFT Reduction: -80% ⭐")

    print("\n" + "=" * 80)
    print("📝 结论")
    print("=" * 80)

    print("\n⚠️  这个 benchmark 无法完全测试 ThunderOMLX 的缓存效果")
    print("⚠️  因为 BlockAwarePrefixCache 需要与 generation 流程集成")

    print("\n✅ 正确的测试方式：")
    print("   1. 启动 ThunderOMLX 服务器")
    print("      python -m omlx.server --model qwen3.5-35b-mlx --enable-prefix-cache")
    print("")
    print("   2. 通过 HTTP API 测试")
    print("      curl -X POST http://localhost:8080/v1/chat/completions ...")
    print("")
    print("   3. 多次请求相同 system prompt，观察 TTFT 变化")

    print("\n" + "=" * 80)

    return cold_avg_ttft, 0  # warm cache 未测试


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ThunderOMLX Prefix Caching Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/lisihao/models/qwen3.5-35b-mlx",
        help="Model path",
    )
    parser.add_argument(
        "--system-prompt-length",
        type=int,
        default=800,
        help="System prompt length in tokens",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of user queries to test",
    )

    args = parser.parse_args()

    benchmark_thunderomlx_cache(
        model_path=args.model,
        system_prompt_length=args.system_prompt_length,
        num_queries=args.num_queries,
    )


if __name__ == "__main__":
    main()
