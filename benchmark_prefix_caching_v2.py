#!/usr/bin/env python3
"""
Prefix Caching TTFT Benchmark V2

正确的测试方法：固定 system prompt，只改变 user query
"""

import argparse
import time
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache


def benchmark_ttft_correct(
    model_path: str,
    system_prompt_length: int = 800,
    num_queries: int = 10,
):
    """
    正确测试 Prefix Caching：
    - 固定的 system prompt (可缓存)
    - 不同的 user query (不可缓存)
    """

    print("=" * 80)
    print("🚀 Prefix Caching TTFT Benchmark V2 (Corrected)")
    print("=" * 80)

    print(f"\n📊 Configuration:")
    print(f"   System Prompt Length: {system_prompt_length} tokens")
    print(f"   Num Queries: {num_queries}")

    # Load model
    print(f"\n⏳ Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded")

    # Create fixed system prompt (800 tokens)
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

    print(f"✅ System Prompt: {system_prompt_tokens} tokens")

    # Generate different user queries
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

    # Warmup
    print("\n🔥 Warmup...")
    test_prompt = system_prompt + "\n\nUser: Test query\nAssistant:"
    _ = "".join([r.text for r in stream_generate(model, tokenizer, test_prompt, max_tokens=8)])
    print("   ✓")

    # Test 1: Cold start (每次都重新 prefill system prompt)
    print(f"\n📊 Test 1: Cold Start (no prefix caching)")
    print("-" * 80)

    cold_ttfts = []
    for i, query in enumerate(user_queries):
        full_prompt = system_prompt + f"\n\nUser: {query}\nAssistant:"

        start_time = time.perf_counter()
        first_token_time = None
        for idx, r in enumerate(stream_generate(model, tokenizer, full_prompt, max_tokens=16)):
            if idx == 0:
                first_token_time = time.perf_counter()
                break

        ttft = first_token_time - start_time
        cold_ttfts.append(ttft)
        print(f"  Query {i+1}: TTFT = {ttft*1000:.1f}ms")

    cold_avg_ttft = sum(cold_ttfts) / len(cold_ttfts)
    print(f"\n⭐ Cold Start Avg TTFT: {cold_avg_ttft*1000:.1f}ms")

    # Test 2: Warm cache (复用 system prompt cache)
    print(f"\n📊 Test 2: Warm Cache (with prefix caching)")
    print("-" * 80)

    # Create cache and warm it up with system prompt
    prompt_cache = make_prompt_cache(model)

    # First request: warm up cache with system prompt
    first_query = user_queries[0]
    first_prompt = system_prompt + f"\n\nUser: {first_query}\nAssistant:"

    print(f"  ✓ Warming cache with system prompt...")
    for r in stream_generate(model, tokenizer, first_prompt, max_tokens=8, prompt_cache=prompt_cache):
        pass

    print(f"  ✓ Cache ready")

    # Subsequent requests: should reuse system prompt cache
    warm_ttfts = []
    for i, query in enumerate(user_queries):
        full_prompt = system_prompt + f"\n\nUser: {query}\nAssistant:"

        start_time = time.perf_counter()
        first_token_time = None
        for idx, r in enumerate(stream_generate(
            model,
            tokenizer,
            full_prompt,
            max_tokens=16,
            prompt_cache=prompt_cache  # Reuse cache
        )):
            if idx == 0:
                first_token_time = time.perf_counter()
                break

        ttft = first_token_time - start_time
        warm_ttfts.append(ttft)
        print(f"  Query {i+1}: TTFT = {ttft*1000:.1f}ms")

    warm_avg_ttft = sum(warm_ttfts) / len(warm_ttfts)
    print(f"\n⭐ Warm Cache Avg TTFT: {warm_avg_ttft*1000:.1f}ms")

    # Results
    print("\n" + "=" * 80)
    print("📈 Results Summary")
    print("=" * 80)

    print(f"\n⭐ Cold Start (prefill system prompt every time):")
    print(f"   Avg TTFT: {cold_avg_ttft*1000:.1f}ms")
    print(f"   System Prompt: {system_prompt_tokens} tokens")

    print(f"\n⭐ Warm Cache (reuse system prompt cache):")
    print(f"   Avg TTFT: {warm_avg_ttft*1000:.1f}ms")
    print(f"   Cached: {system_prompt_tokens} tokens")

    improvement = (cold_avg_ttft - warm_avg_ttft) / cold_avg_ttft * 100
    time_saved = (cold_avg_ttft - warm_avg_ttft) * 1000

    print(f"\n📊 Improvement:")
    print(f"   TTFT Reduction: {improvement:.1f}%")
    print(f"   Time Saved: {time_saved:.1f}ms per request")

    if improvement > 50:
        print(f"\n✅ Excellent! Prefix Caching saves {improvement:.1f}% TTFT! ⭐⭐⭐")
    elif improvement > 30:
        print(f"\n✅ Good! Prefix Caching provides {improvement:.1f}% improvement! ⭐⭐")
    elif improvement > 10:
        print(f"\n✅ Moderate improvement ({improvement:.1f}%)")
    elif improvement > 0:
        print(f"\n⚠️  Limited improvement ({improvement:.1f}%)")
    else:
        print(f"\n❌ No improvement or regression ({improvement:.1f}%)")

    print("\n" + "=" * 80)

    return cold_avg_ttft, warm_avg_ttft


def main():
    parser = argparse.ArgumentParser(description="Prefix Caching TTFT Benchmark V2")
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

    benchmark_ttft_correct(
        model_path=args.model,
        system_prompt_length=args.system_prompt_length,
        num_queries=args.num_queries,
    )


if __name__ == "__main__":
    main()
