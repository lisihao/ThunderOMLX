#!/usr/bin/env python3
"""
Prefix Caching TTFT Benchmark

测试 Prefix Caching 对 TTFT 的优化效果
"""

import argparse
import json
import random
import time
from pathlib import Path
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache


def benchmark_ttft_with_prefix_cache(
    model_path: str,
    workload_file: Path,
    num_samples: int = 20,
    gen_length: int = 32,  # 只生成少量 token，专注测 TTFT
):
    """
    Benchmark TTFT with and without prefix caching
    """

    print("=" * 80)
    print("🚀 Prefix Caching TTFT Benchmark")
    print("=" * 80)

    # Load samples
    print(f"\n⏳ Loading workload samples...")
    samples = []
    with open(workload_file) as f:
        for line in f:
            samples.append(json.loads(line))

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    print(f"✅ Loaded {len(samples)} samples")

    # Group by agent type
    agent_groups = {}
    for sample in samples:
        agent = sample['agent_id']
        if agent not in agent_groups:
            agent_groups[agent] = []
        agent_groups[agent].append(sample)

    print(f"\n📊 Agent Distribution:")
    for agent, agent_samples in sorted(agent_groups.items()):
        print(f"   {agent}: {len(agent_samples)} samples")

    # Load model
    print(f"\n⏳ Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded")

    # Warmup
    print("\n🔥 Warmup...")
    test_prompt = "This is a test. " * 100
    _ = "".join([r.text for r in stream_generate(model, tokenizer, test_prompt, max_tokens=16)])
    print("   ✓")

    # Test 1: Cold start (no cache)
    print(f"\n📊 Test 1: Cold Start (no cache)")
    print("-" * 80)

    cold_ttfts = []
    for agent, agent_samples in sorted(agent_groups.items()):
        print(f"\n{agent}:")

        for i, sample in enumerate(agent_samples[:3]):  # Test 3 samples per agent
            target_length = sample['total_prompt_length']
            prompt = f"Agent {agent} system prompt. " * (target_length // 5)
            tokens = tokenizer.encode(prompt)[:target_length]
            prompt = tokenizer.decode(tokens)

            # Measure TTFT (time to first token)
            start_time = time.perf_counter()

            first_token_time = None
            for idx, r in enumerate(stream_generate(model, tokenizer, prompt, max_tokens=gen_length)):
                if idx == 0:
                    first_token_time = time.perf_counter()
                    break  # 只测 TTFT

            ttft = first_token_time - start_time
            cold_ttfts.append(ttft)
            print(f"  Sample {i+1}: TTFT = {ttft*1000:.1f}ms")

    cold_avg_ttft = sum(cold_ttfts) / len(cold_ttfts)
    print(f"\n⭐ Cold Start Avg TTFT: {cold_avg_ttft*1000:.1f}ms")

    # Test 2: Warm cache (with prefix caching)
    print(f"\n📊 Test 2: Warm Cache (with prefix caching)")
    print("-" * 80)

    warm_ttfts = []
    for agent, agent_samples in sorted(agent_groups.items()):
        print(f"\n{agent}:")

        # Create and warm up cache for this agent
        agent_cache = make_prompt_cache(model)

        # First request: populate cache (skip TTFT measurement)
        sample = agent_samples[0]
        target_length = sample['total_prompt_length']
        prompt = f"Agent {agent} system prompt. " * (target_length // 5)
        tokens = tokenizer.encode(prompt)[:target_length]
        prompt = tokenizer.decode(tokens)

        # Warm up cache
        for r in stream_generate(model, tokenizer, prompt, max_tokens=4, prompt_cache=agent_cache):
            pass

        print(f"  ✓ Cache warmed up")

        # Subsequent requests: measure TTFT with warm cache
        for i, sample in enumerate(agent_samples[:3]):
            target_length = sample['total_prompt_length']
            prompt = f"Agent {agent} system prompt. " * (target_length // 5)
            tokens = tokenizer.encode(prompt)[:target_length]
            prompt = tokenizer.decode(tokens)

            # Measure TTFT with warm cache
            start_time = time.perf_counter()

            first_token_time = None
            for idx, r in enumerate(stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=gen_length,
                prompt_cache=agent_cache  # Reuse cache
            )):
                if idx == 0:
                    first_token_time = time.perf_counter()
                    break

            ttft = first_token_time - start_time
            warm_ttfts.append(ttft)
            print(f"  Sample {i+1}: TTFT = {ttft*1000:.1f}ms")

    warm_avg_ttft = sum(warm_ttfts) / len(warm_ttfts)
    print(f"\n⭐ Warm Cache Avg TTFT: {warm_avg_ttft*1000:.1f}ms")

    # Results
    print("\n" + "=" * 80)
    print("📈 Results Summary")
    print("=" * 80)

    print(f"\n⭐ Cold Start (no cache):")
    print(f"   Avg TTFT: {cold_avg_ttft*1000:.1f}ms")
    print(f"   Samples: {len(cold_ttfts)}")

    print(f"\n⭐ Warm Cache (with prefix caching):")
    print(f"   Avg TTFT: {warm_avg_ttft*1000:.1f}ms")
    print(f"   Samples: {len(warm_ttfts)}")

    improvement = (cold_avg_ttft - warm_avg_ttft) / cold_avg_ttft * 100

    print(f"\n📊 Improvement:")
    print(f"   TTFT Reduction: {improvement:.1f}%")
    print(f"   Absolute: {(cold_avg_ttft - warm_avg_ttft)*1000:.1f}ms faster")

    if improvement > 30:
        print(f"\n✅ Prefix Caching provides {improvement:.1f}% TTFT improvement! ⭐")
    elif improvement > 10:
        print(f"\n✅ Moderate improvement ({improvement:.1f}%)")
    else:
        print(f"\n⚠️  Limited improvement ({improvement:.1f}%)")

    print("\n" + "=" * 80)

    return cold_avg_ttft, warm_avg_ttft


def main():
    parser = argparse.ArgumentParser(description="Prefix Caching TTFT Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/lisihao/models/qwen3.5-35b-mlx",
        help="Model path",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("/Users/lisihao/ThunderOMLX/openclaw-workload/openclaw-workload-7d.jsonl"),
        help="Workload JSONL file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=32,
        help="Generation length (small value to focus on TTFT)",
    )

    args = parser.parse_args()

    benchmark_ttft_with_prefix_cache(
        model_path=args.model,
        workload_file=args.workload,
        num_samples=args.num_samples,
        gen_length=args.gen_length,
    )


if __name__ == "__main__":
    main()
