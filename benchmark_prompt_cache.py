#!/usr/bin/env python3
"""
Prompt Cache Benchmark

Test Generation TPS improvement with prompt cache reuse (针对 OpenClaw 场景)
"""

import argparse
import json
import random
import time
from pathlib import Path
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache


def load_workload_samples(workload_file: Path, num_samples: int = 20):
    """Load workload samples"""
    samples = []
    with open(workload_file) as f:
        for line in f:
            samples.append(json.loads(line))

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples


def benchmark_with_cache(
    model_path: str,
    workload_file: Path,
    num_samples: int = 20,
    gen_length: int = 256,
):
    """
    Benchmark with prompt cache reuse
    """

    print("=" * 80)
    print("🚀 Prompt Cache Benchmark")
    print("=" * 80)

    # Load samples
    print(f"\n⏳ Loading workload samples...")
    samples = load_workload_samples(workload_file, num_samples)
    print(f"✅ Loaded {len(samples)} samples")

    # Analyze cache hit potential
    agent_counts = {}
    for s in samples:
        agent = s['agent_id']
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    print(f"\n📊 Agent Distribution:")
    for agent, count in sorted(agent_counts.items()):
        print(f"   {agent}: {count} samples")

    # Load model
    print(f"\n⏳ Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded")

    # Note: prompt_cache will be created per agent type below
    print(f"\n✅ Model ready for cache testing")

    # Warmup
    print("\n🔥 Warmup (1 run)...")
    test_prompt = "This is a test. " * 100
    _ = "".join([r.text for r in stream_generate(model, tokenizer, test_prompt, max_tokens=32)])
    print("   ✓")

    # Baseline: without cache reuse (test ALL samples for fair comparison)
    print(f"\n📊 Baseline (without cache reuse)...")
    print("-" * 80)

    baseline_results = []
    baseline_start = time.perf_counter()

    for i, sample in enumerate(samples):  # Test ALL samples
        target_length = sample['total_prompt_length']
        base_text = "This is a comprehensive test prompt for benchmarking. "
        prompt = base_text * (target_length // len(base_text.split()) + 1)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
        prompt = tokenizer.decode(tokens)

        start_time = time.perf_counter()
        count = 0
        for r in stream_generate(model, tokenizer, prompt, max_tokens=gen_length):
            count += 1
        end_time = time.perf_counter()

        tps = count / (end_time - start_time)
        baseline_results.append(tps)

    baseline_end = time.perf_counter()
    baseline_avg_tps = sum(baseline_results) / len(baseline_results)
    baseline_total_time = baseline_end - baseline_start
    baseline_time_per_sample = baseline_total_time / len(baseline_results)

    print(f"Baseline Avg TPS: {baseline_avg_tps:.1f} tok/s")
    print(f"Baseline Total Time: {baseline_total_time:.3f}s ({len(baseline_results)} samples)")
    print(f"Baseline Per-Sample: {baseline_time_per_sample:.3f}s")

    # With cache reuse
    print(f"\n📊 With prompt cache reuse...")
    print("-" * 80)

    # Group samples by agent (simulating cache reuse)
    agent_groups = {}
    for sample in samples:
        agent = sample['agent_id']
        if agent not in agent_groups:
            agent_groups[agent] = []
        agent_groups[agent].append(sample)

    cache_results = []
    cache_start = time.perf_counter()

    for agent, agent_samples in sorted(agent_groups.items()):
        print(f"\n{agent} ({len(agent_samples)} samples):")

        # Create cache for this agent type
        agent_cache = make_prompt_cache(model)

        for i, sample in enumerate(agent_samples):
            target_length = sample['total_prompt_length']
            base_text = f"Agent {agent}: This is a test prompt for benchmarking. "
            prompt = base_text * (target_length // len(base_text.split()) + 1)
            tokens = tokenizer.encode(prompt)
            if len(tokens) > target_length:
                tokens = tokens[:target_length]
            prompt = tokenizer.decode(tokens)

            start_time = time.perf_counter()
            count = 0
            # Reuse cache across same agent type
            for r in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=gen_length,
                prompt_cache=agent_cache,
            ):
                count += 1
            end_time = time.perf_counter()

            tps = count / (end_time - start_time)
            cache_results.append(tps)

            if (i + 1) % 3 == 0 or i == len(agent_samples) - 1:
                print(f"  Sample {i+1}/{len(agent_samples)}: {tps:.1f} tok/s")

    cache_end = time.perf_counter()
    cache_avg_tps = sum(cache_results) / len(cache_results)
    cache_total_time = cache_end - cache_start
    cache_time_per_sample = cache_total_time / len(cache_results)

    print("\n" + "-" * 80)

    # Results
    print(f"\n📈 Results:")
    print(f"\n⭐ Without Cache:")
    print(f"   Avg TPS: {baseline_avg_tps:.1f} tok/s")
    print(f"   Total Time: {baseline_total_time:.3f}s ({len(baseline_results)} samples)")
    print(f"   Per-Sample Time: {baseline_time_per_sample:.3f}s")

    print(f"\n⭐ With Prompt Cache:")
    print(f"   Avg TPS: {cache_avg_tps:.1f} tok/s")
    print(f"   Total Time: {cache_total_time:.3f}s ({len(cache_results)} samples)")
    print(f"   Per-Sample Time: {cache_time_per_sample:.3f}s")

    # Calculate improvement (CORRECTED)
    tps_improvement = (cache_avg_tps - baseline_avg_tps) / baseline_avg_tps * 100
    time_improvement = (baseline_time_per_sample - cache_time_per_sample) / baseline_time_per_sample * 100

    print(f"\n📊 Improvement:")
    print(f"   TPS: {tps_improvement:+.1f}%")
    print(f"   Per-Sample Time: {time_improvement:+.1f}%")

    if tps_improvement > 0:
        print(f"\n✅ Prompt Cache provides {tps_improvement:.1f}% speedup!")
    elif tps_improvement > -5:
        print(f"\n⚠️  Minimal impact ({tps_improvement:.1f}%), cache overhead negates benefits")
    else:
        print(f"\n❌ Cache overhead makes it slower ({tps_improvement:.1f}%)")

    print("\n" + "=" * 80)

    return cache_avg_tps, baseline_avg_tps


def main():
    parser = argparse.ArgumentParser(description="Prompt Cache Benchmark")
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
        default=256,
        help="Generation length in tokens",
    )

    args = parser.parse_args()

    benchmark_with_cache(
        model_path=args.model,
        workload_file=args.workload,
        num_samples=args.num_samples,
        gen_length=args.gen_length,
    )


if __name__ == "__main__":
    main()
