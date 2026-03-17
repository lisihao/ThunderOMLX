#!/usr/bin/env python3
"""
OpenClaw Workload Realistic Benchmark

Test Generation TPS with realistic prompt lengths from actual agent usage
"""

import argparse
import json
import random
import time
from pathlib import Path
from mlx_lm import load, stream_generate


def load_workload_samples(workload_file: Path, agent_type: str = None, num_samples: int = 20):
    """Load workload samples from JSONL file"""
    samples = []
    with open(workload_file) as f:
        for line in f:
            data = json.loads(line)
            if agent_type is None or data['agent_id'] == agent_type:
                samples.append(data)

    # Random sample
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples


def benchmark_agent_workload(
    model_path: str,
    workload_file: Path,
    agent_type: str = None,
    num_samples: int = 20,
    gen_length: int = 256,
):
    """
    Benchmark with realistic OpenClaw workload
    """

    print("=" * 80)
    print("📊 OpenClaw Workload Benchmark")
    print("=" * 80)

    # Load samples
    print(f"\n⏳ Loading workload samples...")
    samples = load_workload_samples(workload_file, agent_type, num_samples)
    print(f"✅ Loaded {len(samples)} samples")

    if agent_type:
        print(f"   Agent Type: {agent_type}")
    else:
        print(f"   Agent Type: ALL")

    # Analyze samples
    prompt_lengths = [s['total_prompt_length'] for s in samples]
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
    min_prompt_length = min(prompt_lengths)
    max_prompt_length = max(prompt_lengths)

    print(f"\n📏 Prompt Length Stats:")
    print(f"   Average: {avg_prompt_length:.0f} tokens")
    print(f"   Min: {min_prompt_length} tokens")
    print(f"   Max: {max_prompt_length} tokens")

    # Load model
    print(f"\n⏳ Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded")

    # Warmup
    print("\n🔥 Warmup (1 run)...")
    test_prompt = "This is a test. " * 100
    _ = "".join([r.text for r in stream_generate(model, tokenizer, test_prompt, max_tokens=32)])
    print("   ✓")

    # Run benchmark
    print(f"\n📊 Running benchmark on {len(samples)} samples...")
    print("-" * 80)

    results = []
    for i, sample in enumerate(samples):
        # Generate prompt with target length
        target_length = sample['total_prompt_length']
        base_text = "This is a comprehensive test prompt for benchmarking realistic agent workloads. "
        prompt = base_text * (target_length // len(base_text.split()) + 1)
        tokens = tokenizer.encode(prompt)
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
        prompt = tokenizer.decode(tokens)
        actual_length = len(tokens)

        # Benchmark generation
        start_time = time.perf_counter()

        count = 0
        for r in stream_generate(model, tokenizer, prompt, max_tokens=gen_length):
            count += 1

        end_time = time.perf_counter()
        total_time = end_time - start_time
        generation_tps = count / total_time

        results.append({
            'sample_id': i + 1,
            'agent_id': sample['agent_id'],
            'prompt_length': actual_length,
            'generation_tokens': count,
            'total_time': total_time,
            'generation_tps': generation_tps,
        })

        if (i + 1) % 5 == 0 or i == len(samples) - 1:
            print(f"Sample {i+1}/{len(samples)}: {generation_tps:.1f} tok/s (prompt: {actual_length} tokens)")

    print("-" * 80)

    # Calculate statistics
    print(f"\n📈 Results:")

    avg_tps = sum(r['generation_tps'] for r in results) / len(results)
    std_tps = (sum((r['generation_tps'] - avg_tps) ** 2 for r in results) / len(results)) ** 0.5
    min_tps = min(r['generation_tps'] for r in results)
    max_tps = max(r['generation_tps'] for r in results)

    print(f"\n⭐ Generation TPS: {avg_tps:.1f} ± {std_tps:.1f} tok/s")
    print(f"   Min: {min_tps:.1f} tok/s")
    print(f"   Max: {max_tps:.1f} tok/s")
    print(f"   Samples: {len(results)}")

    # Per-agent breakdown if testing all agents
    if agent_type is None and len(results) > 0:
        print(f"\n📊 Per-Agent Breakdown:")
        agent_stats = {}
        for r in results:
            agent = r['agent_id']
            if agent not in agent_stats:
                agent_stats[agent] = []
            agent_stats[agent].append(r['generation_tps'])

        for agent, tps_list in sorted(agent_stats.items()):
            avg = sum(tps_list) / len(tps_list)
            print(f"   {agent}: {avg:.1f} tok/s ({len(tps_list)} samples)")

    print("\n" + "=" * 80)

    return avg_tps


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Workload Benchmark")
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
        "--agent",
        type=str,
        default=None,
        help="Agent type to test (None = all agents)",
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

    benchmark_agent_workload(
        model_path=args.model,
        workload_file=args.workload,
        agent_type=args.agent,
        num_samples=args.num_samples,
        gen_length=args.gen_length,
    )


if __name__ == "__main__":
    main()
