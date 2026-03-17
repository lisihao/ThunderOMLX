#!/usr/bin/env python3
"""
Batch Decode Benchmark

Test Generation TPS improvement with multiple concurrent requests
"""

import argparse
import time
import threading
from pathlib import Path
from queue import Queue
from mlx_lm import load, stream_generate


def generate_worker(
    model,
    tokenizer,
    prompt,
    max_tokens,
    result_queue,
    worker_id,
):
    """Worker thread for generating tokens"""
    start_time = time.perf_counter()

    count = 0
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        count += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time
    tps = count / total_time

    result_queue.put({
        'worker_id': worker_id,
        'tokens': count,
        'time': total_time,
        'tps': tps,
    })


def benchmark_batch_decode(
    model_path: str,
    prompt_length: int = 800,
    gen_length: int = 256,
    num_concurrent: int = 4,
    num_trials: int = 3,
):
    """
    Benchmark with multiple concurrent requests
    """

    print("=" * 80)
    print("🚀 Batch Decode Benchmark")
    print("=" * 80)

    print(f"\n📊 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Prompt Length: {prompt_length} tokens")
    print(f"   Generation Length: {gen_length} tokens")
    print(f"   Concurrent Requests: {num_concurrent}")
    print(f"   Trials: {num_trials}")

    # Load model
    print(f"\n⏳ Loading model...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded")

    # Generate prompt
    print(f"\n⏳ Generating {prompt_length} token prompt...")
    base_text = "This is a comprehensive test prompt for benchmarking batch decode. "
    prompt = base_text * (prompt_length // len(base_text.split()) + 1)
    tokens = tokenizer.encode(prompt)
    if len(tokens) > prompt_length:
        tokens = tokens[:prompt_length]
    prompt = tokenizer.decode(tokens)
    actual_length = len(tokens)
    print(f"✅ Prompt: {actual_length} tokens")

    # Warmup
    print("\n🔥 Warmup (1 run)...")
    _ = "".join([r.text for r in stream_generate(model, tokenizer, prompt, max_tokens=32)])
    print("   ✓")

    # Run benchmark
    print(f"\n📊 Running {num_trials} trials with {num_concurrent} concurrent requests...")
    print("-" * 80)

    trial_results = []
    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}:")

        result_queue = Queue()
        threads = []

        # Start all workers
        overall_start = time.perf_counter()

        for i in range(num_concurrent):
            thread = threading.Thread(
                target=generate_worker,
                args=(model, tokenizer, prompt, gen_length, result_queue, i + 1)
            )
            thread.start()
            threads.append(thread)

        # Wait for all workers
        for thread in threads:
            thread.join()

        overall_end = time.perf_counter()
        overall_time = overall_end - overall_start

        # Collect results
        worker_results = []
        while not result_queue.empty():
            worker_results.append(result_queue.get())

        # Calculate metrics
        total_tokens = sum(r['tokens'] for r in worker_results)
        avg_worker_tps = sum(r['tps'] for r in worker_results) / len(worker_results)
        aggregate_tps = total_tokens / overall_time

        trial_results.append({
            'trial': trial + 1,
            'overall_time': overall_time,
            'total_tokens': total_tokens,
            'avg_worker_tps': avg_worker_tps,
            'aggregate_tps': aggregate_tps,
        })

        print(f"  Overall Time: {overall_time:.3f}s")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Avg Worker TPS: {avg_worker_tps:.1f} tok/s")
        print(f"  Aggregate TPS: {aggregate_tps:.1f} tok/s")

    print("\n" + "-" * 80)

    # Calculate statistics
    print(f"\n📈 Results Summary:")

    avg_aggregate_tps = sum(r['aggregate_tps'] for r in trial_results) / len(trial_results)
    std_aggregate_tps = (sum((r['aggregate_tps'] - avg_aggregate_tps) ** 2 for r in trial_results) / len(trial_results)) ** 0.5

    avg_worker_tps = sum(r['avg_worker_tps'] for r in trial_results) / len(trial_results)

    print(f"\n⭐ Aggregate TPS: {avg_aggregate_tps:.1f} ± {std_aggregate_tps:.1f} tok/s")
    print(f"   Concurrent Requests: {num_concurrent}")
    print(f"   Avg Worker TPS: {avg_worker_tps:.1f} tok/s")

    # Compare with single-request baseline
    print(f"\n📊 Comparison:")
    print(f"   Baseline (single request): ~65 tok/s (OpenClaw)")
    print(f"   Batch Decode ({num_concurrent} concurrent): {avg_aggregate_tps:.1f} tok/s")
    if avg_aggregate_tps > 65:
        improvement = (avg_aggregate_tps - 65) / 65 * 100
        print(f"   Improvement: +{improvement:.1f}%")
    else:
        degradation = (65 - avg_aggregate_tps) / 65 * 100
        print(f"   Degradation: -{degradation:.1f}%")

    print("\n" + "=" * 80)

    return avg_aggregate_tps


def main():
    parser = argparse.ArgumentParser(description="Batch Decode Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/lisihao/models/qwen3.5-35b-mlx",
        help="Model path",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=800,
        help="Prompt length in tokens",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=256,
        help="Generation length in tokens",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials",
    )

    args = parser.parse_args()

    benchmark_batch_decode(
        model_path=args.model,
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        num_concurrent=args.concurrent,
        num_trials=args.trials,
    )


if __name__ == "__main__":
    main()
