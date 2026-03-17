#!/usr/bin/env python3
"""
Speculative Decoding Benchmark

测试 Speculative Decoding 在不同 draft models 上的性能
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from mlx_lm import load, generate


def benchmark_generation(
    model_path: str,
    draft_model_path: Optional[str],
    prompt_length: int,
    gen_length: int,
    num_trials: int = 3,
    num_speculative_tokens: int = 4,
):
    """
    Benchmark generation with optional speculative decoding

    Args:
        model_path: Target model path
        draft_model_path: Draft model path (None = no speculative decoding)
        prompt_length: Prompt length in tokens
        gen_length: Generation length in tokens
        num_trials: Number of trials
        num_speculative_tokens: K (number of speculative tokens)
    """

    print("=" * 80)
    print("🚀 Speculative Decoding Benchmark")
    print("=" * 80)

    # Configuration
    draft_enabled = draft_model_path is not None
    print(f"\n📊 Configuration:")
    print(f"   Target Model: {model_path}")
    print(f"   Draft Model:  {draft_model_path if draft_enabled else 'None (Baseline)'}")
    print(f"   Speculative Tokens (K): {num_speculative_tokens if draft_enabled else 'N/A'}")
    print(f"   Prompt Length: {prompt_length} tokens")
    print(f"   Generation Length: {gen_length} tokens")
    print(f"   Trials: {num_trials}")

    # Load model
    print(f"\n⏳ Loading target model...")
    model, tokenizer = load(model_path)
    print("✅ Target model loaded")

    # Load draft model if enabled
    draft_model = None
    if draft_enabled:
        print(f"\n⏳ Loading draft model from {draft_model_path}...")
        draft_model, _ = load(draft_model_path)
        print("✅ Draft model loaded")

    # Generate prompt
    print(f"\n⏳ Generating {prompt_length} token prompt...")
    base_text = "This is a comprehensive test prompt for benchmarking. "
    prompt = base_text * (prompt_length // len(base_text.split()) + 1)
    tokens = tokenizer.encode(prompt)
    if len(tokens) > prompt_length:
        tokens = tokens[:prompt_length]
    prompt = tokenizer.decode(tokens)
    print(f"✅ Prompt: {len(tokens)} tokens")

    # Warmup
    print("\n🔥 Warmup (2 runs)...")
    for i in range(2):
        _ = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=32,
            draft_model=draft_model if draft_enabled else None,
            num_draft_tokens=num_speculative_tokens if draft_enabled else None,
            verbose=False,
        )
        print(f"   Warmup {i+1}/2 ✓")

    # Run benchmark
    print(f"\n📊 Running {num_trials} trials...")
    print("-" * 80)

    results = []
    for trial in range(num_trials):
        start_time = time.perf_counter()

        response = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=gen_length,
            draft_model=draft_model if draft_enabled else None,
            num_draft_tokens=num_speculative_tokens if draft_enabled else None,
            verbose=False,
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate tokens (approximate)
        output_tokens = len(tokenizer.encode(response))
        generation_tps = output_tokens / total_time

        results.append({
            'trial': trial + 1,
            'total_time': total_time,
            'output_tokens': output_tokens,
            'generation_tps': generation_tps,
        })

        print(f"Trial {trial+1}/{num_trials}:")
        print(f"  Time: {total_time:.3f}s")
        print(f"  Tokens: {output_tokens}")
        print(f"  Generation TPS: {generation_tps:.1f} tok/s")

    # Calculate statistics
    print("\n" + "=" * 80)
    print("📈 Results")
    print("=" * 80)

    avg_tps = sum(r['generation_tps'] for r in results) / len(results)
    std_tps = (sum((r['generation_tps'] - avg_tps) ** 2 for r in results) / len(results)) ** 0.5

    print(f"\n⭐ Generation TPS: {avg_tps:.1f} ± {std_tps:.1f} tok/s")

    if draft_enabled:
        print(f"   Draft Model: {Path(draft_model_path).name}")
        print(f"   Speculative Tokens: {num_speculative_tokens}")

    print("\n" + "=" * 80)

    # Save results
    output_file = Path("/tmp/benchmark_speculative_decoding_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'target_model': model_path,
                'draft_model': draft_model_path,
                'num_speculative_tokens': num_speculative_tokens if draft_enabled else None,
                'prompt_length': prompt_length,
                'gen_length': gen_length,
                'num_trials': num_trials,
            },
            'results': {
                'avg_generation_tps': avg_tps,
                'std_generation_tps': std_tps,
                'trials': results,
            },
        }, f, indent=2)

    print(f"💾 Results saved to: {output_file}")

    return avg_tps


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/lisihao/models/qwen3.5-35b-mlx",
        help="Target model path",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Draft model path (None = baseline without SD)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=4096,
        help="Prompt length in tokens",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=512,
        help="Generation length in tokens",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=4,
        help="K (number of speculative tokens)",
    )

    args = parser.parse_args()

    benchmark_generation(
        model_path=args.model,
        draft_model_path=args.draft_model,
        prompt_length=args.prompt_length,
        gen_length=args.gen_length,
        num_trials=args.trials,
        num_speculative_tokens=args.num_speculative_tokens,
    )


if __name__ == "__main__":
    main()
