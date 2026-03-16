#!/usr/bin/env python3
"""
Test Speculative Decoding with 1.7B draft model.

Target: Qwen3.5-35B-A3B-6bit (activated 3.5B)
Draft:  Qwen3-1.7B-MLX-4bit
Ratio:  1.7B / 3.5B ≈ 1:2 (much better than 0.8B / 3.5B ≈ 1:4)
"""

import time
from pathlib import Path
from mlx_lm import load, generate

def test_1_7b_speculative():
    """Test using 1.7B draft model."""

    print("🚀 Speculative Decoding with 1.7B Draft Model")
    print("=" * 70)
    print("\n📊 Model Configuration:")
    print("  Target: Qwen3.5-35B-A3B (activated 3.5B)")
    print("  Draft:  Qwen3-1.7B (1.7B)")
    print("  Ratio:  1:2 (1.7B / 3.5B activated)")
    print("=" * 70)

    # Model paths
    target_path = Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    draft_path = Path.home() / ".omlx" / "models" / "Qwen3-1.7B-MLX-4bit"

    # Load models
    print(f"\n📥 Loading target model...")
    target_model, target_tokenizer = load(str(target_path))
    print(f"✅ Target model loaded")

    print(f"\n📥 Loading draft model (1.7B)...")
    draft_model, draft_tokenizer = load(str(draft_path))
    print(f"✅ Draft model loaded")

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The capital of France is",
        "To be or not to be",
        "Explain quantum computing in simple terms:",
    ]

    max_tokens = 50
    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_prompts)}: \"{prompt}\"")
        print(f"{'='*70}")

        # Test with different K values
        for K in [2, 4, 6]:
            print(f"\n🎯 K={K} (draft tokens):")

            # Speculative decoding
            start_time = time.time()
            response_spec = generate(
                target_model,
                target_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                draft_model=draft_model,
                num_draft_tokens=K,
                verbose=False
            )
            spec_time = time.time() - start_time
            spec_tps = max_tokens / spec_time if spec_time > 0 else 0

            print(f"  Time: {spec_time:.2f}s | Throughput: {spec_tps:.1f} tok/s")
            print(f"  Generated: \"{response_spec[:80]}...\"")

        # Baseline
        print(f"\n🔄 Baseline (no draft):")
        start_time = time.time()
        response_baseline = generate(
            target_model,
            target_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        baseline_time = time.time() - start_time
        baseline_tps = max_tokens / baseline_time if baseline_time > 0 else 0

        print(f"  Time: {baseline_time:.2f}s | Throughput: {baseline_tps:.1f} tok/s")
        print(f"  Generated: \"{response_baseline[:80]}...\"")

        # Calculate speedup (use K=4 as reference)
        # Re-run K=4 for accurate comparison
        start_time = time.time()
        response_spec_k4 = generate(
            target_model,
            target_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_model=draft_model,
            num_draft_tokens=4,
            verbose=False
        )
        spec_time_k4 = time.time() - start_time
        speedup = baseline_time / spec_time_k4 if spec_time_k4 > 0 else 1.0

        print(f"\n📊 Speedup (K=4): {speedup:.2f}×")

        results.append({
            "prompt": prompt,
            "baseline_time": baseline_time,
            "spec_time_k4": spec_time_k4,
            "speedup": speedup,
            "baseline_tps": baseline_tps,
            "spec_tps_k4": max_tokens / spec_time_k4 if spec_time_k4 > 0 else 0
        })

    # Summary
    print(f"\n{'='*70}")
    print("📊 Summary")
    print(f"{'='*70}")

    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_baseline_tps = sum(r["baseline_tps"] for r in results) / len(results)
    avg_spec_tps = sum(r["spec_tps_k4"] for r in results) / len(results)

    print(f"\nAverage Results (K=4):")
    print(f"  Baseline:   {avg_baseline_tps:.1f} tok/s")
    print(f"  Speculative: {avg_spec_tps:.1f} tok/s")
    print(f"  Speedup:    {avg_speedup:.2f}×")

    print(f"\nPer-Test Results:")
    for i, r in enumerate(results, 1):
        print(f"  Test {i}: {r['speedup']:.2f}× speedup ({r['baseline_tps']:.1f} → {r['spec_tps_k4']:.1f} tok/s)")

    if avg_speedup >= 1.3:
        print(f"\n✅ SIGNIFICANT SPEEDUP! 1.7B draft model works well!")
    elif avg_speedup >= 1.1:
        print(f"\n⚠️  MODERATE SPEEDUP. 1.7B draft model shows some benefit.")
    else:
        print(f"\n❌ NO SIGNIFICANT SPEEDUP. Even 1.7B draft model doesn't match well.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    test_1_7b_speculative()
