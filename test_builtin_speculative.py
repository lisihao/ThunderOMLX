#!/usr/bin/env python3
"""
Test MLX built-in speculative decoding.
"""

import time
from pathlib import Path
from mlx_lm import load, generate

def test_builtin_speculative():
    """Test using MLX's built-in speculative decoding."""

    print("🚀 MLX Built-in Speculative Decoding Test")
    print("=" * 70)

    # Model paths
    target_path = Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    draft_path = Path.home() / ".omlx" / "models" / "Qwen3.5-0.8B-MLX-4bit"

    # Load models
    print(f"\n📥 Loading target model: {target_path.name}")
    target_model, target_tokenizer = load(str(target_path))
    print(f"✅ Target model loaded")

    print(f"\n📥 Loading draft model: {draft_path.name}")
    draft_model, draft_tokenizer = load(str(draft_path))
    print(f"✅ Draft model loaded")

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The capital of France is",
        "To be or not to be",
    ]

    max_tokens = 50

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: \"{prompt}\"")
        print(f"{'='*70}")

        # Speculative decoding with builtin function
        print(f"\n🎯 Speculative Decoding (K=4):")
        start_time = time.time()

        response = generate(
            target_model,
            target_tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            draft_model=draft_model,  # ← Use draft model
            num_draft_tokens=4,       # ← K = 4
            verbose=False
        )

        spec_time = time.time() - start_time
        spec_tps = max_tokens / spec_time if spec_time > 0 else 0

        print(f"  Generated: \"{response}\"")
        print(f"  Time: {spec_time:.2f}s")
        print(f"  Throughput: {spec_tps:.1f} tok/s")

        # Baseline (no speculative decoding)
        print(f"\n🔄 Baseline (no draft model):")
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

        print(f"  Generated: \"{response_baseline}\"")
        print(f"  Time: {baseline_time:.2f}s")
        print(f"  Throughput: {baseline_tps:.1f} tok/s")

        # Speedup
        speedup = baseline_time / spec_time if spec_time > 0 else 1.0
        print(f"\n📊 Speedup: {speedup:.2f}x")

    print(f"\n{'='*70}")
    print("✅ All tests completed")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_builtin_speculative()
