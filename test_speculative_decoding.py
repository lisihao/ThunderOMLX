#!/usr/bin/env python3
"""
Test Speculative Decoding with Qwen3.5-35B (target) + Qwen3.5-0.8B (draft).
"""

import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_load

from src.omlx.speculative_decoding import SpeculativeDecodingEngine, SpeculativeConfig


def test_speculative_decoding():
    """Test speculative decoding with real models."""

    print("🚀 Speculative Decoding Test")
    print("=" * 70)

    # Model paths
    target_model_path = (
        Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    )
    draft_model_path = (
        Path.home() / ".omlx" / "models" / "Qwen3.5-0.8B-MLX-4bit"
    )

    # Load target model (35B)
    print(f"\n📥 Loading target model: {target_model_path}")
    target_model, target_tokenizer = mlx_load(str(target_model_path))
    print(f"✅ Target model loaded: {target_model.__class__.__name__}")

    # Create speculative engine
    print(f"\n🔧 Creating Speculative Decoding Engine...")
    print(f"   Draft model: {draft_model_path}")
    print(f"   K (speculative tokens): 4")

    engine = SpeculativeDecodingEngine(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        draft_model_path=str(draft_model_path),
        num_speculative_tokens=4,
    )

    print(f"✅ Speculative engine created")

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The capital of France is",
        "To be or not to be",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'='*70}")
        print(f"Prompt: \"{prompt}\"")

        # Tokenize
        prompt_tokens = target_tokenizer.encode(prompt)
        prompt_tokens_mx = mx.array([prompt_tokens])

        print(f"Prompt tokens: {len(prompt_tokens)}")

        # Generate with speculative decoding
        max_tokens = 50
        eos_token_id = target_tokenizer.eos_token_id

        print(f"\n🎯 Generating {max_tokens} tokens with Speculative Decoding...")

        start_time = time.time()
        generated_tokens = []

        for token in engine.generate_speculative(
            prompt_tokens=prompt_tokens_mx,
            max_tokens=max_tokens,
            eos_token_id=eos_token_id,
        ):
            generated_tokens.append(token)

            # Stop if EOS
            if token == eos_token_id:
                break

        elapsed = time.time() - start_time

        # Decode
        generated_text = target_tokenizer.decode(generated_tokens)

        # Statistics
        acceptance_rate = engine.get_acceptance_rate()
        speedup_ratio = engine.get_speedup_ratio()
        tokens_per_second = len(generated_tokens) / elapsed if elapsed > 0 else 0

        print(f"\n📊 Results:")
        print(f"   Generated text: \"{generated_text}\"")
        print(f"   Tokens generated: {len(generated_tokens)}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Throughput: {tokens_per_second:.1f} tok/s")
        print(f"\n📈 Speculative Decoding Stats:")
        print(f"   Acceptance rate: {acceptance_rate:.1%}")
        print(f"   Speedup ratio: {speedup_ratio:.2f}x (theoretical)")
        print(f"   Accepted tokens: {engine.stats['total_accepted_tokens']}")
        print(f"   Drafted tokens: {engine.stats['total_draft_tokens']}")
        print(f"   Bonus tokens: {engine.stats['total_bonus_tokens']}")
        print(f"   Forward passes: {engine.stats['total_forward_passes']}")

    print(f"\n{'='*70}")
    print("✅ All tests completed")
    print(f"{'='*70}")


def test_baseline_generation():
    """Test baseline generation (no speculative decoding) for comparison."""

    print("\n\n🔄 Baseline Generation Test (for comparison)")
    print("=" * 70)

    # Model paths
    target_model_path = (
        Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    )

    # Load target model
    print(f"\n📥 Loading target model: {target_model_path}")
    target_model, target_tokenizer = mlx_load(str(target_model_path))
    print(f"✅ Target model loaded")

    # Test prompt
    prompt = "Once upon a time"
    print(f"\nPrompt: \"{prompt}\"")

    # Tokenize
    prompt_tokens = target_tokenizer.encode(prompt)
    prompt_tokens_mx = mx.array([prompt_tokens])

    print(f"Prompt tokens: {len(prompt_tokens)}")

    # Generate (baseline)
    max_tokens = 50
    eos_token_id = target_tokenizer.eos_token_id

    print(f"\n🎯 Generating {max_tokens} tokens (baseline)...")

    start_time = time.time()

    # Prefill
    logits = target_model(prompt_tokens_mx)
    if isinstance(logits, tuple):
        logits, cache = logits
    else:
        cache = None

    current_token = mx.argmax(logits[0, -1, :]).item()
    generated_tokens = [current_token]

    # Decode loop
    for _ in range(max_tokens - 1):
        input_ids = mx.array([[current_token]])
        logits = target_model(input_ids, cache=cache)

        if isinstance(logits, tuple):
            logits, cache = logits

        next_token = mx.argmax(logits[0, -1, :]).item()
        generated_tokens.append(next_token)
        current_token = next_token

        if next_token == eos_token_id:
            break

    elapsed = time.time() - start_time

    # Decode
    generated_text = target_tokenizer.decode(generated_tokens)

    # Statistics
    tokens_per_second = len(generated_tokens) / elapsed if elapsed > 0 else 0

    print(f"\n📊 Baseline Results:")
    print(f"   Generated text: \"{generated_text}\"")
    print(f"   Tokens generated: {len(generated_tokens)}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {tokens_per_second:.1f} tok/s")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    try:
        # Test speculative decoding
        test_speculative_decoding()

        # Test baseline for comparison
        test_baseline_generation()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
