#!/usr/bin/env python3
"""
Debug inference flow to understand why acceptance rate is 0%.
"""

from pathlib import Path
import mlx.core as mx
from mlx_lm import load as mlx_load

def test_single_token_generation():
    """Test if both models generate same next token for same prompt."""

    print("🔍 Single Token Generation Test")
    print("=" * 70)

    # Model paths
    target_path = Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    draft_path = Path.home() / ".omlx" / "models" / "Qwen3.5-0.8B-MLX-4bit"

    # Load models
    print(f"\n📥 Loading models...")
    target_model, target_tokenizer = mlx_load(str(target_path))
    draft_model, draft_tokenizer = mlx_load(str(draft_path))
    print(f"✅ Models loaded")

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The capital of France is",
        "To be or not to be",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: \"{prompt}\"")
        print(f"{'='*70}")

        # Tokenize
        prompt_tokens = target_tokenizer.encode(prompt)
        prompt_mx = mx.array([prompt_tokens])

        print(f"Tokens: {prompt_tokens}")

        # Target model forward pass
        print(f"\n🎯 Target model:")
        target_logits = target_model(prompt_mx)
        if isinstance(target_logits, tuple):
            target_logits, _ = target_logits

        mx.eval(target_logits)

        # Get top-5 predictions
        target_probs = mx.softmax(target_logits[0, -1, :], axis=-1)
        target_top5_indices = mx.argsort(target_probs, axis=-1)[-5:][::-1]
        target_top5_probs = target_probs[target_top5_indices]

        print(f"  Top-5 predictions:")
        for i, (idx, prob) in enumerate(zip(target_top5_indices.tolist(), target_top5_probs.tolist())):
            token_text = target_tokenizer.decode([idx])
            print(f"    {i+1}. Token {idx:6d} ({prob:.2%}): \"{token_text}\"")

        # Draft model forward pass
        print(f"\n🎯 Draft model:")
        draft_logits = draft_model(prompt_mx)
        if isinstance(draft_logits, tuple):
            draft_logits, _ = draft_logits

        mx.eval(draft_logits)

        # Get top-5 predictions
        draft_probs = mx.softmax(draft_logits[0, -1, :], axis=-1)
        draft_top5_indices = mx.argsort(draft_probs, axis=-1)[-5:][::-1]
        draft_top5_probs = draft_probs[draft_top5_indices]

        print(f"  Top-5 predictions:")
        for i, (idx, prob) in enumerate(zip(draft_top5_indices.tolist(), draft_top5_probs.tolist())):
            token_text = draft_tokenizer.decode([idx])
            print(f"    {i+1}. Token {idx:6d} ({prob:.2%}): \"{token_text}\"")

        # Compare top-1
        target_top1 = target_top5_indices[0].item()
        draft_top1 = draft_top5_indices[0].item()

        print(f"\n📊 Comparison:")
        print(f"  Target top-1: {target_top1} (\"{target_tokenizer.decode([target_top1])}\")")
        print(f"  Draft top-1:  {draft_top1} (\"{draft_tokenizer.decode([draft_top1])}\")")

        if target_top1 == draft_top1:
            print(f"  ✅ MATCH - Would be accepted")
        else:
            print(f"  ❌ MISMATCH - Would be rejected")


def test_sequential_generation():
    """Test if draft model can generate reasonable continuation."""

    print(f"\n\n{'='*70}")
    print("Sequential Generation Test")
    print(f"{'='*70}")

    draft_path = Path.home() / ".omlx" / "models" / "Qwen3.5-0.8B-MLX-4bit"

    print(f"\n📥 Loading draft model...")
    draft_model, draft_tokenizer = mlx_load(str(draft_path))
    print(f"✅ Draft model loaded")

    # Test prompt
    prompt = "Once upon a time"
    print(f"\nPrompt: \"{prompt}\"")

    # Tokenize
    prompt_tokens = draft_tokenizer.encode(prompt)
    prompt_mx = mx.array([prompt_tokens])

    print(f"Tokens: {prompt_tokens}")

    # Prefill
    logits = draft_model(prompt_mx)
    if isinstance(logits, tuple):
        logits, cache = logits
    else:
        cache = None

    mx.eval(logits)

    # Get first token
    current_token = mx.argmax(logits[0, -1, :]).item()
    generated_tokens = [current_token]

    print(f"\n🎯 Generating 10 tokens with draft model...")

    # Generate 10 tokens
    for i in range(10):
        input_ids = mx.array([[current_token]])
        logits = draft_model(input_ids, cache=cache)

        if isinstance(logits, tuple):
            logits, cache = logits

        next_token = mx.argmax(logits[0, -1, :]).item()
        generated_tokens.append(next_token)
        current_token = next_token

        # Decode
        text = draft_tokenizer.decode(generated_tokens)
        print(f"  Step {i+1}: Token {next_token:6d} -> \"{text}\"")


if __name__ == "__main__":
    try:
        # Test 1: Single token generation comparison
        test_single_token_generation()

        # Test 2: Sequential generation
        test_sequential_generation()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
