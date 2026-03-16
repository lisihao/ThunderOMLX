#!/usr/bin/env python3
"""
Test MLX official generate function to verify models work correctly.
"""

from pathlib import Path
from mlx_lm import load, generate

def test_official_generate():
    """Test using mlx-lm's official generate function."""

    print("🔍 Testing MLX Official Generate")
    print("=" * 70)

    # Model paths
    target_path = Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    draft_path = Path.home() / ".omlx" / "models" / "Qwen3.5-0.8B-MLX-4bit"

    test_prompts = [
        "Once upon a time",
        "The capital of France is",
        "To be or not to be",
    ]

    # Test target model
    print(f"\n📥 Loading target model: {target_path.name}")
    target_model, target_tokenizer = load(str(target_path))
    print(f"✅ Target model loaded")

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"Target Model - Prompt: \"{prompt}\"")
        print(f"{'='*70}")

        response = generate(
            target_model,
            target_tokenizer,
            prompt=prompt,
            max_tokens=20,
            verbose=False
        )

        print(f"Generated: \"{response}\"")

    # Test draft model
    print(f"\n\n📥 Loading draft model: {draft_path.name}")
    draft_model, draft_tokenizer = load(str(draft_path))
    print(f"✅ Draft model loaded")

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"Draft Model - Prompt: \"{prompt}\"")
        print(f"{'='*70}")

        response = generate(
            draft_model,
            draft_tokenizer,
            prompt=prompt,
            max_tokens=20,
            verbose=False
        )

        print(f"Generated: \"{response}\"")


if __name__ == "__main__":
    test_official_generate()
