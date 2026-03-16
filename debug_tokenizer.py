#!/usr/bin/env python3
"""
Debug tokenizer compatibility between target and draft models.
"""

from pathlib import Path
from mlx_lm import load as mlx_load

def main():
    print("🔍 Tokenizer Compatibility Check")
    print("=" * 70)

    # Model paths
    target_path = Path.home() / ".omlx" / "models" / "Qwen3.5-35B-A3B-6bit"
    draft_path = Path.home() / ".omlx" / "models" / "Qwen3.5-0.8B-MLX-4bit"

    # Load tokenizers
    print(f"\n📥 Loading target tokenizer: {target_path.name}")
    _, target_tokenizer = mlx_load(str(target_path))

    print(f"📥 Loading draft tokenizer: {draft_path.name}")
    _, draft_tokenizer = mlx_load(str(draft_path))

    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The capital of France is",
        "To be or not to be",
        "Hello, world!",
        "1 + 1 = ",
    ]

    print(f"\n{'='*70}")
    print("Tokenization Comparison")
    print(f"{'='*70}")

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")

        # Tokenize with both
        target_tokens = target_tokenizer.encode(prompt)
        draft_tokens = draft_tokenizer.encode(prompt)

        print(f"  Target tokens: {target_tokens}")
        print(f"  Draft tokens:  {draft_tokens}")

        if target_tokens == draft_tokens:
            print(f"  ✅ Match")
        else:
            print(f"  ❌ MISMATCH")

        # Decode
        target_decoded = target_tokenizer.decode(target_tokens)
        draft_decoded = draft_tokenizer.decode(draft_tokens)

        if target_decoded != prompt or draft_decoded != prompt:
            print(f"  ⚠️  Decode mismatch:")
            print(f"     Target: \"{target_decoded}\"")
            print(f"     Draft:  \"{draft_decoded}\"")

    # Check vocab size
    print(f"\n{'='*70}")
    print("Vocabulary Info")
    print(f"{'='*70}")
    print(f"Target vocab size: {target_tokenizer.vocab_size}")
    print(f"Draft vocab size:  {draft_tokenizer.vocab_size}")

    if target_tokenizer.vocab_size != draft_tokenizer.vocab_size:
        print(f"❌ Vocab size MISMATCH")
    else:
        print(f"✅ Vocab size match")

    # Check special tokens
    print(f"\n{'='*70}")
    print("Special Tokens")
    print(f"{'='*70}")
    print(f"Target EOS token ID: {target_tokenizer.eos_token_id}")
    print(f"Draft EOS token ID:  {draft_tokenizer.eos_token_id}")

    print(f"Target BOS token ID: {target_tokenizer.bos_token_id if hasattr(target_tokenizer, 'bos_token_id') else 'N/A'}")
    print(f"Draft BOS token ID:  {draft_tokenizer.bos_token_id if hasattr(draft_tokenizer, 'bos_token_id') else 'N/A'}")

    print(f"Target PAD token ID: {target_tokenizer.pad_token_id if hasattr(target_tokenizer, 'pad_token_id') else 'N/A'}")
    print(f"Draft PAD token ID:  {draft_tokenizer.pad_token_id if hasattr(draft_tokenizer, 'pad_token_id') else 'N/A'}")


if __name__ == "__main__":
    main()
