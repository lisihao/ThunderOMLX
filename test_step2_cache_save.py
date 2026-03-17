#!/usr/bin/env python3
"""Test Step 2 optimization with actual cache save operations."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer

from omlx.engine.batched import BatchedEngine
from omlx.scheduler import SchedulerConfig

# Enable DEBUG logging to see cache operations
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    model_path = Path("~/models/qwen3.5-35b-mlx").expanduser()

    print("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    # Generate a fixed prompt (8K tokens)
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 1600
    prompt = tokenizer.encode(prompt_text)[:8192]
    prompt_text = tokenizer.decode(prompt)

    print(f"✅ Prompt: {len(prompt)} tokens\n")

    print("⏳ Initializing engine...")
    scheduler_config = SchedulerConfig()
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    await engine.start()
    print("✅ Engine started\n")

    try:
        # Run 3 requests with the SAME prompt (should trigger prefix cache save/load)
        for i in range(3):
            print(f"\n{'='*80}")
            print(f"Request {i+1}/3")
            print('='*80)

            output_tokens = 0
            async for output in engine.stream_generate(
                prompt=prompt_text,
                max_tokens=32,  # Short generation
                temperature=0.0
            ):
                output_tokens += 1

            print(f"✅ Generated {output_tokens} tokens\n")

            # Small delay between requests
            await asyncio.sleep(0.5)

        print("\n" + "="*80)
        print("✅ All requests completed")
        print("="*80)

    finally:
        await engine.stop()
        print("\n✅ Engine stopped")

if __name__ == "__main__":
    asyncio.run(main())
