#!/usr/bin/env python3
from pathlib import Path
from mlx_lm import load, generate
import time

print("🔍 Native MLX - pp1024/tg128\n")

model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
model, tokenizer = load(str(model_path))

# 1024 token prompt
filler = "The quick brown fox jumps over the lazy dog. " * 1000
tokens = tokenizer.encode(filler)[:1024]
prompt = tokenizer.decode(tokens)

print(f"Prompt tokens: {len(tokenizer.encode(prompt))}\n")

start = time.perf_counter()
response = generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=128, verbose=True)
end = time.perf_counter()

print(f"\n✅ Total time: {end - start:.3f}s")
print(f"📊 Overall: {128 / (end - start):.1f} tok/s")
