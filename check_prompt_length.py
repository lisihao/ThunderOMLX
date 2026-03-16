#!/usr/bin/env python3
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/Users/lisihao/models/qwen3.5-35b-mlx")

# 生成测试 prompt
filler = "The quick brown fox jumps over the lazy dog. " * 1000
prompt = filler[:40000]

# 计算实际 tokens
tokens = tokenizer.encode(prompt)
print(f"Prompt length: {len(prompt)} characters")
print(f"Actual tokens: {len(tokens)}")
print(f"Target: 8192 tokens")
print(f"Difference: {8192 - len(tokens)} tokens")
