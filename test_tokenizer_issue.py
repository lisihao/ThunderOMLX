#!/usr/bin/env python3
"""验证 tokenizer.detokenizer 和 get_vocab() 的问题"""

import time
from pathlib import Path
from mlx_lm import load

# 加载模型
model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
model, tokenizer = load(str(model_path))

print("=" * 80)
print("测试 1: 访问 tokenizer.detokenizer (property) 的开销")
print("=" * 80)

# 测试访问 property 10 次
times = []
for i in range(10):
    start = time.perf_counter()
    detok = tokenizer.detokenizer  # 每次都创建新实例
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"第 {i+1} 次: {elapsed:.2f} ms")

print(f"\n平均时间: {sum(times)/len(times):.2f} ms")

print("\n" + "=" * 80)
print("测试 2: 直接调用 get_vocab() 的开销")
print("=" * 80)

# 测试调用 get_vocab() 10 次
times2 = []
for i in range(10):
    start = time.perf_counter()
    vocab = tokenizer.get_vocab()
    elapsed = (time.perf_counter() - start) * 1000
    times2.append(elapsed)
    print(f"第 {i+1} 次: {elapsed:.2f} ms")

print(f"\n平均时间: {sum(times2)/len(times2):.2f} ms")

print("\n" + "=" * 80)
print("测试 3: 缓存 vocab 后创建 detokenizer")
print("=" * 80)

# 预先缓存 vocab
print("预先获取 vocab...")
start = time.perf_counter()
cached_vocab = tokenizer.get_vocab()
elapsed = (time.perf_counter() - start) * 1000
print(f"首次获取: {elapsed:.2f} ms")

# Monkey patch tokenizer.vocab 属性
original_get_vocab = tokenizer.get_vocab
tokenizer.get_vocab = lambda: cached_vocab

# 测试创建 detokenizer（使用缓存的 vocab）
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

times3 = []
for i in range(10):
    start = time.perf_counter()
    detok = NaiveStreamingDetokenizer(tokenizer)
    elapsed = (time.perf_counter() - start) * 1000
    times3.append(elapsed)
    print(f"第 {i+1} 次: {elapsed:.2f} ms")

print(f"\n平均时间: {sum(times3)/len(times3):.2f} ms")

# 恢复原始 get_vocab
tokenizer.get_vocab = original_get_vocab

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"访问 property (创建新实例): {sum(times)/len(times):.2f} ms")
print(f"直接调用 get_vocab():        {sum(times2)/len(times2):.2f} ms")
print(f"使用缓存 vocab:               {sum(times3)/len(times3):.2f} ms")
print(f"\n加速比: {(sum(times)/len(times)) / (sum(times3)/len(times3)):.1f}x")
