#!/usr/bin/env python3
"""
测试 MLX-LM 的 cache API

目标：理解如何正确使用 cache 参数
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from mlx_lm import load

# 加载模型
print("Loading model...")
model, tokenizer = load('mlx-community/Qwen2.5-3B-Instruct-8bit')
print("Model loaded\n")

# 测试 prompt
prompt = "Hello, how are you?"
tokens = tokenizer.encode(prompt)
print(f"Prompt: {prompt}")
print(f"Tokens: {tokens[:10]}... (total: {len(tokens)})\n")

# 测试 1: 不使用 cache
print("="*60)
print("Test 1: Without cache")
print("="*60)

input_ids = mx.array([tokens])
print(f"Input shape: {input_ids.shape}")

logits = model(input_ids, cache=None)
print(f"Logits shape: {logits.shape}")
print(f"Logits type: {type(logits)}")

# 测试 2: 使用 cache
print("\n" + "="*60)
print("Test 2: With cache (chunk 1)")
print("="*60)

# 分成两个 chunk
chunk1 = tokens[:3]
chunk2 = tokens[3:]

input_ids_1 = mx.array([chunk1])
print(f"Chunk 1 shape: {input_ids_1.shape}")

# 第一次调用，cache 为 None
result = model(input_ids_1, cache=None)
print(f"Result type: {type(result)}")

# 检查返回值
if isinstance(result, tuple):
    logits_1, cache_1 = result
    print(f"✅ Model returns (logits, cache)")
    print(f"   Logits shape: {logits_1.shape}")
    print(f"   Cache type: {type(cache_1)}")
    if cache_1 is not None:
        print(f"   Cache structure: {type(cache_1)}")
        if isinstance(cache_1, list):
            print(f"   Cache length: {len(cache_1)}")
            if len(cache_1) > 0:
                print(f"   Cache[0] type: {type(cache_1[0])}")
else:
    print(f"❌ Model only returns logits (not a tuple)")
    print(f"   Need to check model implementation")
    logits_1 = result
    cache_1 = None

# 测试 3: 使用累积的 cache
if cache_1 is not None:
    print("\n" + "="*60)
    print("Test 3: With cache (chunk 2)")
    print("="*60)

    input_ids_2 = mx.array([chunk2])
    print(f"Chunk 2 shape: {input_ids_2.shape}")

    result_2 = model(input_ids_2, cache=cache_1)

    if isinstance(result_2, tuple):
        logits_2, cache_2 = result_2
        print(f"✅ Model returns (logits, cache)")
        print(f"   Logits shape: {logits_2.shape}")
        print(f"   Cache type: {type(cache_2)}")
    else:
        print(f"⚠️  Model only returns logits")
        logits_2 = result_2
        cache_2 = None
else:
    print("\n⚠️  Cache is None, cannot test chunk 2")

# 结论
print("\n" + "="*60)
print("Conclusion")
print("="*60)

if cache_1 is not None:
    print("✅ Model supports cache!")
    print("   - model(inputs, cache=None) returns (logits, cache)")
    print("   - Can accumulate cache across chunks")
    print("   - Ready for Phase 2 implementation")
else:
    print("❌ Model doesn't support cache in expected way")
    print("   - Need alternative approach")
