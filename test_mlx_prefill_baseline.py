#!/usr/bin/env python3
"""
测试 MLX BatchGenerator 的原生 prefill 性能（无 ThunderOMLX 层）
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from mlx_lm import load, generate

def test_mlx_prefill_baseline():
    """测试 MLX 原生 prefill 性能"""
    print("=" * 80)
    print("MLX BatchGenerator 原生 Prefill 性能测试")
    print("=" * 80)
    print()

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return

    print(f"📦 加载模型: {model_path.name}")
    model, tokenizer = load(str(model_path))
    print("✅ 模型加载完成")
    print()

    # 准备 8192 tokens 的 prompt
    prompt_text = "Hello, I am a large language model. " * 300
    tokens = tokenizer.encode(prompt_text)

    # 截取到 8192 tokens
    if len(tokens) > 8192:
        tokens = tokens[:8192]
    elif len(tokens) < 8192:
        # 重复填充到 8192
        while len(tokens) < 8192:
            tokens.extend(tokens)
        tokens = tokens[:8192]

    print(f"📝 Prompt tokens: {len(tokens)}")
    prompt_text = tokenizer.decode(tokens)
    print()

    # 测试 3 次取平均
    prefill_times = []

    for i in range(3):
        print(f"🔄 测试 {i+1}/3...")

        # 清理缓存
        mx.clear_cache()

        # 测试 prefill
        start = time.perf_counter()

        # 使用 mlx_lm.generate 但只生成 1 个 token
        # 这样可以测量 prefill + 1 个 token 的时间
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_tokens=1,
            verbose=False,
            temperature=0.0,
        )

        # 强制同步
        mx.eval(output)

        elapsed = (time.perf_counter() - start) * 1000  # ms
        prefill_times.append(elapsed)

        print(f"   Prefill + 1 gen: {elapsed:.2f}ms")

    print()
    print("=" * 80)
    print("📊 测试结果")
    print("=" * 80)

    avg_time = sum(prefill_times) / len(prefill_times)
    min_time = min(prefill_times)
    max_time = max(prefill_times)

    print(f"平均时间: {avg_time:.2f}ms")
    print(f"最小时间: {min_time:.2f}ms")
    print(f"最大时间: {max_time:.2f}ms")
    print()

    # 计算 prefill TPS（假设第一个 token 生成时间可以忽略）
    prefill_tps = 8192 / (avg_time / 1000)

    print(f"Prefill TPS: {prefill_tps:.1f} tok/s")
    print(f"Prefill 耗时: {avg_time:.2f}ms ({8192} tokens)")
    print()

    # 对比
    print("📊 性能对比:")
    print(f"  MLX 原生:        {prefill_tps:.1f} tok/s")
    print(f"  ThunderOMLX:     696.7 tok/s")
    print(f"  oMLX baseline:   880.3 tok/s")
    print()

    if prefill_tps < 696.7:
        print("⚠️ MLX 原生比 ThunderOMLX 还慢！可能是测试方法问题或 MLX 版本差异")
    elif prefill_tps > 880.3:
        print("✅ MLX 原生比 baseline 快！ThunderOMLX 有性能损失")
    else:
        print("✓ MLX 原生性能正常，ThunderOMLX 接近原生")

if __name__ == "__main__":
    test_mlx_prefill_baseline()
