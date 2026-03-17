#!/usr/bin/env python3
"""
运行带 Profiling 的 Prefill 测试

使用方法:
    export OMLX_ENABLE_PROFILING=true
    python run_profiled_prefill.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 启用 profiling
os.environ['OMLX_ENABLE_PROFILING'] = 'true'

import mlx.core as mx
from transformers import AutoTokenizer

from omlx.model_wrapper import ModelWrapper
from omlx.generate import generate
from omlx.profiling import print_profiling_stats, get_global_profiler


def run_profiled_prefill(
    model_path: str,
    prompt_length: int = 8192,
    max_tokens: int = 1
):
    """运行带 profiling 的 Prefill 测试

    Args:
        model_path: 模型路径
        prompt_length: Prompt 长度
        max_tokens: 生成的 token 数（最小值=1，只测试 Prefill）
    """
    print("=" * 80)
    print("Prefill 性能分析（Profiling Enabled）")
    print("=" * 80)
    print(f"模型: {model_path}")
    print(f"Prompt 长度: {prompt_length}")
    print(f"Max tokens: {max_tokens}\n")

    profiler = get_global_profiler()

    if not profiler.enabled:
        print("⚠️  Profiling 未启用，请设置 export OMLX_ENABLE_PROFILING=true")
        return

    # ========== 模型加载 ==========
    print("🔄 加载模型...")
    model_wrapper = ModelWrapper(
        model_name_or_path=model_path,
        lazy=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("  ✅ 模型加载完成\n")

    # ========== 准备输入 ==========
    print(f"🔄 准备输入 ({prompt_length} tokens)...")
    # 生成足够长的测试 prompt
    test_text = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    input_ids = tokenizer.encode(test_text, return_tensors="np")
    prompt = tokenizer.decode(input_ids[0][:prompt_length])
    print(f"  ✅ 输入准备完成: {len(prompt)} chars\n")

    # ========== Prefill 测试 ==========
    print(f"🔄 运行 Prefill 测试...\n")

    # 预热
    print("  预热...")
    generate(
        model_wrapper,
        "Hello",
        verbose=False,
        max_tokens=1
    )
    mx.metal.clear_cache()

    # 重置 profiler
    profiler.reset()

    # 实际测试
    print("  测试中...")
    response = generate(
        model_wrapper,
        prompt,
        verbose=False,
        max_tokens=max_tokens
    )

    print(f"  ✅ 测试完成\n")
    print(f"  生成: {response[:100]}...\n")

    # ========== 打印统计 ==========
    print("\n" + "=" * 80)
    print("📊 性能分析结果")
    print("=" * 80 + "\n")

    print_profiling_stats(top_n=30, min_percent=0.5)

    # ========== 详细分析 ==========
    stats = profiler.get_stats()

    print("\n" + "=" * 80)
    print("🔍 瓶颈分析")
    print("=" * 80 + "\n")

    # 找出最耗时的操作
    bottlenecks = [
        (name, op_stats)
        for name, op_stats in stats['top_operations']
        if op_stats['percent'] > 5.0  # 占比 > 5%
    ]

    if bottlenecks:
        print(f"发现 {len(bottlenecks)} 个主要瓶颈（占比 > 5%）:\n")
        for name, op_stats in bottlenecks:
            print(f"  ⚠️  {name}")
            print(f"      - 平均时间: {op_stats['avg_ms']:.2f} ms")
            print(f"      - 总时间: {op_stats['total_ms']:.1f} ms")
            print(f"      - 占比: {op_stats['percent']:.1f}%")
            print(f"      - 执行次数: {op_stats['count']}")
            print()
    else:
        print("✅ 无明显瓶颈（所有操作占比 < 5%）")

    # ========== 保存报告 ==========
    import json

    report_path = '/tmp/prefill_profiling_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📄 详细报告已保存到: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="运行带 Profiling 的 Prefill 测试")
    parser.add_argument(
        '--model',
        type=str,
        default='~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf',
        help='模型路径'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=8192,
        help='Prompt 长度'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1,
        help='最大生成 token 数（默认=1，只测试 Prefill）'
    )

    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)

    try:
        run_profiled_prefill(
            model_path=model_path,
            prompt_length=args.length,
            max_tokens=args.max_tokens
        )
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
