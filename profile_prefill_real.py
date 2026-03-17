#!/usr/bin/env python3
"""
真实 Prefill 性能分析工具

使用真实模型测试 Prefill 性能，分析瓶颈
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from transformers import AutoTokenizer

from omlx import generate
from omlx.models.qwen3 import ModelArgs, Model


def run_prefill_analysis(
    model_path: str,
    prompt_length: int = 8192,
    trials: int = 5
):
    """运行 Prefill 性能分析

    Args:
        model_path: 模型路径
        prompt_length: Prompt 长度
        trials: 测试轮数
    """
    print("=" * 80)
    print("Prefill 性能分析（真实模型）")
    print("=" * 80)
    print(f"模型: {model_path}")
    print(f"Prompt 长度: {prompt_length}")
    print(f"测试轮数: {trials}\n")

    # ========== 1. 加载模型 ==========
    print("🔄 加载模型...")
    load_start = time.perf_counter()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型（使用 omlx 的方式）
    from omlx.model_wrapper import ModelWrapper

    model_wrapper = ModelWrapper(
        model_name_or_path=model_path,
        lazy=True
    )

    load_time = time.perf_counter() - load_start
    print(f"  ✅ 模型加载完成: {load_time:.2f}s\n")

    # ========== 2. 准备输入 ==========
    print(f"🔄 准备输入 ({prompt_length} tokens)...")

    # 生成足够长的文本
    test_text = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 8)

    # Tokenize
    input_ids = tokenizer.encode(test_text, return_tensors="np")
    tokens = mx.array(input_ids[0][:prompt_length])
    mx.eval(tokens)

    actual_length = len(tokens)
    print(f"  ✅ 输入准备完成: {actual_length} tokens\n")

    # ========== 3. 预热 ==========
    print("🔄 预热...")
    warmup_start = time.perf_counter()

    # 小规模预热
    warmup_tokens = mx.array(input_ids[0][:128])
    _ = generate(
        model_wrapper.model,
        tokenizer,
        warmup_tokens,
        max_tokens=1,
        verbose=False
    )

    mx.metal.clear_cache()

    warmup_time = time.perf_counter() - warmup_start
    print(f"  ✅ 预热完成: {warmup_time:.2f}s\n")

    # ========== 4. Prefill 性能测试 ==========
    print(f"🔄 Prefill 性能测试 ({trials} 轮)...\n")

    timings = {
        'total': [],
        'token_throughput': [],
        'per_token_latency': []
    }

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}:")

        # 清理缓存
        mx.metal.clear_cache()

        # ===== 执行 Prefill =====
        # 使用 generate 只做 prefill（max_tokens=0）
        trial_start = time.perf_counter()

        # 获取第一个 token 的时间（Prefill）
        output = generate(
            model_wrapper.model,
            tokenizer,
            tokens,
            max_tokens=1,  # 只生成 1 个 token 来测量 prefill
            verbose=False
        )

        mx.synchronize()  # 确保所有操作完成

        trial_time = time.perf_counter() - trial_start

        # 计算吞吐量
        throughput = actual_length / trial_time
        per_token_latency = (trial_time / actual_length) * 1000  # ms

        timings['total'].append(trial_time)
        timings['token_throughput'].append(throughput)
        timings['per_token_latency'].append(per_token_latency)

        print(f"    时间: {trial_time:.3f}s")
        print(f"    吞吐量: {throughput:.1f} tok/s")
        print(f"    每 token 延迟: {per_token_latency:.3f} ms\n")

    # ========== 5. 统计分析 ==========
    avg_time = sum(timings['total']) / len(timings['total'])
    avg_throughput = sum(timings['token_throughput']) / len(timings['token_throughput'])
    avg_per_token = sum(timings['per_token_latency']) / len(timings['per_token_latency'])

    min_time = min(timings['total'])
    max_time = max(timings['total'])
    max_throughput = max(timings['token_throughput'])
    min_throughput = min(timings['token_throughput'])

    print("=" * 80)
    print("📊 Prefill 性能汇总")
    print("=" * 80)
    print(f"Prompt 长度: {actual_length} tokens")
    print(f"测试轮数: {trials}\n")

    print(f"平均 Prefill 时间: {avg_time:.3f}s")
    print(f"平均吞吐量: {avg_throughput:.1f} tok/s")
    print(f"平均每 token 延迟: {avg_per_token:.3f} ms\n")

    print(f"最快: {min_time:.3f}s ({max_throughput:.1f} tok/s)")
    print(f"最慢: {max_time:.3f}s ({min_throughput:.1f} tok/s)")
    print(f"变异系数: {(max_time - min_time) / avg_time * 100:.1f}%\n")

    # ========== 6. Metal 统计 ==========
    print("📊 Metal 统计:")
    try:
        # 获取 Metal 统计信息
        metal_stats = mx.metal.get_active_memory()
        print(f"  Active Memory: {metal_stats / 1024**3:.2f} GB")

        cache_stats = mx.metal.get_cache_memory()
        print(f"  Cache Memory: {cache_stats / 1024**3:.2f} GB")
    except:
        print("  ⚠️  Metal 统计信息不可用")

    print()

    # ========== 7. 瓶颈分析 ==========
    print("🔍 瓶颈分析:")

    # 根据性能数据判断瓶颈
    if avg_throughput < 800:
        print("  ⚠️  Prefill 吞吐量较低（< 800 tok/s）")
        print("  可能的瓶颈:")
        print("    - Metal kernel 效率")
        print("    - 内存带宽")
        print("    - Flash Attention 配置")
    elif avg_throughput < 1200:
        print("  ✅ Prefill 吞吐量中等（800-1200 tok/s）")
        print("  可能的优化:")
        print("    - 调整 batch size")
        print("    - 优化 tensor 布局")
    else:
        print("  ✅ Prefill 吞吐量较高（> 1200 tok/s）")
        print("  性能良好")

    print()

    # ========== 8. 保存报告 ==========
    report = {
        'model_path': model_path,
        'prompt_length': actual_length,
        'trials': trials,
        'avg_prefill_time_s': avg_time,
        'avg_throughput_tps': avg_throughput,
        'avg_per_token_latency_ms': avg_per_token,
        'min_time_s': min_time,
        'max_time_s': max_time,
        'max_throughput_tps': max_throughput,
        'min_throughput_tps': min_throughput,
        'variability_percent': (max_time - min_time) / avg_time * 100,
        'all_timings': timings
    }

    report_path = '/tmp/prefill_analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 报告已保存到: {report_path}")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="真实 Prefill 性能分析")
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
        '--trials',
        type=int,
        default=5,
        help='测试轮数'
    )

    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)

    try:
        run_prefill_analysis(
            model_path=model_path,
            prompt_length=args.length,
            trials=args.trials
        )
        return 0
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
