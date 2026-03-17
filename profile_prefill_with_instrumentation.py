#!/usr/bin/env python3
"""
使用通用 Profiling 框架分析 Prefill 性能

基于现有的 instrumentation 框架扩展
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

from omlx.profiling import get_global_profiler, print_profiling_stats
from omlx.model_wrapper import ModelWrapper


def analyze_prefill_performance(
    model_path: str,
    prompt_length: int = 8192,
    trials: int = 3
):
    """分析 Prefill 性能

    使用 profiling 框架自动记录各个阶段的时间
    """
    print("=" * 80)
    print("Prefill 性能分析（使用 Profiling 框架）")
    print("=" * 80)
    print(f"模型: {model_path}")
    print(f"Prompt 长度: {prompt_length}")
    print(f"测试轮数: {trials}\n")

    profiler = get_global_profiler()

    # ========== 模型加载 ==========
    print("🔄 加载模型...")
    with profiler.section("0.model_load"):
        model_wrapper = ModelWrapper(
            model_name_or_path=model_path,
            lazy=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("  ✅ 模型加载完成\n")

    # ========== 准备输入 ==========
    print(f"🔄 准备输入 ({prompt_length} tokens)...")
    with profiler.section("1.input_prep"):
        # 生成测试 prompt
        test_text = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)

        with profiler.section("1.input_prep.tokenize"):
            input_ids = tokenizer.encode(test_text, return_tensors="np")

        with profiler.section("1.input_prep.to_mlx"):
            tokens = mx.array(input_ids[0][:prompt_length])
            mx.eval(tokens)

    actual_length = len(tokens)
    print(f"  ✅ 输入准备完成: {actual_length} tokens\n")

    # ========== 预热 ==========
    print("🔄 预热...")
    with profiler.section("2.warmup"):
        # 使用小输入预热
        warmup_tokens = mx.array(input_ids[0][:128])

        # 调用模型（简化 - 实际需要完整的 generate 流程）
        # 这里暂时用 mock，实际应该调用 engine
        # output = model_wrapper.model(warmup_tokens)

        mx.metal.clear_cache()

    print("  ✅ 预热完成\n")

    # ========== Prefill 测试 ==========
    print(f"🔄 Prefill 性能测试 ({trials} 轮)...\n")

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}:")

        mx.metal.clear_cache()

        with profiler.section(f"3.prefill_trial_{trial}"):
            # ===== 实际的 Prefill 流程 =====
            # 注意：这里需要集成到实际的 scheduler/engine 中
            # 当前是简化版本，展示如何使用 profiling

            # 1. Token Embedding
            with profiler.section(f"3.prefill_trial_{trial}.embedding"):
                # embeddings = model_wrapper.model.model.embed_tokens(tokens)
                # mx.eval(embeddings)
                pass

            # 2. Layer-by-layer Forward
            # 实际应该从 scheduler 或 engine 中注入 profiling
            num_layers = 64  # Qwen3-30B
            for layer_idx in range(num_layers):
                layer_prefix = f"3.prefill_trial_{trial}.layer_{layer_idx}"

                # QKV Projection
                with profiler.section(f"{layer_prefix}.qkv_proj"):
                    # q, k, v = layer.self_attn.qkv_proj(hidden_states)
                    # mx.eval(q, k, v)
                    pass

                # Attention
                with profiler.section(f"{layer_prefix}.attention"):
                    # attn_out = layer.self_attn(q, k, v)
                    # mx.eval(attn_out)
                    pass

                # FFN
                with profiler.section(f"{layer_prefix}.ffn"):
                    # ffn_out = layer.mlp(attn_out)
                    # mx.eval(ffn_out)
                    pass

            # 3. Output Projection
            with profiler.section(f"3.prefill_trial_{trial}.output_proj"):
                # logits = model_wrapper.model.lm_head(hidden_states)
                # mx.eval(logits)
                pass

            # 4. Synchronize
            with profiler.section(f"3.prefill_trial_{trial}.synchronize"):
                mx.synchronize()

        print(f"    ✅ Trial {trial + 1} 完成\n")

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

        # 生成优化建议
        print("💡 优化建议:\n")

        for name, op_stats in bottlenecks[:3]:  # Top 3
            if 'qkv_proj' in name:
                print("  1. QKV Projection 是瓶颈:")
                print("     - 考虑 Fused QKV Projection")
                print("     - 批量 bfloat16 eval 优化")
            elif 'attention' in name:
                print("  2. Attention 是瓶颈:")
                print("     - 启用 Chunked Prefill")
                print("     - 升级到 FlashAttention-3")
            elif 'ffn' in name:
                print("  3. FFN 是瓶颈:")
                print("     - MoE Expert 并行化")
                print("     - Activation Fusion")

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

    parser = argparse.ArgumentParser(description="Prefill 性能分析（Profiling 框架）")
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
        default=3,
        help='测试轮数'
    )

    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)

    try:
        analyze_prefill_performance(
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
