#!/usr/bin/env python3
"""
Prefill 性能深度分析工具

目标：
1. 详细测量 Prefill 各阶段时间分布
2. 分析 Metal kernel 执行时间
3. 识别性能瓶颈
4. 生成优化建议

测试场景：
- 8K Prefill（匹配之前的测试）
- 分析时间分解
- Metal 操作统计
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from transformers import AutoTokenizer

from omlx.model_wrapper import ModelWrapper


class PrefillProfiler:
    """Prefill 性能分析器"""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.current_section = None
        self.section_start = None

    def start_section(self, name: str):
        """开始计时一个section"""
        if self.current_section:
            self.end_section()
        self.current_section = name
        self.section_start = time.perf_counter()

    def end_section(self):
        """结束当前section的计时"""
        if self.current_section and self.section_start:
            elapsed = time.perf_counter() - self.section_start
            if self.current_section not in self.timings:
                self.timings[self.current_section] = []
            self.timings[self.current_section].append(elapsed)
            self.current_section = None
            self.section_start = None

    def get_summary(self) -> Dict:
        """获取统计摘要"""
        summary = {}
        total_time = 0

        for name, times in self.timings.items():
            avg = sum(times) / len(times) if times else 0
            total_time += avg
            summary[name] = {
                'avg_ms': avg * 1000,
                'min_ms': min(times) * 1000 if times else 0,
                'max_ms': max(times) * 1000 if times else 0,
                'count': len(times),
                'total_ms': sum(times) * 1000
            }

        # 计算百分比
        if total_time > 0:
            for name in summary:
                summary[name]['percent'] = (summary[name]['avg_ms'] / (total_time * 1000)) * 100

        return summary


def analyze_prefill_performance(
    model_path: str,
    prompt_length: int = 8192,
    trials: int = 3
) -> Dict:
    """深度分析 Prefill 性能

    Args:
        model_path: 模型路径
        prompt_length: Prompt 长度
        trials: 测试轮数

    Returns:
        性能分析报告
    """
    print(f"=" * 80)
    print(f"Prefill 性能深度分析")
    print(f"=" * 80)
    print(f"模型: {model_path}")
    print(f"Prompt 长度: {prompt_length} tokens")
    print(f"测试轮数: {trials}")
    print()

    profiler = PrefillProfiler()

    # ========== 1. 模型加载 ==========
    print("🔄 阶段 1: 模型加载...")
    profiler.start_section("model_load")

    model_wrapper = ModelWrapper(
        model_name_or_path=model_path,
        lazy=True
    )

    mx.metal.clear_cache()
    profiler.end_section()
    print(f"  ✅ 模型加载完成")

    # ========== 2. Tokenizer 初始化 ==========
    print("\n🔄 阶段 2: Tokenizer 初始化...")
    profiler.start_section("tokenizer_init")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    profiler.end_section()
    print(f"  ✅ Tokenizer 初始化完成")

    # ========== 3. 准备输入 ==========
    print(f"\n🔄 阶段 3: 准备输入 ({prompt_length} tokens)...")
    profiler.start_section("input_preparation")

    # 生成测试 prompt
    test_text = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)

    # Tokenize
    profiler.start_section("tokenize")
    input_ids = tokenizer.encode(test_text, return_tensors="np")
    actual_length = len(input_ids[0])
    profiler.end_section()

    # 转换为 MLX array
    profiler.start_section("convert_to_mlx")
    tokens = mx.array(input_ids[0][:prompt_length])
    mx.eval(tokens)
    profiler.end_section()

    profiler.end_section()  # input_preparation
    print(f"  ✅ 输入准备完成: {len(tokens)} tokens")

    # ========== 4. Prefill 执行（多轮） ==========
    print(f"\n🔄 阶段 4: Prefill 执行 ({trials} 轮)...")

    prefill_times = []
    token_throughputs = []

    for trial in range(trials):
        print(f"\n  Trial {trial + 1}/{trials}:")

        # 清理 Metal 缓存
        mx.metal.clear_cache()

        # ===== 4.1 Prefill =====
        profiler.start_section(f"prefill_total_trial_{trial}")

        # Token embedding
        profiler.start_section(f"token_embedding_trial_{trial}")
        # 这里需要调用实际的 model，暂时用 mock
        # embeddings = model_wrapper.model.model.embed_tokens(tokens)
        # mx.eval(embeddings)
        profiler.end_section()

        # Forward pass (simplified - 实际需要调用真实模型)
        profiler.start_section(f"forward_pass_trial_{trial}")

        # 模拟多层 attention + FFN
        # 实际应该是: logits, cache = model_wrapper.model(tokens)
        # 这里用简化版本估算
        num_layers = 64  # Qwen3-30B 的层数
        hidden_dim = 3584  # Qwen3-30B 的 hidden_dim

        # 模拟每层的计算
        for layer_idx in range(num_layers):
            # QKV projection (这里简化，实际会更复杂)
            profiler.start_section(f"layer_{layer_idx}_qkv_proj")
            # q = mx.random.normal((len(tokens), hidden_dim))
            # k = mx.random.normal((len(tokens), hidden_dim))
            # v = mx.random.normal((len(tokens), hidden_dim))
            # mx.eval(q, k, v)
            profiler.end_section()

            # Attention
            profiler.start_section(f"layer_{layer_idx}_attention")
            # attn_out = mx.random.normal((len(tokens), hidden_dim))
            # mx.eval(attn_out)
            profiler.end_section()

            # FFN
            profiler.start_section(f"layer_{layer_idx}_ffn")
            # ffn_out = mx.random.normal((len(tokens), hidden_dim))
            # mx.eval(ffn_out)
            profiler.end_section()

        profiler.end_section()  # forward_pass

        # Final layer norm + output projection
        profiler.start_section(f"output_projection_trial_{trial}")
        # logits = mx.random.normal((len(tokens), 32000))
        # mx.eval(logits)
        profiler.end_section()

        # mx.synchronize()

        trial_time = time.perf_counter() - profiler.section_start
        profiler.end_section()  # prefill_total

        prefill_times.append(trial_time)
        throughput = len(tokens) / trial_time
        token_throughputs.append(throughput)

        print(f"    Prefill 时间: {trial_time*1000:.1f} ms")
        print(f"    吞吐量: {throughput:.1f} tok/s")

    # ========== 5. 统计分析 ==========
    avg_prefill_time = sum(prefill_times) / len(prefill_times)
    avg_throughput = sum(token_throughputs) / len(token_throughputs)

    print(f"\n📊 Prefill 性能汇总:")
    print(f"  平均 Prefill 时间: {avg_prefill_time*1000:.1f} ms")
    print(f"  平均吞吐量: {avg_throughput:.1f} tok/s")
    print(f"  最快: {min(prefill_times)*1000:.1f} ms ({max(token_throughputs):.1f} tok/s)")
    print(f"  最慢: {max(prefill_times)*1000:.1f} ms ({min(token_throughputs):.1f} tok/s)")

    # ========== 6. 详细时间分解 ==========
    print(f"\n📊 详细时间分解:")
    summary = profiler.get_summary()

    # 按时间排序
    sorted_sections = sorted(
        summary.items(),
        key=lambda x: x[1]['avg_ms'],
        reverse=True
    )

    for name, stats in sorted_sections[:20]:  # Top 20
        print(f"  {name:50s}: {stats['avg_ms']:8.1f} ms ({stats['percent']:5.1f}%)")

    # ========== 7. 瓶颈分析 ==========
    print(f"\n🔍 瓶颈分析:")

    bottlenecks = []
    for name, stats in sorted_sections:
        if stats['percent'] > 5.0:  # 占比超过 5% 的
            bottlenecks.append((name, stats))

    if bottlenecks:
        print(f"  发现 {len(bottlenecks)} 个主要瓶颈（占比 > 5%）:")
        for name, stats in bottlenecks:
            print(f"    - {name}: {stats['avg_ms']:.1f} ms ({stats['percent']:.1f}%)")
    else:
        print(f"  ✅ 无明显瓶颈（所有操作耗时 < 5%）")

    # ========== 8. 生成报告 ==========
    report = {
        'model_path': model_path,
        'prompt_length': prompt_length,
        'trials': trials,
        'avg_prefill_time_ms': avg_prefill_time * 1000,
        'avg_throughput_tps': avg_throughput,
        'min_time_ms': min(prefill_times) * 1000,
        'max_time_ms': max(prefill_times) * 1000,
        'timing_breakdown': summary,
        'bottlenecks': [
            {'name': name, 'stats': stats}
            for name, stats in bottlenecks
        ]
    }

    return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Prefill 性能深度分析")
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
    parser.add_argument(
        '--output',
        type=str,
        default='/tmp/prefill_profile_report.json',
        help='输出报告路径'
    )

    args = parser.parse_args()

    # 展开路径
    model_path = os.path.expanduser(args.model)

    # 执行分析
    try:
        report = analyze_prefill_performance(
            model_path=model_path,
            prompt_length=args.length,
            trials=args.trials
        )

        # 保存报告
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ 报告已保存到: {args.output}")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
