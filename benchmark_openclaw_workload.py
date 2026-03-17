#!/usr/bin/env python3
"""
OpenClaw Workload Benchmark

测试真实的 OpenClaw 多 Agent 场景下的 Generation TPS
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np

from omlx import Engine, EngineConfig


def load_workload(workload_file: Path, sample_size: int = 100) -> List[Dict]:
    """加载 OpenClaw workload 样本"""
    workload = []
    with open(workload_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            workload.append(json.loads(line))
    return workload


def simulate_prompt(length: int) -> str:
    """生成指定长度的模拟 prompt"""
    # 使用重复的文本来模拟，实际 token 数会接近指定长度
    base_text = "This is a test prompt for benchmark. "
    repeat_count = length // len(base_text.split()) + 1
    return (base_text * repeat_count)[:length * 5]  # 近似 5 chars/token


def run_openclaw_benchmark(
    model_path: str,
    workload_file: Path,
    sample_size: int = 100,
    output_length: int = 128,
):
    """运行 OpenClaw workload benchmark"""

    print("=" * 80)
    print("🚀 OpenClaw Workload Benchmark")
    print("=" * 80)

    # 加载 workload
    print(f"\n⏳ Loading workload from {workload_file}...")
    workload = load_workload(workload_file, sample_size)
    print(f"✅ Loaded {len(workload)} samples")

    # 统计 workload 特征
    prompt_lengths = [w['total_prompt_length'] for w in workload]
    print(f"\n📊 Workload Statistics:")
    print(f"   Prompt Length: {np.mean(prompt_lengths):.0f} ± {np.std(prompt_lengths):.0f} tokens")
    print(f"   Min: {np.min(prompt_lengths)}, Max: {np.max(prompt_lengths)}")
    print(f"   Agents: {set(w['agent_id'] for w in workload)}")

    # 初始化引擎
    print(f"\n⏳ Initializing engine with model {model_path}...")
    config = EngineConfig(
        model_path=model_path,
        max_num_batched_tokens=8192,
        completion_batch_size=32,
    )
    engine = Engine(config)
    print("✅ Engine initialized")

    # Warmup
    print("\n🔥 Warmup (3 requests)...")
    for i in range(3):
        prompt = simulate_prompt(500)
        _ = engine.generate(prompt, max_tokens=32)
        print(f"   Warmup {i+1}/3 ✓")

    # 运行 benchmark
    print(f"\n📊 Running benchmark ({len(workload)} samples)...")
    print("-" * 80)

    results = []
    total_prefill_time = 0
    total_generation_time = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, sample in enumerate(workload):
        # 生成模拟 prompt
        prompt_length = sample['total_prompt_length']
        prompt = simulate_prompt(prompt_length)

        # 执行推理
        start_time = time.perf_counter()

        response = engine.generate(
            prompt,
            max_tokens=output_length,
            return_timings=True,
        )

        end_time = time.perf_counter()

        # 收集统计
        if hasattr(response, 'prompt_tokens'):
            input_tokens = response.prompt_tokens
            output_tokens = response.generation_tokens
            prefill_time = response.prompt_time
            generation_time = response.generation_time
        else:
            # Fallback
            input_tokens = prompt_length
            output_tokens = output_length
            total_time = end_time - start_time
            prefill_time = total_time * 0.4  # 估算
            generation_time = total_time * 0.6

        total_prefill_time += prefill_time
        total_generation_time += generation_time
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        results.append({
            'agent_id': sample['agent_id'],
            'prompt_length': input_tokens,
            'output_tokens': output_tokens,
            'prefill_time': prefill_time,
            'generation_time': generation_time,
            'prefill_tps': input_tokens / prefill_time if prefill_time > 0 else 0,
            'generation_tps': output_tokens / generation_time if generation_time > 0 else 0,
        })

        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(workload)} ({(i+1)/len(workload)*100:.0f}%)")

    # 计算总体统计
    print("\n" + "=" * 80)
    print("📈 Benchmark Results")
    print("=" * 80)

    avg_prefill_tps = total_input_tokens / total_prefill_time if total_prefill_time > 0 else 0
    avg_generation_tps = total_output_tokens / total_generation_time if total_generation_time > 0 else 0

    generation_tps_list = [r['generation_tps'] for r in results if r['generation_tps'] > 0]
    generation_tps_std = np.std(generation_tps_list)

    print(f"\n📊 Overall Statistics ({len(workload)} samples):")
    print(f"   Total Input Tokens:  {total_input_tokens}")
    print(f"   Total Output Tokens: {total_output_tokens}")
    print(f"   Total Prefill Time:  {total_prefill_time:.2f}s")
    print(f"   Total Generation Time: {total_generation_time:.2f}s")
    print(f"\n⭐ Performance:")
    print(f"   Processing TPS:    {avg_prefill_tps:.1f} tok/s")
    print(f"   Generation TPS:    {avg_generation_tps:.1f} ± {generation_tps_std:.1f} tok/s ⭐")

    # 按 Agent 分组统计
    print(f"\n📊 By Agent Type:")
    agents = {}
    for r in results:
        agent_id = r['agent_id']
        if agent_id not in agents:
            agents[agent_id] = []
        agents[agent_id].append(r['generation_tps'])

    for agent_id in sorted(agents.keys()):
        tps_list = agents[agent_id]
        print(f"   {agent_id:15s}: {np.mean(tps_list):.1f} ± {np.std(tps_list):.1f} tok/s ({len(tps_list)} samples)")

    print("\n" + "=" * 80)

    # 保存结果
    output_file = Path("/tmp/benchmark_openclaw_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'model': model_path,
                'workload_file': str(workload_file),
                'sample_size': len(workload),
                'output_length': output_length,
            },
            'summary': {
                'avg_prefill_tps': avg_prefill_tps,
                'avg_generation_tps': avg_generation_tps,
                'generation_tps_std': generation_tps_std,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
            },
            'by_agent': {
                agent_id: {
                    'mean_tps': float(np.mean(tps_list)),
                    'std_tps': float(np.std(tps_list)),
                    'count': len(tps_list),
                }
                for agent_id, tps_list in agents.items()
            },
            'detailed_results': results,
        }, f, indent=2)

    print(f"💾 Results saved to: {output_file}")

    # 清理
    engine.stop()
    print("\n✅ Benchmark completed")

    return avg_generation_tps


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Workload Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/lisihao/models/qwen3.5-35b-mlx",
        help="Model path",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("/Users/lisihao/ThunderOMLX/openclaw-workload/openclaw-workload-7d.jsonl"),
        help="Workload file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        help="Generation length",
    )

    args = parser.parse_args()

    run_openclaw_benchmark(
        model_path=args.model,
        workload_file=args.workload,
        sample_size=args.sample_size,
        output_length=args.output_length,
    )


if __name__ == "__main__":
    main()
