#!/usr/bin/env python3
"""
简单的 Profiling 功能测试
"""

import os
import sys

# Add src to path
sys.path.insert(0, 'src')

# 启用 profiling
os.environ['OMLX_ENABLE_PROFILING'] = 'true'

from omlx.profiling import get_global_profiler, print_profiling_stats
import time


def do_work(duration_ms):
    """执行一些实际计算（而不是sleep）"""
    import math
    end_time = time.perf_counter() + (duration_ms / 1000.0)
    count = 0
    while time.perf_counter() < end_time:
        # 做一些计算
        math.sqrt(count * 3.14159)
        count += 1


def simulate_prefill():
    """模拟 Prefill 流程以测试 profiling"""

    profiler = get_global_profiler()

    print("=" * 80)
    print("测试 Profiling 框架")
    print("=" * 80)
    print(f"Profiling enabled: {profiler.enabled}\n")

    # 测试 1: 使用 record() 直接记录
    print("测试 1: 使用 record() 方法")
    profiler.record("test.operation1", 10.5)
    profiler.record("test.operation2", 25.3)
    profiler.record("test.operation1", 12.7)

    # 测试 2: 使用 context manager
    print("测试 2: 使用 context manager")
    with profiler.section("test.context_manager"):
        do_work(10)

    # 测试 3: 使用 start/end
    print("测试 3: 使用 start/end")
    profiler.start("test.start_end")
    do_work(15)
    profiler.end("test.start_end")

    # 模拟完整的 Prefill 流程
    print("测试 4: 模拟完整 Prefill 流程")
    profiler.start("prefill.total")

    # 1. Prepare inputs
    profiler.start("prefill.prepare_inputs")
    do_work(10)  # 模拟 10ms
    profiler.end("prefill.prepare_inputs")

    # 2. Model forward
    profiler.start("prefill.model_forward")
    do_work(100)  # 模拟 100ms
    profiler.end("prefill.model_forward")

    # 3. Cache ops
    profiler.start("prefill.cache_ops")
    do_work(5)  # 模拟 5ms
    profiler.end("prefill.cache_ops")

    # 4. Final step
    with profiler.section("prefill.final_step"):
        do_work(20)  # 模拟 20ms

    # 5. Synchronize
    with profiler.section("prefill.synchronize"):
        do_work(3)  # 模拟 3ms

    profiler.end("prefill.total")

    # 打印统计
    print("\n" + "=" * 80)
    print("📊 Profiling 统计")
    print("=" * 80 + "\n")

    print_profiling_stats(top_n=20, min_percent=0.1)

    # 获取详细数据
    stats = profiler.get_stats()

    print("\n" + "=" * 80)
    print("🔍 瓶颈分析")
    print("=" * 80 + "\n")

    bottlenecks = [
        (name, op_stats)
        for name, op_stats in stats['top_operations']
        if op_stats['percent'] > 5.0
    ]

    if bottlenecks:
        print(f"发现 {len(bottlenecks)} 个主要瓶颈（占比 > 5%）:\n")
        for name, op_stats in bottlenecks:
            print(f"  ⚠️  {name}")
            print(f"      - 平均时间: {op_stats['avg_ms']:.2f} ms")
            print(f"      - 占比: {op_stats['percent']:.1f}%")
            print()

    # 保存到文件
    import json
    with open('/tmp/profiling_test.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("✅ 测试完成！")
    print("📄 详细数据保存到: /tmp/profiling_test.json")


if __name__ == "__main__":
    simulate_prefill()
