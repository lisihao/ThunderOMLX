#!/usr/bin/env python3
"""
分析 Profiling 结果，识别 1.5s 开销的来源
"""
import re
from collections import defaultdict

def parse_profile_line(line):
    """解析 cProfile 输出的一行"""
    # 格式: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    match = re.match(r'\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)', line)
    if match:
        return {
            'ncalls': match.group(1),
            'tottime': float(match.group(2)),
            'tottime_percall': float(match.group(3)),
            'cumtime': float(match.group(4)),
            'cumtime_percall': float(match.group(5)),
            'function': match.group(6).strip()
        }
    return None

def categorize_function(func_name):
    """将函数分类"""
    if 'forward' in func_name.lower() or 'model' in func_name.lower():
        return 'model_forward'
    elif 'cache' in func_name.lower() or 'ssd' in func_name.lower():
        return 'cache_io'
    elif 'pilot' in func_name.lower() or 'context' in func_name.lower():
        return 'context_pilot'
    elif 'tensor' in func_name.lower() or 'array' in func_name.lower():
        return 'tensor_ops'
    elif 'lock' in func_name.lower() or 'thread' in func_name.lower():
        return 'synchronization'
    elif 'queue' in func_name.lower():
        return 'queue_ops'
    elif '__' in func_name or 'built-in' in func_name:
        return 'python_builtin'
    else:
        return 'other'

def analyze_profile(profile_file):
    """分析 profile 文件"""
    print(f"\n{'=' * 70}")
    print(f"📊 分析: {profile_file}")
    print(f"{'=' * 70}\n")

    categories = defaultdict(float)
    top_functions = []

    with open(profile_file, 'r') as f:
        lines = f.readlines()

    # 跳过头部
    in_data = False
    for line in lines:
        if 'ncalls' in line and 'tottime' in line:
            in_data = True
            continue

        if not in_data:
            continue

        if line.strip() == '' or line.startswith('---'):
            break

        entry = parse_profile_line(line)
        if entry and entry['tottime'] > 0.01:  # 只看 > 10ms 的函数
            category = categorize_function(entry['function'])
            categories[category] += entry['tottime']

            if len(top_functions) < 20:
                top_functions.append(entry)

    # 打印分类统计
    print("🔍 按类别分组（tottime > 10ms）:\n")
    print(f"{'类别':<20} {'总时间 (s)':<12} {'占比':<10}")
    print("-" * 50)

    total_time = sum(categories.values())
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    for cat, time_spent in sorted_cats:
        percentage = (time_spent / total_time * 100) if total_time > 0 else 0
        print(f"{cat:<20} {time_spent:<12.2f} {percentage:>6.1f}%")

    print(f"\n{'总计':<20} {total_time:<12.2f} 100.0%")

    # 打印 Top 函数
    print(f"\n{'=' * 70}")
    print("🔝 Top 15 耗时函数 (by tottime):\n")
    print(f"{'Tottime (s)':<12} {'Calls':<10} {'Function':<50}")
    print("-" * 70)

    for i, entry in enumerate(top_functions[:15]):
        func_short = entry['function'][:50]
        print(f"{entry['tottime']:<12.2f} {entry['ncalls']:<10} {func_short}")

    # 识别"其他开销"
    print(f"\n{'=' * 70}")
    print("💡 '其他开销' 分析:\n")

    model_forward_time = categories.get('model_forward', 0)
    cache_io_time = categories.get('cache_io', 0)
    other_time = total_time - model_forward_time - cache_io_time

    print(f"   模型 Forward: {model_forward_time:.2f}s")
    print(f"   Cache I/O: {cache_io_time:.2f}s")
    print(f"   其他开销: {other_time:.2f}s")
    print()

    # 其他开销的细分
    other_breakdown = {k: v for k, v in sorted_cats if k not in ['model_forward', 'cache_io']}
    if other_breakdown:
        print("   其他开销细分:")
        for cat, time_spent in other_breakdown:
            print(f"      - {cat}: {time_spent:.2f}s ({time_spent/other_time*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print("🎯 优化建议:\n")

    # 基于分析给出建议
    if categories.get('tensor_ops', 0) > 0.5:
        print("   ⚠️  Tensor 操作耗时较高（> 0.5s）")
        print("      → 检查是否有不必要的拷贝")
        print("      → 考虑 in-place 操作")

    if categories.get('synchronization', 0) > 0.3:
        print("   ⚠️  同步/锁操作耗时较高（> 0.3s）")
        print("      → 检查锁的使用是否必要")

    if categories.get('python_builtin', 0) > 0.5:
        print("   ⚠️  Python 内置操作耗时较高（> 0.5s）")
        print("      → 考虑使用 Cython/Numba 加速")

    if categories.get('other', 0) > 1.0:
        print("   ⚠️  未分类的其他开销较高（> 1.0s）")
        print("      → 需要深入分析 Top 函数列表")

if __name__ == '__main__':
    print("=" * 70)
    print("🔍 Prefill Profiling 结果分析")
    print("=" * 70)

    # 分析两个文件
    analyze_profile('/tmp/prefill_profile_tottime.txt')

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)
    print("\n下一步:")
    print("   1. 查看上面的分类统计")
    print("   2. 关注'其他开销'的细分")
    print("   3. 基于优化建议制定方案")
