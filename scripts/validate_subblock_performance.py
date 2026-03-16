#!/usr/bin/env python3
"""
子 block 缓存性能模拟器

模拟不同方案的 CPU 开销（不含真实 KV Cache）：
1. 方案 A: 三级子 block 匹配（提议方案）
2. 方案 B: Approximate Skip（现有方案）

输出：每种方案在不同 prompt 长度下的理论 CPU 周期开销
"""

import time
import hashlib
from typing import Dict, List

# 配置
MAIN_BLOCK_SIZE = 256
SUB_BLOCK_SIZE = 32
TEST_PROMPT_LENGTHS = [116, 200, 300, 500, 1000]  # 各种长度

# CPU 周期开销估算（相对值）
HASH_COST = 100  # SHA256 哈希计算
DICT_LOOKUP_COST = 10  # 字典查找
ZERO_FILL_COST = 50  # 零填充操作


def simulate_three_level_matching(prompt_length: int) -> Dict:
    """
    模拟三级匹配的 CPU 开销

    三级匹配流程：
    1. 主 block 匹配（每个主 block 需要哈希 + 查找）
    2. 剩余 tokens 子 block 匹配（每个子 block 需要哈希 + 查找）
    3. 零填充（如果仍有未匹配）

    Returns:
        {
            'hash_operations': int,  # 哈希计算次数
            'dict_lookups': int,     # 字典查找次数
            'zero_fill_operations': int,  # 零填充次数
            'cpu_cycles': int,       # 理论 CPU 周期
        }
    """
    # 主 block 匹配
    num_main_blocks = prompt_length // MAIN_BLOCK_SIZE
    main_hash_ops = num_main_blocks
    main_lookups = num_main_blocks

    # 剩余 tokens
    remaining_tokens = prompt_length % MAIN_BLOCK_SIZE

    # 子 block 匹配
    num_sub_blocks = (remaining_tokens + SUB_BLOCK_SIZE - 1) // SUB_BLOCK_SIZE
    sub_hash_ops = num_sub_blocks
    sub_lookups = num_sub_blocks

    # 零填充（假设部分子 block 未命中需要零填充）
    # 保守估计：50% 的子 block 需要零填充
    zero_fill_ops = max(1, num_sub_blocks // 2)

    total_hash = main_hash_ops + sub_hash_ops
    total_lookups = main_lookups + sub_lookups

    cpu_cycles = (
        total_hash * HASH_COST +
        total_lookups * DICT_LOOKUP_COST +
        zero_fill_ops * ZERO_FILL_COST
    )

    return {
        'hash_operations': total_hash,
        'dict_lookups': total_lookups,
        'zero_fill_operations': zero_fill_ops,
        'cpu_cycles': cpu_cycles,
    }


def simulate_approximate_skip(prompt_length: int) -> Dict:
    """
    模拟 Approximate Skip 的 CPU 开销

    Approximate Skip 流程：
    1. 主 block 匹配（每个主 block 需要哈希 + 查找）
    2. 如果命中率 >= 95%，零填充未匹配部分

    Returns:
        {
            'hash_operations': int,
            'dict_lookups': int,
            'zero_fill_operations': int,
            'cpu_cycles': int,
        }
    """
    # 主 block 匹配
    num_main_blocks = (prompt_length + MAIN_BLOCK_SIZE - 1) // MAIN_BLOCK_SIZE
    hash_ops = num_main_blocks
    lookups = num_main_blocks

    # 剩余 tokens（假设需要零填充）
    remaining_tokens = prompt_length % MAIN_BLOCK_SIZE
    zero_fill_ops = 1 if remaining_tokens > 0 else 0

    cpu_cycles = (
        hash_ops * HASH_COST +
        lookups * DICT_LOOKUP_COST +
        zero_fill_ops * ZERO_FILL_COST
    )

    return {
        'hash_operations': hash_ops,
        'dict_lookups': lookups,
        'zero_fill_operations': zero_fill_ops,
        'cpu_cycles': cpu_cycles,
    }


def main():
    print("🔬 子 block 缓存性能模拟器")
    print("=" * 70)
    print(f"配置: MAIN_BLOCK_SIZE={MAIN_BLOCK_SIZE}, SUB_BLOCK_SIZE={SUB_BLOCK_SIZE}")
    print()

    results = []

    for length in TEST_PROMPT_LENGTHS:
        print(f"📊 测试 prompt 长度: {length} tokens")
        print("-" * 70)

        # 方案 A: 三级匹配
        result_a = simulate_three_level_matching(length)

        # 方案 B: Approximate Skip
        result_b = simulate_approximate_skip(length)

        # 对比
        overhead_ratio = result_a['cpu_cycles'] / result_b['cpu_cycles'] if result_b['cpu_cycles'] > 0 else 1.0

        print(f"  方案 A (三级子 block 匹配):")
        print(f"    哈希操作:   {result_a['hash_operations']} 次")
        print(f"    字典查找:   {result_a['dict_lookups']} 次")
        print(f"    零填充:     {result_a['zero_fill_operations']} 次")
        print(f"    CPU 周期:   {result_a['cpu_cycles']}")

        print(f"  方案 B (Approximate Skip):")
        print(f"    哈希操作:   {result_b['hash_operations']} 次")
        print(f"    字典查找:   {result_b['dict_lookups']} 次")
        print(f"    零填充:     {result_b['zero_fill_operations']} 次")
        print(f"    CPU 周期:   {result_b['cpu_cycles']}")

        print(f"  开销比 (A/B): {overhead_ratio:.2f}x")

        if overhead_ratio > 1.2:
            print(f"  ❌ 方案 A 开销显著高于方案 B")
        elif overhead_ratio > 1.0:
            print(f"  ⚠️ 方案 A 开销略高于方案 B")
        else:
            print(f"  ✅ 方案 A 开销可接受")

        print()

        results.append({
            'length': length,
            'method_a': result_a,
            'method_b': result_b,
            'overhead_ratio': overhead_ratio
        })

    # 总结
    print("=" * 70)
    print("📊 总结")
    print("=" * 70)

    avg_overhead = sum(r['overhead_ratio'] for r in results) / len(results)
    print(f"平均开销比: {avg_overhead:.2f}x")
    print()

    if avg_overhead > 1.5:
        print("❌ 结论：三级匹配开销显著高于 Approximate Skip（>1.5x）")
        print("   不推荐实施，性能收益可能被开销抵消")
    elif avg_overhead > 1.2:
        print("⚠️ 结论：三级匹配开销较高（1.2-1.5x）")
        print("   需要性能收益显著超过 20% 才值得实施")
    elif avg_overhead > 1.0:
        print("⚠️ 结论：三级匹配开销略高（1.0-1.2x）")
        print("   如果性能收益明显（如 TTFT 提升 > 50%），可以考虑")
    else:
        print("✅ 结论：三级匹配开销可接受")
        print("   建议进一步验证实际性能收益")


if __name__ == "__main__":
    main()
