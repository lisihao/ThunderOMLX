#!/usr/bin/env python3
"""
碎片分布分析脚本
分析真实工作负载下不同 block_size 的碎片分布统计
"""

import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# 配置
TEST_PROMPT_LENGTHS = [50, 100, 116, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000]
BLOCK_SIZES = [32, 64, 128, 256]

# 模拟真实工作负载的权重分布（基于实际 LLM 使用模式）
REALISTIC_WEIGHTS = {
    50: 5,    # 短对话
    100: 15,  # 常见短 prompt
    116: 8,   # 特定长度
    200: 20,  # 中等长度
    250: 12,
    300: 15,  # 长对话
    400: 10,
    500: 6,
    750: 4,
    1000: 3,
    1500: 1,
    2000: 1,  # 非常长的 prompt（罕见）
}


def find_thunderomlx_logs() -> Optional[List[str]]:
    """尝试查找 ThunderOMLX 日志文件"""
    log_paths = [
        "/tmp/thunderomlx.log",
        "./thunderomlx.log",
        "./logs/thunderomlx.log",
        os.path.expanduser("~/.thunderomlx/logs/requests.log"),
    ]

    for path in log_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.readlines()
            except (IOError, PermissionError):
                continue
    return None


def extract_prompt_lengths_from_logs(log_lines: List[str]) -> List[int]:
    """从日志中提取 prompt 长度"""
    lengths = []
    # 匹配常见的日志格式
    patterns = [
        r'prompt.*?length[:\s]+(\d+)',
        r'tokens[:\s]+(\d+)',
        r'input_length[:\s]+(\d+)',
        r'seq_len[:\s]+(\d+)',
        r'"length":\s*(\d+)',
    ]

    for line in log_lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    length = int(match.group(1))
                    if 1 <= length <= 100000:  # 合理范围
                        lengths.append(length)
                except ValueError:
                    continue
                break

    return lengths


def generate_weighted_lengths() -> List[int]:
    """生成基于真实权重的 prompt 长度列表"""
    lengths = []
    for length, weight in REALISTIC_WEIGHTS.items():
        lengths.extend([length] * weight)
    return lengths


def calculate_fragment(prompt_length: int, block_size: int) -> int:
    """计算单个 prompt 的碎片大小"""
    return prompt_length % block_size


def calculate_blocks_used(prompt_length: int, block_size: int) -> int:
    """计算使用的完整 block 数量"""
    return prompt_length // block_size


def analyze_block_size(prompt_lengths: List[int], block_size: int) -> Dict:
    """分析特定 block_size 的碎片分布"""
    fragments = []
    total_tokens = 0
    total_fragments = 0
    total_blocks = 0

    for length in prompt_lengths:
        fragment = calculate_fragment(length, block_size)
        blocks = calculate_blocks_used(length, block_size)

        fragments.append(fragment)
        total_tokens += length
        total_fragments += fragment
        total_blocks += blocks

    # 计算统计数据
    avg_fragment = total_fragments / len(prompt_lengths) if prompt_lengths else 0
    fragment_ratio = (total_fragments / total_tokens * 100) if total_tokens > 0 else 0
    hit_rate = 100 - fragment_ratio

    # 生成碎片分布直方图
    histogram = generate_histogram(fragments, block_size - 1)

    return {
        'block_size': block_size,
        'avg_fragment': avg_fragment,
        'fragment_ratio': fragment_ratio,
        'hit_rate': hit_rate,
        'total_tokens': total_tokens,
        'total_blocks': total_blocks,
        'total_fragments': total_fragments,
        'histogram': histogram,
        'fragments': fragments,
    }


def generate_histogram(fragments: List[int], max_value: int) -> Dict[str, Tuple[int, float]]:
    """生成碎片分布直方图数据"""
    if not fragments:
        return {}

    # 根据最大值动态确定 bin 数量和范围
    if max_value <= 32:
        bin_size = 8
    elif max_value <= 64:
        bin_size = 16
    elif max_value <= 128:
        bin_size = 32
    else:
        bin_size = 64

    bins = {}
    bin_count = (max_value // bin_size) + 1

    for i in range(bin_count):
        start = i * bin_size
        end = min((i + 1) * bin_size - 1, max_value)
        key = f"{start}-{end}"
        bins[key] = 0

    # 统计每个 bin 的数量
    for fragment in fragments:
        bin_index = fragment // bin_size
        start = bin_index * bin_size
        end = min((bin_index + 1) * bin_size - 1, max_value)
        key = f"{start}-{end}"
        if key in bins:
            bins[key] += 1

    # 转换为百分比
    total = len(fragments)
    result = {}
    for key, count in bins.items():
        result[key] = (count, count / total * 100 if total > 0 else 0)

    return result


def render_bar(percentage: float, width: int = 10) -> str:
    """渲染文本进度条"""
    filled = int(percentage / 100 * width)
    empty = width - filled
    return '█' * filled + '░' * empty


def print_analysis(results: List[Dict]) -> None:
    """打印分析结果"""
    print("\n📊 真实碎片分布统计")
    print("==================\n")

    for result in results:
        bs = result['block_size']
        print(f"Block Size: {bs}")
        print("-" * (12 + len(str(bs))))
        print(f"  平均碎片: {result['avg_fragment']:.1f} tokens")
        print(f"  碎片占比: {result['fragment_ratio']:.1f}%")
        print(f"  理论命中率: {result['hit_rate']:.1f}%")
        print()
        print("  碎片分布:")

        histogram = result['histogram']
        for key in sorted(histogram.keys(), key=lambda x: int(x.split('-')[0])):
            count, percentage = histogram[key]
            bar = render_bar(percentage)
            print(f"  {key:>8}: {bar} {percentage:.0f}%")

        print()

    # 打印总结
    print("\n📊 总结")
    print("======")

    best = min(results, key=lambda x: x['fragment_ratio'])
    worst = max(results, key=lambda x: x['fragment_ratio'])

    print(f"最佳 block_size: {best['block_size']} (碎片占比: {best['fragment_ratio']:.1f}%)")
    print(f"最差 block_size: {worst['block_size']} (碎片占比: {worst['fragment_ratio']:.1f}%)")

    # 额外统计
    print(f"\n详细对比:")
    print(f"{'Block Size':<12} {'平均碎片':<12} {'碎片占比':<12} {'命中率':<12} {'效率评分'}")
    print("-" * 60)

    for result in sorted(results, key=lambda x: x['hit_rate'], reverse=True):
        bs = result['block_size']
        # 效率评分：命中率越高越好，但也要考虑 block_size 的灵活性
        efficiency = result['hit_rate'] * (1 - bs / 512)  # 惩罚过大的 block_size
        print(f"{bs:<12} {result['avg_fragment']:<12.1f} {result['fragment_ratio']:<12.1f}% {result['hit_rate']:<12.1f}% {efficiency:.2f}")


def analyze_real_distribution() -> None:
    """分析真实工作负载分布"""
    print("🔍 分析模式: 真实工作负载模拟")
    print(f"   Prompt 长度样本: {len(generate_weighted_lengths())} 个")
    print(f"   唯一长度值: {len(TEST_PROMPT_LENGTHS)} 种")

    # 生成加权长度
    prompt_lengths = generate_weighted_lengths()

    # 分析每个 block_size
    results = []
    for bs in BLOCK_SIZES:
        result = analyze_block_size(prompt_lengths, bs)
        results.append(result)

    print_analysis(results)


def analyze_uniform_distribution() -> None:
    """分析均匀分布（每个长度权重相同）"""
    print("🔍 分析模式: 均匀分布")
    print(f"   Prompt 长度样本: {len(TEST_PROMPT_LENGTHS)} 种")

    results = []
    for bs in BLOCK_SIZES:
        result = analyze_block_size(TEST_PROMPT_LENGTHS, bs)
        results.append(result)

    print_analysis(results)


def analyze_custom_distribution(lengths: List[int], source: str = "自定义") -> None:
    """分析自定义分布"""
    print(f"🔍 分析模式: {source}")
    print(f"   Prompt 样本数: {len(lengths)}")

    results = []
    for bs in BLOCK_SIZES:
        result = analyze_block_size(lengths, bs)
        results.append(result)

    print_analysis(results)


def export_statistics(results: List[Dict], output_file: str = "fragment_stats.csv") -> None:
    """导出统计数据到 CSV 文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("block_size,avg_fragment,fragment_ratio,hit_rate,total_tokens,total_blocks\n")
            for r in results:
                f.write(f"{r['block_size']},{r['avg_fragment']:.2f},{r['fragment_ratio']:.2f},{r['hit_rate']:.2f},{r['total_tokens']},{r['total_blocks']}\n")
        print(f"\n✅ 统计数据已导出到: {output_file}")
    except IOError as e:
        print(f"\n⚠️  导出失败: {e}")


def print_fragment_details():
    """打印每个 prompt 长度在各 block_size 下的碎片详情"""
    print("\n📋 各 Prompt 长度碎片详情")
    print("=" * 70)
    print(f"{'Prompt Length':<15}", end="")
    for bs in BLOCK_SIZES:
        print(f"BS={bs:<10}", end="")
    print()
    print("-" * 70)

    for length in TEST_PROMPT_LENGTHS:
        print(f"{length:<15}", end="")
        for bs in BLOCK_SIZES:
            fragment = calculate_fragment(length, bs)
            blocks = calculate_blocks_used(length, bs)
            wasted_pct = fragment / length * 100
            print(f"{fragment}({wasted_pct:.0f}%){' ':<4}", end="")
        print()


def main():
    """主函数"""
    print("=" * 60)
    print("       ThunderOMLX 碎片分布分析工具")
    print("=" * 60)

    # 尝试从日志读取
    log_lines = find_thunderomlx_logs()

    if log_lines:
        print("\n📁 发现 ThunderOMLX 日志文件")
        log_lengths = extract_prompt_lengths_from_logs(log_lines)
        if log_lengths:
            print(f"   从日志中提取到 {len(log_lengths)} 个 prompt 长度")
            analyze_custom_distribution(log_lengths, "日志文件")
        else:
            print("   ⚠️  未能从日志中提取有效长度，使用模拟数据")
            analyze_real_distribution()
    else:
        print("\n📁 未找到 ThunderOMLX 日志文件，使用模拟数据")
        analyze_real_distribution()

    # 额外分析
    print("\n" + "=" * 60)
    print("       补充分析: 均匀分布对比")
    print("=" * 60)
    analyze_uniform_distribution()

    # 详细碎片表
    print_fragment_details()

    # 推荐配置
    print("\n" + "=" * 60)
    print("       推荐配置")
    print("=" * 60)
    print("""
基于分析结果推荐:
- 对于短 prompt 场景 (< 500 tokens): block_size=64
- 对于混合场景: block_size=128 (平衡碎片和效率)
- 对于长 prompt 场景 (> 1000 tokens): block_size=256

实际选择还需考虑:
1. GPU 内存大小
2. KV Cache 存储开销
3. 缓存驱逐策略
""")


if __name__ == "__main__":
    main()
