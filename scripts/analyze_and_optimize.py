"""
Adaptive Cache Optimizer - 模式分析与优化建议

读取数据库中的推理数据，分析模式，生成优化建议。

基于 Phase 2 设计：模式分析引擎
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def analyze_prompt_distribution(aco: AdaptiveCacheOptimizer, agent_id: str) -> Dict:
    """
    分析 prompt 长度分布

    Returns:
        {
            'p50': int,
            'p90': int,
            'p99': int,
            'min': int,
            'max': int,
            'avg': float,
        }
    """
    import sqlite3
    import numpy as np

    with aco._get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT total_prompt_length
            FROM agent_metrics
            WHERE agent_id = ?
            ORDER BY total_prompt_length
            """,
            (agent_id,)
        )
        lengths = [row[0] for row in cursor.fetchall()]

    if not lengths:
        return {}

    lengths_array = np.array(lengths)

    return {
        'p50': int(np.percentile(lengths_array, 50)),
        'p90': int(np.percentile(lengths_array, 90)),
        'p99': int(np.percentile(lengths_array, 99)),
        'min': int(np.min(lengths_array)),
        'max': int(np.max(lengths_array)),
        'avg': float(np.mean(lengths_array)),
    }


def analyze_padding_by_block_size(
    aco: AdaptiveCacheOptimizer,
    agent_id: str,
    candidate_block_sizes: List[int]
) -> Dict[int, Dict]:
    """
    分析不同 block_size 下的 padding overhead

    Args:
        aco: ACO 实例
        agent_id: Agent 标识
        candidate_block_sizes: 候选 block sizes (e.g. [64, 128, 256])

    Returns:
        {
            64: {'avg_padding_overhead': 2.5, 'p95_padding_overhead': 5.0},
            128: {'avg_padding_overhead': 10.0, 'p95_padding_overhead': 15.0},
            ...
        }
    """
    import sqlite3
    import numpy as np

    # 获取所有 total_prompt_length
    with aco._get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT total_prompt_length
            FROM agent_metrics
            WHERE agent_id = ?
            """,
            (agent_id,)
        )
        prompt_lengths = [row[0] for row in cursor.fetchall()]

    if not prompt_lengths:
        return {}

    results = {}

    for block_size in candidate_block_sizes:
        padding_overheads = []

        for prompt_length in prompt_lengths:
            # 计算 padding tokens
            remainder = prompt_length % block_size
            if remainder == 0:
                padding_tokens = 0
            else:
                padding_needed = block_size - remainder
                # 最多 padding 64 tokens
                padding_tokens = padding_needed if padding_needed <= 64 else 0

            # 计算 padding overhead (%)
            if prompt_length > 0:
                padding_overhead = (padding_tokens / prompt_length) * 100
                padding_overheads.append(padding_overhead)

        if padding_overheads:
            padding_array = np.array(padding_overheads)
            results[block_size] = {
                'avg_padding_overhead': float(np.mean(padding_array)),
                'p95_padding_overhead': float(np.percentile(padding_array, 95)),
            }

    return results


def recommend_block_size(
    padding_analysis: Dict[int, Dict],
    prompt_distribution: Dict
) -> Tuple[int, str]:
    """
    推荐最优 block_size

    Returns:
        (推荐的 block_size, 推荐原因)
    """
    if not padding_analysis:
        return None, "无足够数据"

    # 按平均 padding overhead 排序
    sorted_blocks = sorted(
        padding_analysis.items(),
        key=lambda x: x[1]['avg_padding_overhead']
    )

    best_block_size = sorted_blocks[0][0]
    best_padding = sorted_blocks[0][1]['avg_padding_overhead']

    # 生成推荐原因
    reason = f"平均 padding overhead 最低 ({best_padding:.1f}%)"

    # 检查 P95 是否也较低
    p95_padding = sorted_blocks[0][1]['p95_padding_overhead']
    if p95_padding < 5.0:
        reason += f"，P95 仅 {p95_padding:.1f}%"

    return best_block_size, reason


def calculate_optimization_impact(
    aco: AdaptiveCacheOptimizer,
    agent_id: str,
    current_block_size: int,
    recommended_block_size: int
) -> Dict:
    """
    计算优化影响

    Returns:
        {
            'padding_reduction_pct': float,  # padding 减少百分比
            'estimated_speedup_pct': float,  # 预计加速百分比
            'annual_token_savings': int,     # 年节省 tokens (假设每天 100 次推理)
        }
    """
    import sqlite3
    import numpy as np

    # 获取平均 prompt 长度
    with aco._get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT AVG(total_prompt_length) AS avg_len
            FROM agent_metrics
            WHERE agent_id = ?
            """,
            (agent_id,)
        )
        avg_prompt_length = cursor.fetchone()[0]

    if not avg_prompt_length:
        return {}

    # 计算当前和优化后的 padding
    def calc_avg_padding(block_size):
        remainder = int(avg_prompt_length) % block_size
        if remainder == 0:
            return 0
        padding_needed = block_size - remainder
        return padding_needed if padding_needed <= 64 else 0

    current_padding = calc_avg_padding(current_block_size)
    recommended_padding = calc_avg_padding(recommended_block_size)

    # padding 减少百分比
    if current_padding > 0:
        padding_reduction_pct = ((current_padding - recommended_padding) / current_padding) * 100
    else:
        padding_reduction_pct = 0.0

    # 预计加速（简化模型：padding 减少 → prefill 加速）
    # 假设 padding 占 prefill 时间的 20%
    estimated_speedup_pct = padding_reduction_pct * 0.2

    # 年节省 tokens (假设每天 100 次推理)
    daily_inferences = 100
    annual_token_savings = int(
        (current_padding - recommended_padding) * daily_inferences * 365
    )

    return {
        'padding_reduction_pct': padding_reduction_pct,
        'estimated_speedup_pct': estimated_speedup_pct,
        'annual_token_savings': annual_token_savings,
    }


def generate_markdown_report(
    agent_id: str,
    prompt_distribution: Dict,
    padding_analysis: Dict[int, Dict],
    recommended_block_size: int,
    recommendation_reason: str,
    optimization_impact: Dict,
    current_block_size: int
) -> str:
    """生成 Markdown 格式的优化报告"""

    report = f"""## Agent: {agent_id}

### Prompt 长度分布

| 指标 | 值 |
|------|-----|
| 平均 | {prompt_distribution.get('avg', 0):.0f} tokens |
| P50 | {prompt_distribution.get('p50', 0)} tokens |
| P90 | {prompt_distribution.get('p90', 0)} tokens |
| P99 | {prompt_distribution.get('p99', 0)} tokens |
| 最小 | {prompt_distribution.get('min', 0)} tokens |
| 最大 | {prompt_distribution.get('max', 0)} tokens |

### Padding Overhead 分析

| Block Size | 平均 Padding | P95 Padding |
|------------|--------------|-------------|
"""

    for block_size in sorted(padding_analysis.keys()):
        data = padding_analysis[block_size]
        marker = " ✅ **推荐**" if block_size == recommended_block_size else ""
        marker += " (当前)" if block_size == current_block_size else ""
        report += f"| {block_size}{marker} | {data['avg_padding_overhead']:.1f}% | {data['p95_padding_overhead']:.1f}% |\n"

    report += f"""
### 优化建议

- **推荐 block_size**: `{recommended_block_size}`
- **当前 block_size**: `{current_block_size}`
- **推荐原因**: {recommendation_reason}

### 优化影响

"""

    if optimization_impact:
        report += f"""| 指标 | 值 |
|------|-----|
| Padding 减少 | {optimization_impact.get('padding_reduction_pct', 0):.1f}% |
| 预计加速 | {optimization_impact.get('estimated_speedup_pct', 0):.1f}% |
| 年节省 tokens | {optimization_impact.get('annual_token_savings', 0):,} |

"""

    if recommended_block_size != current_block_size:
        report += f"""### 如何应用

修改配置文件，将 `{agent_id}` 的 `block_size` 从 `{current_block_size}` 改为 `{recommended_block_size}`：

```python
# 在 Scheduler 初始化或配置中
agent_configs = {{
    "{agent_id}": {{
        "block_size": {recommended_block_size},  # 优化前: {current_block_size}
    }},
}}
```

"""
    else:
        report += "### 当前配置已是最优 ✅\n\n"

    report += "---\n\n"

    return report


def main(db_path: str):
    """
    主函数：分析数据库并生成优化建议
    """
    print("=" * 70)
    print("Adaptive Cache Optimizer - 模式分析与优化建议")
    print("=" * 70)

    # 初始化 ACO
    aco = AdaptiveCacheOptimizer(db_path)

    # 获取所有 agent IDs
    import sqlite3
    with aco._get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT agent_id FROM agent_metrics")
        agent_ids = [row[0] for row in cursor.fetchall()]

    if not agent_ids:
        print("\n❌ 数据库为空，请先运行 init_adaptive_cache_data.py")
        return

    print(f"\n找到 {len(agent_ids)} 个 agents:")
    for agent_id in agent_ids:
        print(f"  - {agent_id}")

    # 候选 block sizes
    candidate_block_sizes = [64, 128, 256]

    # 完整报告
    full_report = f"""# Adaptive Cache Optimizer - 优化报告

**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据库**: {db_path}

---

"""

    # Agent 配置映射（从 init script 读取）
    agent_current_configs = {
        "chief-of-staff": {"block_size": 128},
        "product-strategist": {"block_size": 64},
        "ux-designer": {"block_size": 256},
        "ai-engineer": {"block_size": 256},
    }

    # 为每个 agent 生成分析
    for agent_id in sorted(agent_ids):
        print(f"\n{'='*70}")
        print(f"分析 Agent: {agent_id}")
        print(f"{'='*70}")

        # 1. Prompt 长度分布
        print("\n1️⃣ 分析 Prompt 长度分布...")
        prompt_dist = analyze_prompt_distribution(aco, agent_id)
        print(f"   平均: {prompt_dist.get('avg', 0):.0f} tokens, "
              f"P90: {prompt_dist.get('p90', 0)} tokens, "
              f"P99: {prompt_dist.get('p99', 0)} tokens")

        # 2. Padding 分析
        print("\n2️⃣ 分析不同 block_size 的 Padding Overhead...")
        padding_analysis = analyze_padding_by_block_size(
            aco, agent_id, candidate_block_sizes
        )

        for block_size, data in sorted(padding_analysis.items()):
            print(f"   block_size={block_size}: "
                  f"平均 {data['avg_padding_overhead']:.1f}%, "
                  f"P95 {data['p95_padding_overhead']:.1f}%")

        # 3. 推荐 block_size
        print("\n3️⃣ 生成优化建议...")
        recommended_block_size, reason = recommend_block_size(
            padding_analysis, prompt_dist
        )
        print(f"   推荐 block_size: {recommended_block_size}")
        print(f"   原因: {reason}")

        # 4. 计算优化影响
        current_block_size = agent_current_configs.get(agent_id, {}).get("block_size", 128)

        if recommended_block_size != current_block_size:
            print("\n4️⃣ 计算优化影响...")
            impact = calculate_optimization_impact(
                aco, agent_id, current_block_size, recommended_block_size
            )
            print(f"   Padding 减少: {impact.get('padding_reduction_pct', 0):.1f}%")
            print(f"   预计加速: {impact.get('estimated_speedup_pct', 0):.1f}%")
            print(f"   年节省 tokens: {impact.get('annual_token_savings', 0):,}")
        else:
            print("\n✅ 当前配置已是最优")
            impact = {}

        # 5. 生成 Markdown 报告
        agent_report = generate_markdown_report(
            agent_id,
            prompt_dist,
            padding_analysis,
            recommended_block_size,
            reason,
            impact,
            current_block_size
        )

        full_report += agent_report

    # 保存报告
    report_path = Path(db_path).parent / "optimization_report.md"
    report_path.write_text(full_report, encoding='utf-8')

    print(f"\n{'='*70}")
    print(f"✅ 分析完成！")
    print(f"{'='*70}")
    print(f"\n📄 完整报告已保存到: {report_path}")
    print(f"\n下一步:")
    print(f"  1. 查看报告: cat {report_path}")
    print(f"  2. 应用优化: 修改 Agent 配置，更新 block_size")
    print(f"  3. 验证效果: 运行推理，观察性能提升")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析 Adaptive Cache Optimizer 数据并生成优化建议")
    parser.add_argument(
        "--db-path",
        default="~/.cache/thunderomlx/adaptive_cache.db",
        help="数据库路径"
    )

    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()

    if not db_path.exists():
        print(f"\n❌ 数据库不存在: {db_path}")
        print(f"请先运行: python scripts/init_adaptive_cache_data.py")
        sys.exit(1)

    main(str(db_path))
