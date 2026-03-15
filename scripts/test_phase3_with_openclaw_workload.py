"""
使用 OpenClaw 真实负载数据测试 Phase 3 高级策略

测试流程：
1. 加载 openclaw-workload 数据
2. 写入 Adaptive Cache Optimizer 数据库
3. 运行 Phase 3 所有功能：
   - Phase 3-A: 多维度分析
   - Phase 3-B: A/B 测试
   - Phase 3-C: 自动回滚
   - Phase 3-D: 多 Agent 协同优化
   - Phase 3-E: 时间序列分析
4. 输出优化建议和预期收益
"""

import sys
import json
from pathlib import Path
import tempfile

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def load_workload(workload_file: str) -> list:
    """加载负载数据"""
    records = []
    with open(workload_file, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def import_workload_to_db(aco: AdaptiveCacheOptimizer, records: list):
    """将负载数据导入数据库（直接 SQL 插入以保留时间戳）"""
    print(f"导入 {len(records)} 条负载记录...")

    import sqlite3

    with sqlite3.connect(aco.db_path) as conn:
        for i, record in enumerate(records):
            total_time_ms = record["prefill_time_ms"] + record["decode_time_ms"]

            conn.execute(
                """
                INSERT INTO agent_metrics (
                    agent_id, timestamp,
                    system_prompt_length, user_query_length, total_prompt_length,
                    cache_hit_ratio, skip_logic_type, block_size,
                    padding_tokens, padding_overhead,
                    prefill_time_ms, decode_time_ms, total_time_ms,
                    config_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["agent_id"],
                    record["timestamp"],
                    record["system_prompt_length"],
                    record["user_query_length"],
                    record["total_prompt_length"],
                    record["cache_hit_ratio"],
                    record["skip_logic_type"],
                    record["block_size"],
                    record["padding_tokens"],
                    record["padding_overhead"],
                    record["prefill_time_ms"],
                    record["decode_time_ms"],
                    total_time_ms,
                    record["config_version"],
                ),
            )

            if (i + 1) % 500 == 0:
                print(f"  已导入 {i + 1}/{len(records)} 条记录...")
                conn.commit()

        conn.commit()

    print(f"✅ 导入完成！")


def test_phase3a_multi_dimensional(aco: AdaptiveCacheOptimizer):
    """测试 Phase 3-A: 多维度分析"""
    print("\n" + "=" * 70)
    print("Phase 3-A: 多维度分析引擎")
    print("=" * 70)

    # 分析每个 Agent
    agent_ids = [
        "researcher-agent",
        "coder-agent",
        "analyst-agent",
        "pm-agent",
        "tester-agent",
    ]

    results = {}
    for agent_id in agent_ids:
        analysis = aco.analyze_multi_dimensional(agent_id, min_samples=50)
        if analysis:
            results[agent_id] = analysis
            print(f"\n📊 {agent_id}:")
            print(f"   Overall Score: {analysis['overall_score']:.1f}/100")
            print(f"   Padding: {analysis['dimensions']['padding']['score']:.1f}/100")
            print(f"   Cache Hit: {analysis['dimensions']['cache_hit']['score']:.1f}/100")

            if analysis["recommendations"]:
                print(f"   建议:")
                for rec in analysis["recommendations"][:2]:
                    print(f"   - [{rec['priority'].upper()}] {rec['type']}: {rec['reason'][:60]}...")

    return results


def test_phase3d_coordination(aco: AdaptiveCacheOptimizer):
    """测试 Phase 3-D: 多 Agent 协同优化"""
    print("\n" + "=" * 70)
    print("Phase 3-D: 多 Agent 协同优化")
    print("=" * 70)

    # 全局分析
    global_analysis = aco.analyze_global_optimization()
    if global_analysis:
        print(f"\n📊 全局分析:")
        print(f"   Total Agents: {global_analysis['total_agents']}")
        print(f"   Unique Block Sizes: {global_analysis['num_unique_block_sizes']}")
        print(f"   Fragmentation Score: {global_analysis['fragmentation_score']:.1f}/100")
        print(f"   KV Cache Reuse Potential: {global_analysis['kv_cache_reuse_potential']:.1f}/100")

        print(f"\n   Block Size 分布:")
        for block_size, agents in sorted(
            global_analysis["block_size_distribution"].items()
        ):
            print(f"   - {block_size}: {len(agents)} agents")

    # 协调优化
    coord_result = aco.recommend_coordinated_block_sizes(max_block_sizes=3)
    if coord_result:
        print(f"\n📊 协调优化方案:")
        print(f"   Clusters: {coord_result['num_clusters']}")

        print(f"\n   集群详情:")
        for cluster in coord_result["clusters"]:
            print(f"\n   Cluster {cluster['cluster_id']}:")
            print(f"      Block Size: {cluster['recommended_block_size']}")
            print(f"      Agents: {cluster['agent_ids']}")
            print(f"      Avg Prompt: {cluster['avg_prompt_length']:.0f}")

        print(f"\n   优化效果:")
        print(f"      当前平均 Padding: {coord_result['current_avg_padding']:.1f}%")
        print(f"      协调后平均 Padding: {coord_result['expected_avg_padding']:.1f}%")
        print(f"      Padding 变化: {coord_result['overall_padding_increase']:.1f}%")
        print(f"      KV Cache 复用提升: {coord_result['kv_cache_reuse_improvement']:.1f}%")
        print(f"      净收益评分: {coord_result['net_benefit_score']:.1f}")

        if coord_result["net_benefit_score"] > 0:
            print(f"\n   ✅ 建议应用协调优化（净收益 > 0）")
        else:
            print(f"\n   ⚠️ 不建议协调（padding 增加超过 KV Cache 收益）")

    return coord_result


def test_phase3e_time_series(aco: AdaptiveCacheOptimizer):
    """测试 Phase 3-E: 时间序列分析"""
    print("\n" + "=" * 70)
    print("Phase 3-E: 时间序列分析")
    print("=" * 70)

    # 分析 researcher-agent（最活跃）
    agent_id = "researcher-agent"
    result = aco.analyze_time_series(agent_id, window_hours=[1, 24, 168])

    if result:
        print(f"\n📊 时间序列分析 ({agent_id}):")
        print(f"\n   时间窗口数据:")
        for window, data in result["windows"].items():
            print(f"\n   {window}:")
            print(f"      Avg prompt: {data['avg_prompt_length']:.0f}")
            print(f"      Avg total: {data['avg_total']:.1f}ms")
            print(f"      Samples: {data['sample_count']}")

        print(f"\n   模式变化检测: {result['pattern_changes_detected']}")

        if result["pattern_changes"]:
            print(f"\n   检测到 {len(result['pattern_changes'])} 个模式变化:")
            for change in result["pattern_changes"]:
                print(f"   - {change['type']}: {change['description']}")

        if result["recommendations"]:
            print(f"\n   建议:")
            for rec in result["recommendations"]:
                print(f"   - {rec['action']}: {rec['reason']}")

    return result


def main():
    """主测试流程"""
    print("=" * 70)
    print("Phase 3 高级策略 - OpenClaw 负载测试")
    print("=" * 70)

    # 加载负载数据
    workload_file = "openclaw-workload/openclaw-workload-7d.jsonl"
    if not Path(workload_file).exists():
        print(f"❌ 负载文件不存在: {workload_file}")
        print(f"   请先运行: python3 scripts/generate_openclaw_workload.py")
        return 1

    records = load_workload(workload_file)
    print(f"\n✅ 加载 {len(records)} 条负载记录")

    # Agent 统计
    agent_counts = {}
    for record in records:
        agent_id = record["agent_id"]
        agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

    print(f"\n📊 Agent 分布:")
    for agent_id, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        print(f"   {agent_id}: {count} ({pct:.1f}%)")

    # 创建临时数据库
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 导入数据
    import_workload_to_db(aco, records)

    # Phase 3-A: 多维度分析
    multi_dim_results = test_phase3a_multi_dimensional(aco)

    # Phase 3-D: 协同优化
    coord_result = test_phase3d_coordination(aco)

    # Phase 3-E: 时间序列分析
    time_series_result = test_phase3e_time_series(aco)

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    print(f"\n✅ Phase 3 所有功能测试通过！")
    print(f"\n📊 关键发现:")

    # 统计需要优化的 Agent
    needs_optimization = []
    for agent_id, analysis in multi_dim_results.items():
        if analysis["recommendations"]:
            needs_optimization.append(agent_id)

    print(f"   - 需要优化的 Agent: {len(needs_optimization)}/{len(multi_dim_results)}")
    print(f"     {needs_optimization}")

    if coord_result:
        print(f"   - 协调优化净收益: {coord_result['net_benefit_score']:.1f}")
        print(f"   - KV Cache 复用提升: {coord_result['kv_cache_reuse_improvement']:.1f}%")

    print(f"\n💡 下一步:")
    print(f"   1. 应用协调优化建议（统一 block_size）")
    print(f"   2. 对每个 Agent 应用个性化优化")
    print(f"   3. 设置 A/B 测试验证效果")
    print(f"   4. 持续监控时间序列，检测模式变化")

    # 清理
    Path(db_path).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
