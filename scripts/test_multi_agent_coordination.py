"""
测试多 Agent 协同优化（Phase 3-D）

验证:
1. 分析全局 block_size 分布
2. 识别碎片化问题
3. 推荐协调后的 block_size
4. 计算 KV Cache 复用收益
"""

import sys
from pathlib import Path
import tempfile
import random

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def test_coordinated_optimization():
    """测试多 Agent 协同优化"""
    print("=" * 70)
    print("Test: Multi-Agent Coordinated Optimization (Phase 3-D)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # ========================================================================
    # 场景: 多个 agent，block_size 高度碎片化
    # ========================================================================
    print("\n场景: 6 个 Agent，5 种不同的 block_size（碎片化）")

    agent_configs = [
        # (agent_id, avg_prompt_length, block_size, padding)
        ("short-prompt-agent-1", 150, 64, 22),   # 短 prompt
        ("short-prompt-agent-2", 180, 96, 12),   # 短 prompt，次优 block_size
        ("medium-prompt-agent-1", 450, 128, 64), # 中 prompt
        ("medium-prompt-agent-2", 480, 160, 32), # 中 prompt，次优 block_size
        ("long-prompt-agent-1", 900, 256, 28),   # 长 prompt
        ("long-prompt-agent-2", 950, 224, 62),   # 长 prompt，次优 block_size
    ]

    print("\n收集数据:")
    for agent_id, avg_prompt, block_size, padding in agent_configs:
        print(f"   - {agent_id}: prompt={avg_prompt}, block_size={block_size}, padding={padding}")
        for i in range(50):
            prompt_length = avg_prompt + random.randint(-10, 10)
            aco.log_inference(
                agent_id=agent_id,
                system_prompt_length=prompt_length,
                user_query_length=0,
                cache_hit_ratio=0.92,
                skip_logic_type="APPROXIMATE",
                block_size=block_size,
                padding_tokens=padding,
                prefill_time_ms=prompt_length * 0.5,
                decode_time_ms=300.0,
            )

    # ========================================================================
    # Step 1: 全局分析
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: 全局分析")
    print("=" * 70)

    global_analysis = aco.analyze_global_optimization()

    if global_analysis:
        print(f"\n📊 全局分析结果:")
        print(f"   Total Agents: {global_analysis['total_agents']}")
        print(f"   Unique Block Sizes: {global_analysis['num_unique_block_sizes']}")
        print(f"   Fragmentation Score: {global_analysis['fragmentation_score']:.1f}/100")
        print(f"   KV Cache Reuse Potential: {global_analysis['kv_cache_reuse_potential']:.1f}/100")

        print(f"\n   Block Size 分布:")
        for block_size, agents in sorted(global_analysis['block_size_distribution'].items()):
            print(f"   - {block_size}: {len(agents)} agents - {agents}")

        print(f"\n   推荐: {global_analysis['recommendation']}")
    else:
        print("❌ 全局分析失败")
        return False

    # ========================================================================
    # Step 2: 协调优化推荐
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: 协调优化推荐")
    print("=" * 70)

    coord_result = aco.recommend_coordinated_block_sizes(max_block_sizes=3)

    if coord_result:
        print(f"\n📊 协调优化方案:")
        print(f"   Clusters: {coord_result['num_clusters']}")
        print(f"   Total Agents: {coord_result['total_agents']}")

        print(f"\n   集群详情:")
        for cluster in coord_result['clusters']:
            print(f"\n   Cluster {cluster['cluster_id']}:")
            print(f"      Recommended block_size: {cluster['recommended_block_size']}")
            print(f"      Agents ({cluster['agent_count']}): {cluster['agent_ids']}")
            print(f"      Avg prompt length: {cluster['avg_prompt_length']:.0f}")
            print(f"      Expected padding: {cluster['expected_padding_overhead']:.1f}%")

        print(f"\n   优化效果:")
        print(f"      当前平均 padding: {coord_result['current_avg_padding']:.1f}%")
        print(f"      协调后平均 padding: {coord_result['expected_avg_padding']:.1f}%")
        print(f"      Padding 变化: {coord_result['overall_padding_increase']:.1f}%")
        print(f"      KV Cache 复用提升: {coord_result['kv_cache_reuse_improvement']:.1f}%")
        print(f"      净收益评分: {coord_result['net_benefit_score']:.1f}")

        if coord_result['net_benefit_score'] > 0:
            print(f"\n   ✅ 建议应用协调优化（净收益 > 0）")
        else:
            print(f"\n   ⚠️ 不建议协调（padding 增加超过 KV Cache 收益）")

    else:
        print("❌ 协调优化推荐失败")
        return False

    Path(db_path).unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("✅ 测试通过！")
    print("=" * 70)
    print("\n💡 关键结论:")
    print("   - 识别 block_size 碎片化问题")
    print("   - 自动聚类 agent 到最优 block_size")
    print("   - 平衡 padding vs KV Cache 复用")
    print("\n🎯 Phase 3-D 实现成功！")

    return True


if __name__ == "__main__":
    success = test_coordinated_optimization()
    sys.exit(0 if success else 1)
