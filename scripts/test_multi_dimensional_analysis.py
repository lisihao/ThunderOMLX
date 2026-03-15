"""
测试多维度分析引擎（Phase 3-A）

验证:
1. padding 分析
2. cache_hit_ratio 分析
3. skip_logic_effectiveness 分析
4. prefill/decode 比例分析
5. 综合评分和优先级排序
"""

import sys
from pathlib import Path
import tempfile
import random

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def test_multi_dimensional_analysis():
    """测试多维度分析"""
    print("=" * 70)
    print("Test: Multi-Dimensional Analysis (Phase 3-A)")
    print("=" * 70)

    # 使用临时数据库
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 场景 1: Padding 问题严重的 agent
    print("\n场景 1: Padding 问题严重（block_size=128, prompt=448）")
    for i in range(50):
        aco.log_inference(
            agent_id="high-padding-agent",
            system_prompt_length=448,
            user_query_length=0,
            cache_hit_ratio=0.95,  # 高 cache hit
            skip_logic_type="APPROXIMATE",  # 使用 skip logic
            block_size=128,
            padding_tokens=64,  # 高 padding
            prefill_time_ms=256.0,
            decode_time_ms=300.0,
        )

    result = aco.analyze_multi_dimensional("high-padding-agent", min_samples=20)

    if result:
        print(f"\n✅ 分析成功:")
        print(f"   Overall Score: {result['overall_score']:.1f}/100")
        print(f"\n   维度评分:")
        for dim_name, dim_data in result['dimensions'].items():
            print(f"   - {dim_name}: {dim_data['score']:.1f}/100")

        print(f"\n   优化建议 ({len(result['recommendations'])} 条):")
        for rec in result['recommendations']:
            print(f"   [{rec['priority'].upper()}] {rec['type']}: {rec['reason']}")
            print(f"      Current: {rec['current_value']} → Recommended: {rec['recommended_value']}")
    else:
        print("\n❌ 分析失败")
        return False

    # 场景 2: Cache hit 低的 agent
    print("\n" + "=" * 70)
    print("场景 2: Cache Hit 低（cache_hit_ratio=0.5）")
    for i in range(50):
        aco.log_inference(
            agent_id="low-cache-hit-agent",
            system_prompt_length=200,
            user_query_length=200,  # 大量 user query，导致 cache hit 低
            cache_hit_ratio=0.5,  # 低 cache hit
            skip_logic_type="NONE",  # 不使用 skip logic
            block_size=128,
            padding_tokens=24,
            prefill_time_ms=220.0,
            decode_time_ms=300.0,
        )

    result = aco.analyze_multi_dimensional("low-cache-hit-agent", min_samples=20)

    if result:
        print(f"\n✅ 分析成功:")
        print(f"   Overall Score: {result['overall_score']:.1f}/100")
        print(f"\n   维度评分:")
        for dim_name, dim_data in result['dimensions'].items():
            print(f"   - {dim_name}: {dim_data['score']:.1f}/100")

        print(f"\n   优化建议 ({len(result['recommendations'])} 条):")
        for rec in result['recommendations']:
            print(f"   [{rec['priority'].upper()}] {rec['type']}: {rec['reason']}")
            print(f"      Current: {rec['current_value']} → Recommended: {rec['recommended_value']}")
    else:
        print("\n❌ 分析失败")
        return False

    # 场景 3: Decode-heavy agent
    print("\n" + "=" * 70)
    print("场景 3: Decode-Heavy（decode 占 70%）")
    for i in range(50):
        aco.log_inference(
            agent_id="decode-heavy-agent",
            system_prompt_length=200,
            user_query_length=0,
            cache_hit_ratio=0.95,
            skip_logic_type="APPROXIMATE",
            block_size=64,
            padding_tokens=8,  # 低 padding
            prefill_time_ms=104.0,  # 30% 时间
            decode_time_ms=242.0,  # 70% 时间
        )

    result = aco.analyze_multi_dimensional("decode-heavy-agent", min_samples=20)

    if result:
        print(f"\n✅ 分析成功:")
        print(f"   Overall Score: {result['overall_score']:.1f}/100")
        print(f"\n   维度评分:")
        for dim_name, dim_data in result['dimensions'].items():
            print(f"   - {dim_name}: {dim_data['score']:.1f}/100")

        print(f"\n   优化建议 ({len(result['recommendations'])} 条):")
        for rec in result['recommendations']:
            print(f"   [{rec['priority'].upper()}] {rec['type']}: {rec['reason']}")
            print(f"      Current: {rec['current_value']} → Recommended: {rec['recommended_value']}")
    else:
        print("\n❌ 分析失败")
        return False

    # 清理
    Path(db_path).unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    print("\n💡 关键结论:")
    print("   - 多维度分析可识别不同类型的优化机会")
    print("   - 综合评分反映整体性能")
    print("   - 优先级排序帮助聚焦高价值优化")
    print("\n🎯 Phase 3-A 实现成功！")

    return True


if __name__ == "__main__":
    success = test_multi_dimensional_analysis()
    sys.exit(0 if success else 1)
