"""
测试自动回滚机制（Phase 3-C）

验证:
1. 记录优化前的性能基线
2. 应用优化
3. 监控优化后的性能
4. 检测性能下降
5. 自动回滚到之前的配置
"""

import sys
from pathlib import Path
import tempfile

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def test_successful_optimization_no_rollback():
    """测试成功的优化（不需要回滚）"""
    print("=" * 70)
    print("场景 1: 成功的优化（不需要回滚）")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)
    agent_id = "test-agent"

    # Phase 1: 建立基线（block_size=128, 性能较差）
    print("\nPhase 1: 建立基线（block_size=128）")
    for i in range(100):
        aco.log_inference(
            agent_id=agent_id,
            system_prompt_length=448,
            user_query_length=0,
            cache_hit_ratio=0.95,
            skip_logic_type="APPROXIMATE",
            block_size=128,
            padding_tokens=64,
            prefill_time_ms=256.0,  # 慢
            decode_time_ms=300.0,
        )
    print("   ✅ 基线: avg_total=556.0ms, avg_padding=14.3%")

    # Phase 2: 应用优化（block_size=64）
    print("\nPhase 2: 应用优化（block_size=64）")
    config_id = aco.apply_optimization_with_baseline(
        agent_id=agent_id,
        new_block_size=64,
        old_block_size=128,
        reason="Reduce padding"
    )
    print(f"   ✅ 优化已应用 (config #{config_id})")

    # Phase 3: 收集优化后数据（性能改善）
    print("\nPhase 3: 收集优化后数据")
    for i in range(100):
        aco.log_inference(
            agent_id=agent_id,
            system_prompt_length=448,
            user_query_length=0,
            cache_hit_ratio=0.95,
            skip_logic_type="APPROXIMATE",
            block_size=64,
            padding_tokens=0,  # 更好
            prefill_time_ms=224.0,  # 更快
            decode_time_ms=300.0,
        )
    print("   ✅ 优化后: avg_total=524.0ms, avg_padding=0.0%")

    # Phase 4: 监控优化效果
    print("\nPhase 4: 监控优化效果")
    result = aco.monitor_optimization_effect(config_id, monitoring_samples=100)

    if result:
        print(f"   Baseline: {result['baseline_total_ms']:.1f}ms")
        print(f"   Post: {result['post_total_ms']:.1f}ms")
        print(f"   Degradation: {result['degradation_pct']:.1f}%")
        print(f"   Should Rollback: {result['should_rollback']}")

        if not result['should_rollback']:
            print(f"   ✅ 优化成功！无需回滚")
        else:
            print(f"   ❌ 错误：不应该触发回滚")
            return False
    else:
        print("   ❌ 监控失败")
        return False

    Path(db_path).unlink(missing_ok=True)
    return True


def test_failed_optimization_with_rollback():
    """测试失败的优化（需要回滚）"""
    print("\n" + "=" * 70)
    print("场景 2: 失败的优化（需要回滚）")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)
    agent_id = "test-agent"

    # Phase 1: 建立基线（block_size=64, 性能较好）
    print("\nPhase 1: 建立基线（block_size=64）")
    for i in range(100):
        aco.log_inference(
            agent_id=agent_id,
            system_prompt_length=448,
            user_query_length=0,
            cache_hit_ratio=0.95,
            skip_logic_type="APPROXIMATE",
            block_size=64,
            padding_tokens=0,
            prefill_time_ms=224.0,  # 快
            decode_time_ms=300.0,
        )
    print("   ✅ 基线: avg_total=524.0ms, avg_padding=0.0%")

    # Phase 2: 错误的优化（block_size=256, 反而更差）
    print("\nPhase 2: 应用错误优化（block_size=256）")
    config_id = aco.apply_optimization_with_baseline(
        agent_id=agent_id,
        new_block_size=256,
        old_block_size=64,
        reason="Wrong optimization"
    )
    print(f"   ✅ 优化已应用 (config #{config_id})")

    # Phase 3: 收集优化后数据（性能变差！）
    print("\nPhase 3: 收集优化后数据（性能变差）")
    for i in range(100):
        aco.log_inference(
            agent_id=agent_id,
            system_prompt_length=448,
            user_query_length=0,
            cache_hit_ratio=0.95,
            skip_logic_type="APPROXIMATE",
            block_size=256,
            padding_tokens=64,  # 更差
            prefill_time_ms=256.0,  # 更慢
            decode_time_ms=320.0,  # 更慢
        )
    print("   ❌ 优化后: avg_total=576.0ms, avg_padding=14.3%（变差了！）")

    # Phase 4: 监控优化效果
    print("\nPhase 4: 监控优化效果")
    result = aco.monitor_optimization_effect(config_id, monitoring_samples=100)

    if result:
        print(f"   Baseline: {result['baseline_total_ms']:.1f}ms")
        print(f"   Post: {result['post_total_ms']:.1f}ms")
        print(f"   Degradation: {result['degradation_pct']:.1f}%")
        print(f"   Should Rollback: {result['should_rollback']}")

        if result['should_rollback']:
            print(f"   ⚠️ 检测到性能下降: {result['rollback_reason']}")

            # Phase 5: 执行回滚
            print("\nPhase 5: 执行回滚")
            success = aco.rollback_optimization(config_id, result['rollback_reason'])
            if success:
                print(f"   ✅ 回滚成功: block_size 256 → 64")
            else:
                print(f"   ❌ 回滚失败")
                return False
        else:
            print(f"   ❌ 错误：应该触发回滚")
            return False
    else:
        print("   ❌ 监控失败")
        return False

    Path(db_path).unlink(missing_ok=True)
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Auto Rollback Mechanism - Phase 3-C")
    print("=" * 70)

    # 场景 1: 成功的优化
    success1 = test_successful_optimization_no_rollback()

    # 场景 2: 失败的优化（需要回滚）
    success2 = test_failed_optimization_with_rollback()

    if success1 and success2:
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        print("\n💡 关键结论:")
        print("   - 自动记录优化前的性能基线")
        print("   - 持续监控优化后的性能")
        print("   - 自动检测性能下降并触发回滚")
        print("\n🎯 Phase 3-C 实现成功！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
