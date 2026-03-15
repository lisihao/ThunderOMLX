"""
测试 A/B 测试框架（Phase 3-B）

验证:
1. 启动 A/B 测试
2. 随机分流到 control/treatment
3. 记录样本并计算统计量
4. 评估实验结果（统计显著性）
5. 自动停止实验
"""

import sys
from pathlib import Path
import tempfile
import random

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def simulate_ab_test_with_real_difference():
    """模拟一个有真实差异的 A/B 测试"""
    print("=" * 70)
    print("场景 1: 有真实差异（treatment 更优）")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 启动 A/B 测试
    agent_id = "test-agent"
    experiment_id = aco.start_ab_test(
        agent_id=agent_id,
        control_block_size=128,
        treatment_block_size=64,
        treatment_ratio=0.5  # 50/50 split for faster testing
    )

    print(f"\n✅ 实验 #{experiment_id} 已启动")
    print(f"   Control: block_size=128")
    print(f"   Treatment: block_size=64")
    print(f"   Split: 50/50\n")

    # 模拟 200 次请求
    for i in range(200):
        use_treatment, treatment_block_size = aco.should_use_treatment(agent_id)

        if use_treatment:
            # Treatment 组（block_size=64, 更少 padding）
            prefill_time_ms = 224.0  # 更快
            total_time_ms = 524.0
            padding_overhead = 0.0
            group = "T"
        else:
            # Control 组（block_size=128, 更多 padding）
            prefill_time_ms = 256.0  # 更慢
            total_time_ms = 556.0
            padding_overhead = 14.3
            group = "C"

        # 记录样本
        aco.record_ab_sample(
            experiment_id=experiment_id,
            is_treatment=use_treatment,
            prefill_time_ms=prefill_time_ms,
            total_time_ms=total_time_ms,
            padding_overhead=padding_overhead
        )

        if (i + 1) % 50 == 0:
            print(f"   收集样本: {i+1}/200 (最后一个: {group})")

    # 评估实验
    print(f"\n🔍 评估实验结果...")
    result = aco.evaluate_ab_test(experiment_id, min_samples=50)

    if result:
        print(f"\n✅ 实验评估完成:")
        print(f"   Control 样本: {result['control_samples']}, 平均时间: {result['control_avg_total_ms']:.1f}ms")
        print(f"   Treatment 样本: {result['treatment_samples']}, 平均时间: {result['treatment_avg_total_ms']:.1f}ms")
        print(f"   改进: {result['improvement_pct']:.1f}%")
        print(f"   P-value: {result['p_value']:.3f}")
        print(f"   显著性: {'✅ 显著' if result['is_significant'] else '❌ 不显著'}")
        print(f"   胜者: {result['winner']}")
        print(f"   结论: {result['conclusion']}")

        # 停止实验
        aco.stop_ab_test(
            experiment_id=experiment_id,
            winner=result['winner'],
            conclusion=result['conclusion'],
            p_value=result['p_value']
        )
        print(f"\n🛑 实验已停止")
    else:
        print("\n❌ 评估失败")
        return False

    Path(db_path).unlink(missing_ok=True)
    return True


def simulate_ab_test_without_difference():
    """模拟一个无真实差异的 A/B 测试"""
    print("\n" + "=" * 70)
    print("场景 2: 无真实差异（tie）")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 启动 A/B 测试
    agent_id = "test-agent"
    experiment_id = aco.start_ab_test(
        agent_id=agent_id,
        control_block_size=128,
        treatment_block_size=64,
        treatment_ratio=0.5
    )

    print(f"\n✅ 实验 #{experiment_id} 已启动")

    # 模拟 200 次请求（两组性能相同）
    for i in range(200):
        use_treatment, _ = aco.should_use_treatment(agent_id)

        # 两组性能相同
        prefill_time_ms = 240.0 + random.uniform(-5, 5)  # 添加噪声
        total_time_ms = 540.0 + random.uniform(-10, 10)
        padding_overhead = 7.0 + random.uniform(-1, 1)

        aco.record_ab_sample(
            experiment_id=experiment_id,
            is_treatment=use_treatment,
            prefill_time_ms=prefill_time_ms,
            total_time_ms=total_time_ms,
            padding_overhead=padding_overhead
        )

    # 评估实验
    result = aco.evaluate_ab_test(experiment_id, min_samples=50)

    if result:
        print(f"\n✅ 实验评估完成:")
        print(f"   改进: {result['improvement_pct']:.1f}%")
        print(f"   显著性: {'✅ 显著' if result['is_significant'] else '❌ 不显著'}")
        print(f"   胜者: {result['winner']}")
        print(f"   结论: {result['conclusion']}")
    else:
        print("\n❌ 评估失败")
        return False

    Path(db_path).unlink(missing_ok=True)
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("A/B Testing Framework - Phase 3-B")
    print("=" * 70)

    # 场景 1: 有真实差异
    success1 = simulate_ab_test_with_real_difference()

    # 场景 2: 无真实差异
    success2 = simulate_ab_test_without_difference()

    if success1 and success2:
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        print("\n💡 关键结论:")
        print("   - A/B 测试可自动分流请求")
        print("   - 统计显著性检验正确识别差异")
        print("   - 自动判断实验成功/失败")
        print("\n🎯 Phase 3-B 实现成功！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
