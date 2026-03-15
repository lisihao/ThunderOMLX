"""
Adaptive Cache Optimizer - Phase 3 高级策略综合演示

展示:
1. Phase 3-A: 多维度分析引擎
2. Phase 3-B: A/B 测试框架
3. Phase 3-C: 自动回滚机制

完整流程:
Agent → 数据收集 → 多维度分析 → 发现优化机会 →
A/B 测试验证 → 应用优化 → 监控效果 → 自动回滚（如果失败）
"""

import sys
from pathlib import Path
import tempfile
import random

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def demo_complete_workflow():
    """演示完整的优化工作流"""
    print("=" * 70)
    print("Phase 3 Advanced Strategies - 完整演示")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)
    agent_id = "production-agent"

    # ========================================================================
    # Step 1: 数据收集阶段
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: 数据收集阶段")
    print("=" * 70)
    print("收集 200 次推理数据（block_size=128, 次优配置）...")

    for i in range(200):
        aco.log_inference(
            agent_id=agent_id,
            system_prompt_length=448,
            user_query_length=random.randint(0, 10),
            cache_hit_ratio=0.92 + random.uniform(-0.05, 0.05),
            skip_logic_type="APPROXIMATE" if random.random() > 0.1 else "NONE",
            block_size=128,
            padding_tokens=64,
            prefill_time_ms=256.0 + random.uniform(-10, 10),
            decode_time_ms=300.0 + random.uniform(-20, 20),
        )

    print(f"✅ 已收集 200 次推理数据")

    # ========================================================================
    # Step 2: 多维度分析
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: 多维度分析（Phase 3-A）")
    print("=" * 70)

    analysis = aco.analyze_multi_dimensional(agent_id, min_samples=50)

    if analysis:
        print(f"\n📊 分析结果:")
        print(f"   Overall Score: {analysis['overall_score']:.1f}/100")
        print(f"\n   维度评分:")
        for dim_name, dim_data in analysis['dimensions'].items():
            print(f"   - {dim_name}: {dim_data['score']:.1f}/100")

        print(f"\n   优化建议:")
        for rec in analysis['recommendations']:
            print(f"   [{rec['priority'].upper()}] {rec['type']}: {rec['reason']}")

        if not analysis['recommendations']:
            print("   无优化建议")
            return

        # 选择第一个建议
        top_recommendation = analysis['recommendations'][0]

        if top_recommendation['type'] != 'block_size':
            print(f"\n⚠️ 顶级建议不是 block_size 优化，跳过 A/B 测试")
            return

        recommended_block_size = top_recommendation['recommended_value']
        current_block_size = top_recommendation['current_value']

    else:
        print("❌ 分析失败")
        return

    # ========================================================================
    # Step 3: A/B 测试（Phase 3-B）
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: A/B 测试（Phase 3-B）")
    print("=" * 70)
    print(f"启动 A/B 测试: Control={current_block_size}, Treatment={recommended_block_size}")

    experiment_id = aco.start_ab_test(
        agent_id=agent_id,
        control_block_size=current_block_size,
        treatment_block_size=recommended_block_size,
        treatment_ratio=0.2  # 20% 流量到实验组
    )

    print(f"✅ 实验 #{experiment_id} 已启动")
    print(f"收集 300 次 A/B 测试样本...")

    for i in range(300):
        use_treatment, treatment_block_size = aco.should_use_treatment(agent_id)

        if use_treatment:
            # Treatment 组（block_size=64）
            prefill_time_ms = 224.0 + random.uniform(-5, 5)
            total_time_ms = 524.0 + random.uniform(-10, 10)
            padding_overhead = 0.0
        else:
            # Control 组（block_size=128）
            prefill_time_ms = 256.0 + random.uniform(-5, 5)
            total_time_ms = 556.0 + random.uniform(-10, 10)
            padding_overhead = 14.3

        aco.record_ab_sample(
            experiment_id=experiment_id,
            is_treatment=use_treatment,
            prefill_time_ms=prefill_time_ms,
            total_time_ms=total_time_ms,
            padding_overhead=padding_overhead
        )

    # 评估 A/B 测试
    ab_result = aco.evaluate_ab_test(experiment_id, min_samples=30)

    if ab_result:
        print(f"\n📊 A/B 测试结果:")
        print(f"   Control: {ab_result['control_avg_total_ms']:.1f}ms ({ab_result['control_samples']} 样本)")
        print(f"   Treatment: {ab_result['treatment_avg_total_ms']:.1f}ms ({ab_result['treatment_samples']} 样本)")
        print(f"   改进: {ab_result['improvement_pct']:.1f}%")
        print(f"   显著性: {'✅ 显著' if ab_result['is_significant'] else '❌ 不显著'}")
        print(f"   胜者: {ab_result['winner']}")

        # 停止实验
        aco.stop_ab_test(
            experiment_id=experiment_id,
            winner=ab_result['winner'],
            conclusion=ab_result['conclusion'],
            p_value=ab_result['p_value']
        )

        if ab_result['winner'] != 'treatment':
            print(f"\n⚠️ 实验失败，不应用优化")
            return
    else:
        print("❌ A/B 测试评估失败")
        return

    # ========================================================================
    # Step 4: 应用优化（Phase 3-C）
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: 应用优化 + 回滚监控（Phase 3-C）")
    print("=" * 70)
    print(f"应用优化: block_size {current_block_size} → {recommended_block_size}")

    config_id = aco.apply_optimization_with_baseline(
        agent_id=agent_id,
        new_block_size=recommended_block_size,
        old_block_size=current_block_size,
        reason=top_recommendation['reason']
    )

    print(f"✅ 优化已应用 (config #{config_id})")
    print(f"开始监控期（收集 100 次样本）...")

    # 收集优化后数据
    for i in range(100):
        aco.log_inference(
            agent_id=agent_id,
            system_prompt_length=448,
            user_query_length=random.randint(0, 10),
            cache_hit_ratio=0.92 + random.uniform(-0.05, 0.05),
            skip_logic_type="APPROXIMATE" if random.random() > 0.1 else "NONE",
            block_size=recommended_block_size,
            padding_tokens=0,
            prefill_time_ms=224.0 + random.uniform(-5, 5),
            decode_time_ms=300.0 + random.uniform(-10, 10),
        )

    # 监控优化效果
    monitor_result = aco.monitor_optimization_effect(config_id, monitoring_samples=100)

    if monitor_result:
        print(f"\n📊 监控结果:")
        print(f"   Baseline: {monitor_result['baseline_total_ms']:.1f}ms")
        print(f"   Post: {monitor_result['post_total_ms']:.1f}ms")
        print(f"   Degradation: {monitor_result['degradation_pct']:.1f}%")
        print(f"   Should Rollback: {monitor_result['should_rollback']}")

        if monitor_result['should_rollback']:
            print(f"   ⚠️ 检测到性能下降: {monitor_result['rollback_reason']}")
            success = aco.rollback_optimization(config_id, monitor_result['rollback_reason'])
            if success:
                print(f"   🔄 已回滚到 block_size={current_block_size}")
        else:
            print(f"   ✅ 优化成功！性能改善 {abs(monitor_result['degradation_pct']):.1f}%")

    # ========================================================================
    # Step 5: 总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("Phase 3 高级策略演示完成")
    print("=" * 70)
    print("\n✅ 完整工作流:")
    print("   1. 多维度分析 → 识别优化机会")
    print("   2. A/B 测试 → 验证优化效果")
    print("   3. 应用优化 + 监控 → 自动回滚保护")
    print("\n🎯 Phase 3 高级策略引擎工作正常！")

    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    demo_complete_workflow()
