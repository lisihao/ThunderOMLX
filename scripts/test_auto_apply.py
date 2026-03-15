"""
测试 Adaptive Cache Optimizer 自动应用功能

验证:
1. analyze_patterns() 方法正确分析数据
2. get_recommendations() 返回优化建议
3. apply_optimization() 记录配置变更
"""

import sys
from pathlib import Path
import tempfile

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def test_analyze_patterns():
    """测试模式分析功能"""
    print("=" * 70)
    print("Test 1: analyze_patterns()")
    print("=" * 70)

    # 使用临时数据库
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 插入测试数据（模拟 agent，当前 block_size=128，应该推荐 64）
    # 选择 prompt_length=320，使得 block_size=128 时 padding=64，block_size=64 时 padding=0
    for i in range(30):
        aco.log_inference(
            agent_id="test-agent",
            system_prompt_length=318,  # 平均 320 tokens
            user_query_length=2 + (i % 3),
            cache_hit_ratio=0.95,
            skip_logic_type="APPROXIMATE",
            block_size=128,  # 当前配置
            padding_tokens=64,  # 320 % 128 = 64, padding = 64
            prefill_time_ms=100.0,
            decode_time_ms=50.0,
        )

    # 调试：查看数据库中的实际数据
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT AVG(total_prompt_length), AVG(padding_overhead), block_size
            FROM agent_metrics
            WHERE agent_id = 'test-agent'
        """)
        row = cursor.fetchone()
        print(f"\n调试信息:")
        print(f"  平均 prompt 长度: {row[0]}")
        print(f"  平均 padding overhead: {row[1]:.1f}%")
        print(f"  当前 block_size: {row[2]}")

    # 分析模式
    result = aco.analyze_patterns("test-agent", min_samples=20)

    if result:
        print("\n✅ 分析成功:")
        print(f"   Agent: {result['agent_id']}")
        print(f"   当前 block_size: {result['current_block_size']}")
        print(f"   推荐 block_size: {result['recommended_block_size']}")
        print(f"   当前 padding: {result['current_padding_overhead']:.1f}%")
        print(f"   优化后 padding: {result['recommended_padding_overhead']:.1f}%")
        print(f"   改进幅度: {result['improvement_pct']:.1f}%")
        print(f"   原因: {result['reason']}")
        print(f"   样本数: {result['sample_count']}")
    else:
        print("\n❌ 分析失败: 没有优化建议")

    # 清理
    Path(db_path).unlink(missing_ok=True)

    return result is not None


def test_get_recommendations():
    """测试批量获取优化建议"""
    print("\n" + "=" * 70)
    print("Test 2: get_recommendations()")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 插入多个 agent 的数据
    agents = {
        "agent-1": (400, 128),  # 应该推荐 64
        "agent-2": (640, 64),   # 应该推荐 128
        "agent-3": (850, 256),  # 已最优
    }

    for agent_id, (system_prompt_length, block_size) in agents.items():
        for i in range(30):
            aco.log_inference(
                agent_id=agent_id,
                system_prompt_length=system_prompt_length,
                user_query_length=10 + (i % 5),
                cache_hit_ratio=0.95,
                skip_logic_type="APPROXIMATE",
                block_size=block_size,
                padding_tokens=20,
                prefill_time_ms=100.0,
                decode_time_ms=50.0,
            )

    # 获取所有推荐
    recommendations = aco.get_recommendations(min_samples=20)

    print(f"\n找到 {len(recommendations)} 个优化建议:")
    for rec in recommendations:
        print(f"\n   Agent: {rec['agent_id']}")
        print(f"      {rec['current_block_size']} → {rec['recommended_block_size']}")
        print(f"      改进: {rec['improvement_pct']:.1f}%")

    # 清理
    Path(db_path).unlink(missing_ok=True)

    return len(recommendations) > 0


def test_apply_optimization():
    """测试应用优化并记录到 config_history"""
    print("\n" + "=" * 70)
    print("Test 3: apply_optimization()")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)

    # 应用优化
    aco.apply_optimization(
        agent_id="test-agent",
        new_block_size=64,
        old_block_size=128,
        reason="Test optimization: reduce padding from 15% to 5%"
    )

    # 验证记录
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT agent_id, old_block_size, new_block_size, change_reason
            FROM config_history
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

    if row:
        print("\n✅ 优化已记录:")
        print(f"   Agent: {row[0]}")
        print(f"   旧 block_size: {row[1]}")
        print(f"   新 block_size: {row[2]}")
        print(f"   原因: {row[3]}")
    else:
        print("\n❌ 优化记录失败")

    # 清理
    Path(db_path).unlink(missing_ok=True)

    return row is not None


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Adaptive Cache Optimizer - 自动应用功能测试")
    print("=" * 70)

    tests = [
        ("analyze_patterns", test_analyze_patterns),
        ("get_recommendations", test_get_recommendations),
        ("apply_optimization", test_apply_optimization),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ 测试 {name} 失败: {e}")
            results[name] = False

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
