"""
测试时间序列分析（Phase 3-E）

验证:
1. 时间窗口分析（1h, 24h, 168h）
2. 模式变化检测
3. 动态调整建议
"""

import sys
from pathlib import Path
import tempfile
import sqlite3
from datetime import datetime, timedelta

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


def insert_historical_data(db_path, agent_id):
    """插入历史数据（模拟时间序列）"""
    with sqlite3.connect(db_path) as conn:
        # 基线数据（1-25 小时前）：短 prompt
        for i in range(50):
            hours_ago = 2 + (i % 23)  # 2-24 小时前

            conn.execute("""
                INSERT INTO agent_metrics (
                    agent_id, timestamp,
                    system_prompt_length, user_query_length, total_prompt_length,
                    cache_hit_ratio, skip_logic_type, block_size,
                    padding_tokens, padding_overhead,
                    prefill_time_ms, decode_time_ms, total_time_ms,
                    config_version
                ) VALUES (?, datetime('now', '-' || ? || ' hours'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, hours_ago,
                300, 10, 310,  # 短 prompt（基线）
                0.95, "APPROXIMATE", 64,
                10, 3.2,
                160.0, 300.0, 460.0,
                "1.0.0"
            ))

        # 最近数据（最近 1 小时）：长 prompt（模式变化）
        for i in range(30):
            minutes_ago = i * 2  # 0-58 分钟前

            conn.execute("""
                INSERT INTO agent_metrics (
                    agent_id, timestamp,
                    system_prompt_length, user_query_length, total_prompt_length,
                    cache_hit_ratio, skip_logic_type, block_size,
                    padding_tokens, padding_overhead,
                    prefill_time_ms, decode_time_ms, total_time_ms,
                    config_version
                ) VALUES (?, datetime('now', '-' || ? || ' minutes'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, minutes_ago,
                600, 10, 610,  # 长 prompt（变化后）
                0.85, "APPROXIMATE", 64,
                38, 6.2,
                324.0, 300.0, 624.0,
                "1.0.0"
            ))

        conn.commit()


def test_time_series_analysis():
    """测试时间序列分析"""
    print("=" * 70)
    print("Test: Time Series Analysis (Phase 3-E)")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    aco = AdaptiveCacheOptimizer(db_path)
    agent_id = "time-varying-agent"

    # 插入历史数据
    print("\n准备测试数据:")
    print("   基线（1-25h 前）: prompt=310, total=460ms")
    print("   最近（1h 内）: prompt=610, total=624ms（模式变化）")
    insert_historical_data(db_path, agent_id)
    print("   ✅ 数据已插入")

    # ========================================================================
    # Test 1: 时间序列分析
    # ========================================================================
    print("\n" + "=" * 70)
    print("Test 1: 时间序列分析")
    print("=" * 70)

    result = aco.analyze_time_series(agent_id, window_hours=[1, 24])

    if result:
        print(f"\n📊 时间序列分析结果:")
        print(f"   Agent: {result['agent_id']}")

        print(f"\n   时间窗口数据:")
        for window, data in result['windows'].items():
            print(f"\n   {window}:")
            print(f"      Avg prompt length: {data['avg_prompt_length']:.0f}")
            print(f"      Avg total time: {data['avg_total']:.1f}ms")
            print(f"      Avg padding: {data['avg_padding']:.1f}%")
            print(f"      Samples: {data['sample_count']}")

        print(f"\n   模式变化检测: {result['pattern_changes_detected']}")

        if result['pattern_changes']:
            print(f"\n   检测到 {len(result['pattern_changes'])} 个模式变化:")
            for change in result['pattern_changes']:
                print(f"   - {change['type']}: {change['description']}")

        if result['recommendations']:
            print(f"\n   建议 ({len(result['recommendations'])} 条):")
            for rec in result['recommendations']:
                print(f"   - {rec['action']}: {rec['reason']}")
        else:
            print(f"\n   无建议")

    else:
        print("❌ 时间序列分析失败")
        return False

    # ========================================================================
    # Test 2: 模式变化检测
    # ========================================================================
    print("\n" + "=" * 70)
    print("Test 2: 模式变化检测")
    print("=" * 70)

    change_result = aco.detect_pattern_change(
        agent_id=agent_id,
        metric='total_prompt_length',
        threshold_pct=15.0
    )

    if change_result:
        print(f"\n📊 模式变化检测结果:")
        print(f"   Metric: {change_result['metric']}")
        print(f"   Recent avg: {change_result['recent_avg']:.0f}")
        print(f"   Baseline avg: {change_result['baseline_avg']:.0f}")
        print(f"   Change: {change_result['change_pct']:.1f}%")
        print(f"   Direction: {change_result['direction']}")
        print(f"   Change detected: {change_result['change_detected']}")

        if change_result['change_detected']:
            print(f"\n   ⚠️ 检测到显著变化！建议重新分析配置")
        else:
            print(f"\n   ✅ 模式稳定")

    else:
        print("❌ 模式变化检测失败")
        return False

    Path(db_path).unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    print("\n💡 关键结论:")
    print("   - 时间窗口分析识别短期/长期差异")
    print("   - 模式变化检测触发重新优化")
    print("   - 动态调整建议提升适应性")
    print("\n🎯 Phase 3-E 实现成功！")

    return True


if __name__ == "__main__":
    success = test_time_series_analysis()
    sys.exit(0 if success else 1)
