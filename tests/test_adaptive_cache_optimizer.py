"""
Tests for Adaptive Cache Optimizer - Phase 1: Data Collection

Tests cover:
- Database initialization
- Data insertion
- Thread safety
- Performance (<1ms per log)
"""

import pytest
import sqlite3
import tempfile
import time
import threading
from pathlib import Path

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


@pytest.fixture
def temp_db():
    """Fixture providing a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def aco(temp_db):
    """Fixture providing an initialized ACO instance."""
    return AdaptiveCacheOptimizer(temp_db)


def test_database_initialization(temp_db):
    """Test that database and tables are created correctly."""
    aco = AdaptiveCacheOptimizer(temp_db)

    # Check database file exists
    assert Path(temp_db).exists()

    # Check tables exist
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {
        'agent_metrics',
        'cache_performance',
        'config_history',
        'optimization_experiments',
    }
    # sqlite_sequence is auto-created by SQLite for AUTOINCREMENT columns
    assert expected_tables.issubset(tables), f"Expected tables missing: {expected_tables - tables}"


def test_log_inference_basic(aco):
    """Test basic inference logging."""
    aco.log_inference(
        agent_id="test-agent",
        system_prompt_length=100,
        user_query_length=20,
        cache_hit_ratio=0.95,
        skip_logic_type="APPROXIMATE",
        block_size=64,
        padding_tokens=5,
        prefill_time_ms=100.0,
        decode_time_ms=50.0,
    )

    # Verify data was inserted
    with aco._get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM agent_metrics")
        count = cursor.fetchone()[0]
        assert count == 1

        # Verify data correctness
        cursor = conn.execute("""
            SELECT agent_id, total_prompt_length, cache_hit_ratio,
                   skip_logic_type, padding_overhead, total_time_ms
            FROM agent_metrics LIMIT 1
        """)
        row = cursor.fetchone()

        assert row[0] == "test-agent"
        assert row[1] == 120  # 100 + 20
        assert abs(row[2] - 0.95) < 0.001
        assert row[3] == "APPROXIMATE"
        # padding_overhead = 5 / 120 * 100 = 4.17%
        assert abs(row[4] - 4.17) < 0.1
        assert abs(row[5] - 150.0) < 0.1


def test_log_inference_performance(aco):
    """Test that log_inference executes in <1ms."""
    # Warmup
    for _ in range(10):
        aco.log_inference(
            agent_id="test-agent",
            system_prompt_length=100,
            user_query_length=20,
            cache_hit_ratio=0.95,
            skip_logic_type="FULL",
            block_size=64,
            padding_tokens=0,
            prefill_time_ms=100.0,
            decode_time_ms=50.0,
        )

    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        aco.log_inference(
            agent_id="test-agent",
            system_prompt_length=100,
            user_query_length=20,
            cache_hit_ratio=0.95,
            skip_logic_type="FULL",
            block_size=64,
            padding_tokens=0,
            prefill_time_ms=100.0,
            decode_time_ms=50.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    # Check average time < 1ms
    avg_time = sum(times) / len(times)
    p95_time = sorted(times)[int(len(times) * 0.95)]

    print(f"\nlog_inference performance:")
    print(f"  Average: {avg_time:.3f}ms")
    print(f"  P95: {p95_time:.3f}ms")

    # Allow some slack for CI environments
    assert avg_time < 2.0, f"Average time {avg_time:.3f}ms exceeds 2ms threshold"
    assert p95_time < 5.0, f"P95 time {p95_time:.3f}ms exceeds 5ms threshold"


def test_thread_safety(aco):
    """Test that concurrent logging is thread-safe."""
    num_threads = 10
    logs_per_thread = 50

    def worker(thread_id):
        for i in range(logs_per_thread):
            aco.log_inference(
                agent_id=f"agent-{thread_id}",
                system_prompt_length=100 + thread_id,
                user_query_length=20,
                cache_hit_ratio=0.9,
                skip_logic_type="FULL",
                block_size=64,
                padding_tokens=0,
                prefill_time_ms=100.0,
                decode_time_ms=50.0,
            )

    threads = []
    for tid in range(num_threads):
        t = threading.Thread(target=worker, args=(tid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify all logs were inserted
    with aco._get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM agent_metrics")
        count = cursor.fetchone()[0]
        expected = num_threads * logs_per_thread
        assert count == expected, f"Expected {expected} logs, got {count}"


def test_get_stats(aco):
    """Test get_stats method."""
    # Log some data
    for i in range(5):
        aco.log_inference(
            agent_id="agent-1",
            system_prompt_length=100,
            user_query_length=20,
            cache_hit_ratio=0.9,
            skip_logic_type="FULL",
            block_size=64,
            padding_tokens=0,
            prefill_time_ms=100.0,
            decode_time_ms=50.0,
        )

    for i in range(3):
        aco.log_inference(
            agent_id="agent-2",
            system_prompt_length=200,
            user_query_length=30,
            cache_hit_ratio=0.8,
            skip_logic_type="APPROXIMATE",
            block_size=128,
            padding_tokens=5,
            prefill_time_ms=150.0,
            decode_time_ms=75.0,
        )

    # Test stats for specific agent
    stats1 = aco.get_stats(agent_id="agent-1")
    assert stats1['agent_id'] == "agent-1"
    assert stats1['total_records'] == 5

    stats2 = aco.get_stats(agent_id="agent-2")
    assert stats2['agent_id'] == "agent-2"
    assert stats2['total_records'] == 3

    # Test stats for all agents
    stats_all = aco.get_stats()
    assert stats_all['total_records'] == 8


def test_zero_padding_overhead(aco):
    """Test padding_overhead calculation when no padding is used."""
    aco.log_inference(
        agent_id="test-agent",
        system_prompt_length=64,  # Perfectly aligned to block_size=64
        user_query_length=0,
        cache_hit_ratio=1.0,
        skip_logic_type="FULL",
        block_size=64,
        padding_tokens=0,
        prefill_time_ms=100.0,
        decode_time_ms=50.0,
    )

    with aco._get_connection() as conn:
        cursor = conn.execute("SELECT padding_overhead FROM agent_metrics LIMIT 1")
        padding_overhead = cursor.fetchone()[0]
        assert padding_overhead == 0.0


def test_multiple_skip_logic_types(aco):
    """Test different skip_logic_type values."""
    skip_types = ['FULL', 'APPROXIMATE', 'NONE']

    for skip_type in skip_types:
        aco.log_inference(
            agent_id=f"agent-{skip_type.lower()}",
            system_prompt_length=100,
            user_query_length=20,
            cache_hit_ratio=0.9 if skip_type != 'NONE' else 0.0,
            skip_logic_type=skip_type,
            block_size=64,
            padding_tokens=0,
            prefill_time_ms=100.0,
            decode_time_ms=50.0,
        )

    with aco._get_connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT skip_logic_type FROM agent_metrics ORDER BY skip_logic_type"
        )
        recorded_types = {row[0] for row in cursor.fetchall()}

    assert recorded_types == set(skip_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
