#!/usr/bin/env python3
"""
测试 LRU-2 缓存驱逐策略

验证点：
1. 新 blocks 进入 COLD queue
2. 第 2 次访问触发 COLD → HOT 晋升
3. 驱逐顺序：COLD 优先，HOT 次之
4. LRU 顺序维护
5. 线程安全
"""

import sys
import hashlib
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.paged_ssd_cache import PagedSSDCacheIndex, PagedSSDBlockMetadata


def create_metadata(block_id: str, file_size: int = 1024) -> PagedSSDBlockMetadata:
    """创建测试用的 metadata"""
    block_hash = hashlib.sha256(block_id.encode()).digest()
    return PagedSSDBlockMetadata(
        block_hash=block_hash,
        file_path=Path(f"/tmp/{block_id}.safetensors"),
        file_size=file_size,
        token_count=256,
        created_at=0.0,
        last_access=0.0,
        num_layers=32,
    )


def test_1_add_enters_cold_queue():
    """测试 1: 新 blocks 进入 COLD queue"""
    print("测试 1: 新 blocks 进入 COLD queue...")

    index = PagedSSDCacheIndex(max_size_bytes=10 * 1024**3)
    meta_a = create_metadata("block_a")

    index.add(meta_a)
    stats = index.get_stats()

    assert stats['cold_blocks'] == 1, f"Expected 1 cold block, got {stats['cold_blocks']}"
    assert stats['hot_blocks'] == 0, f"Expected 0 hot blocks, got {stats['hot_blocks']}"
    assert stats['total_blocks'] == 1, f"Expected 1 total block, got {stats['total_blocks']}"

    print("  ✅ 通过")


def test_2_second_touch_promotes_to_hot():
    """测试 2: 第 2 次访问触发 COLD → HOT 晋升"""
    print("测试 2: 第 2 次访问触发 COLD → HOT 晋升...")

    index = PagedSSDCacheIndex(max_size_bytes=10 * 1024**3)
    meta_a = create_metadata("block_a")

    index.add(meta_a)  # 第 1 次访问
    stats = index.get_stats()
    assert stats['cold_blocks'] == 1
    assert stats['hot_blocks'] == 0

    index.touch(meta_a.block_hash)  # 第 2 次访问
    stats = index.get_stats()

    assert stats['cold_blocks'] == 0, f"Expected 0 cold blocks, got {stats['cold_blocks']}"
    assert stats['hot_blocks'] == 1, f"Expected 1 hot block, got {stats['hot_blocks']}"
    assert stats['promotions'] == 1, f"Expected 1 promotion, got {stats['promotions']}"

    print("  ✅ 通过")


def test_3_cold_evicted_before_hot():
    """测试 3: COLD blocks 优先驱逐"""
    print("测试 3: COLD blocks 优先驱逐...")

    index = PagedSSDCacheIndex(max_size_bytes=10 * 1024**3)
    meta_a = create_metadata("block_a")  # COLD
    meta_b = create_metadata("block_b")  # HOT

    index.add(meta_a)
    index.add(meta_b)
    index.touch(meta_b.block_hash)  # Promote B to HOT

    stats = index.get_stats()
    assert stats['cold_blocks'] == 1
    assert stats['hot_blocks'] == 1

    # Evict all blocks (应该先驱逐 A (COLD)，再驱逐 B (HOT))
    evicted = index.evict_until_size(0)

    assert len(evicted) == 2, f"Expected 2 evictions, got {len(evicted)}"
    assert evicted[0].block_hash == meta_a.block_hash, "Expected COLD block (A) to be evicted first"
    assert evicted[1].block_hash == meta_b.block_hash, "Expected HOT block (B) to be evicted second"

    stats = index.get_stats()
    assert stats['cold_evictions'] == 1, f"Expected 1 cold eviction, got {stats['cold_evictions']}"
    assert stats['hot_evictions'] == 1, f"Expected 1 hot eviction, got {stats['hot_evictions']}"

    print("  ✅ 通过")


def test_4_lru_order_within_queue():
    """测试 4: 队列内 LRU 顺序维护"""
    print("测试 4: 队列内 LRU 顺序维护...")

    index = PagedSSDCacheIndex(max_size_bytes=10 * 1024**3)
    meta_a = create_metadata("block_a")
    meta_b = create_metadata("block_b")
    meta_c = create_metadata("block_c")

    # Add in order A, B, C
    index.add(meta_a)
    index.add(meta_b)
    index.add(meta_c)

    # Evict all (应该按 LRU 顺序: A, B, C)
    evicted = index.evict_until_size(0)

    assert len(evicted) == 3, f"Expected 3 evictions, got {len(evicted)}"
    assert evicted[0].block_hash == meta_a.block_hash, "First eviction should be A"
    assert evicted[1].block_hash == meta_b.block_hash, "Second eviction should be B"
    assert evicted[2].block_hash == meta_c.block_hash, "Third eviction should be C"

    print("  ✅ 通过")


def test_5_touch_moves_to_mru():
    """测试 5: touch() 移动到 MRU 位置"""
    print("测试 5: touch() 移动到 MRU 位置...")

    index = PagedSSDCacheIndex(max_size_bytes=10 * 1024**3)
    meta_a = create_metadata("block_a")
    meta_b = create_metadata("block_b")

    index.add(meta_a)
    index.add(meta_b)

    # Touch A (move to MRU in COLD)
    index.touch(meta_a.block_hash)  # A 的第 2 次访问会晋升到 HOT

    # Evict all
    evicted = index.evict_until_size(0)

    # B 应该先被驱逐（在 COLD queue 中是 LRU）
    # A 在 HOT queue 中，后驱逐
    assert evicted[0].block_hash == meta_b.block_hash, "B should be evicted first (LRU in COLD)"
    assert evicted[1].block_hash == meta_a.block_hash, "A should be evicted second (promoted to HOT)"

    print("  ✅ 通过")


def test_6_thread_safety():
    """测试 6: 线程安全"""
    print("测试 6: 线程安全...")

    import concurrent.futures
    import random

    index = PagedSSDCacheIndex(max_size_bytes=100 * 1024**3)

    def worker(worker_id):
        """并发 worker"""
        for i in range(100):
            meta = create_metadata(f"w{worker_id}_b{i}")
            index.add(meta)

            if random.random() < 0.5:
                index.touch(meta.block_hash)

            index.get(meta.block_hash)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        concurrent.futures.wait(futures)

    stats = index.get_stats()
    assert stats['total_blocks'] > 0, "Should have blocks after concurrent operations"
    print(f"  ✅ 通过 (total_blocks={stats['total_blocks']}, cold={stats['cold_blocks']}, hot={stats['hot_blocks']})")


def test_7_get_stats_returns_all_fields():
    """测试 7: get_stats() 返回所有字段"""
    print("测试 7: get_stats() 返回所有字段...")

    index = PagedSSDCacheIndex(max_size_bytes=10 * 1024**3)
    stats = index.get_stats()

    required_fields = [
        'total_blocks', 'cold_blocks', 'hot_blocks',
        'total_size', 'max_size',
        'cold_evictions', 'hot_evictions', 'promotions'
    ]

    for field in required_fields:
        assert field in stats, f"Missing field: {field}"

    print("  ✅ 通过")


def test_8_scenario_agent_conversation():
    """测试 8: Agent 多轮对话场景"""
    print("测试 8: Agent 多轮对话场景...")

    index = PagedSSDCacheIndex(max_size_bytes=5 * 1024)  # 只允许 5KB

    # System prompt (会被频繁访问)
    system_meta = create_metadata("system_prompt", file_size=1024)

    # 用户 queries (一次性访问)
    user_metas = [create_metadata(f"user_query_{i}", file_size=1024) for i in range(10)]

    # 场景：system prompt 访问 10 次
    index.add(system_meta)
    for _ in range(9):
        index.touch(system_meta.block_hash)  # 第 2 次访问会晋升到 HOT

    # 添加 10 个用户 queries (都在 COLD queue)
    for meta in user_metas:
        index.add(meta)

    stats = index.get_stats()
    print(f"  添加后: cold={stats['cold_blocks']}, hot={stats['hot_blocks']}, size={stats['total_size']}")

    # 触发驱逐 (目标: 5KB)
    evicted = index.evict_until_size(5 * 1024)

    # System prompt 应该保留（在 HOT queue）
    # User queries 应该被驱逐（在 COLD queue）
    system_still_exists = index.contains(system_meta.block_hash)
    assert system_still_exists, "System prompt should be retained (in HOT queue)"

    stats = index.get_stats()
    print(f"  驱逐后: cold={stats['cold_blocks']}, hot={stats['hot_blocks']}, evicted={len(evicted)}")
    print(f"  ✅ 通过 (system prompt 保留在 HOT queue)")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("LRU-2 缓存驱逐策略测试")
    print("=" * 60)

    try:
        test_1_add_enters_cold_queue()
        test_2_second_touch_promotes_to_hot()
        test_3_cold_evicted_before_hot()
        test_4_lru_order_within_queue()
        test_5_touch_moves_to_mru()
        test_6_thread_safety()
        test_7_get_stats_returns_all_fields()
        test_8_scenario_agent_conversation()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
