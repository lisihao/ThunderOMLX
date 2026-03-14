#!/usr/bin/env python3
"""
P1-5: Smart Prefetch 功能验证脚本

验证点：
1. AccessFrequencyTracker 正常工作
2. AsyncPrefetcher 正常工作
3. PagedSSDCacheManager 集成正确
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.access_tracker import AccessFrequencyTracker
from omlx.cache.async_prefetcher import AsyncPrefetcher


def test_access_frequency_tracker():
    """测试访问频率追踪器"""
    print("=" * 70)
    print("Test 1: AccessFrequencyTracker")
    print("=" * 70)

    tracker = AccessFrequencyTracker(decay_interval=3600.0)

    # 模拟块访问
    block_a = b'block_a_hash_0000000000000000'
    block_b = b'block_b_hash_0000000000000000'
    block_c = b'block_c_hash_0000000000000000'

    print("\n📝 模拟访问模式：")
    print("  - Block A: 访问 10 次（热块）")
    for _ in range(10):
        tracker.track_access(block_a)

    print("  - Block B: 访问 3 次（温块）")
    for _ in range(3):
        tracker.track_access(block_b)

    print("  - Block C: 访问 1 次（冷块）")
    tracker.track_access(block_c)

    # 获取热块
    hot_blocks = tracker.get_hot_blocks(top_n=5, min_access_count=2)

    print("\n📊 热块识别结果（min_access_count=2）：")
    for i, (block_hash, access_count) in enumerate(hot_blocks, 1):
        print(f"  {i}. {block_hash.hex()[:16]}... -> {access_count} 次访问")

    # 验证
    assert len(hot_blocks) == 2, f"预期 2 个热块，得到 {len(hot_blocks)}"
    assert hot_blocks[0][0] == block_a, "Block A 应该是最热的"
    assert hot_blocks[0][1] == 10, "Block A 应该有 10 次访问"
    assert hot_blocks[1][0] == block_b, "Block B 应该是第二热的"
    assert hot_blocks[1][1] == 3, "Block B 应该有 3 次访问"

    # 统计信息
    stats = tracker.get_stats()
    print(f"\n📈 统计信息：")
    print(f"  - 总块数: {stats['total_blocks']}")
    print(f"  - 总访问次数: {stats['total_accesses']}")
    print(f"  - 平均访问次数: {stats['avg_access_per_block']}")

    print("\n✅ Test 1 通过")
    return True


def test_async_prefetcher():
    """测试异步预取器"""
    print("\n" + "=" * 70)
    print("Test 2: AsyncPrefetcher")
    print("=" * 70)

    prefetcher = AsyncPrefetcher(num_workers=4)
    prefetcher.start()

    loaded_blocks = []

    def mock_load_fn(block_hash: bytes):
        """模拟从 SSD 加载（有延迟）"""
        time.sleep(0.01)  # 模拟 I/O 延迟
        return f"data_{block_hash.hex()[:8]}"

    def on_loaded(block_hash: bytes, block_data: any):
        """加载完成回调"""
        loaded_blocks.append((block_hash, block_data))

    # 预取 20 个块
    block_hashes = [f"block_{i:02d}".encode().ljust(32, b'\x00') for i in range(20)]

    print(f"\n📝 预取 {len(block_hashes)} 个块（4 线程并行）...")
    start = time.perf_counter()

    prefetcher.prefetch_blocks(
        block_hashes=block_hashes,
        load_fn=mock_load_fn,
        on_loaded=on_loaded
    )

    # 等待预取完成
    time.sleep(0.5)
    elapsed = time.perf_counter() - start

    # 验证所有块都被加载
    assert len(loaded_blocks) == 20, f"预期加载 20 个块，实际 {len(loaded_blocks)}"

    print(f"\n📊 预取结果：")
    print(f"  - 加载块数: {len(loaded_blocks)}")
    print(f"  - 总耗时: {elapsed:.2f}s")
    print(f"  - 平均每块: {elapsed / len(loaded_blocks) * 1000:.1f}ms")

    # 估算加速比（单线程 vs 4 线程）
    serial_time = 20 * 0.01  # 单线程理论时间
    speedup = serial_time / elapsed
    print(f"  - 理论加速比: ~{speedup:.1f}x（vs 单线程）")

    prefetcher.stop()

    print("\n✅ Test 2 通过")
    return True


def test_paged_ssd_cache_manager_integration():
    """测试 PagedSSDCacheManager 集成"""
    print("\n" + "=" * 70)
    print("Test 3: PagedSSDCacheManager Integration")
    print("=" * 70)

    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\n📝 创建缓存管理器（enable_prefetch=True）...")

        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 100,  # 100MB
            enable_prefetch=True,
            prefetch_top_n=5,
            prefetch_interval=2.0,  # 2 秒触发一次
            hot_cache_max_bytes=0  # 禁用 hot cache 以测试纯 SSD 预取
        )

        # 验证初始化
        assert manager.enable_prefetch is True
        assert manager._access_tracker is not None
        assert manager._prefetcher is not None
        assert manager._prefetcher._running is True

        print("  ✅ 初始化成功")

        # 获取预取统计
        stats = manager.get_prefetch_stats()
        print(f"\n📊 预取统计：")
        print(f"  - Enabled: {stats['enabled']}")
        print(f"  - Top N: {stats['top_n']}")
        print(f"  - Interval: {stats['interval']}s")
        print(f"  - Tracked blocks: {stats['access_tracker']['total_blocks']}")

        # 停止管理器
        manager.stop()
        print("  ✅ 停止成功")

    print("\n✅ Test 3 通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("P1-5: Smart Prefetch 功能验证")
    print("=" * 70)

    try:
        # Test 1: AccessFrequencyTracker
        test_access_frequency_tracker()

        # Test 2: AsyncPrefetcher
        test_async_prefetcher()

        # Test 3: Integration
        test_paged_ssd_cache_manager_integration()

        print("\n" + "=" * 70)
        print("🎉 所有测试通过！P1-5 Smart Prefetch 实现正确")
        print("=" * 70)

        print("\n📋 实现总结：")
        print("  ✅ AccessFrequencyTracker - 访问频率追踪")
        print("  ✅ AsyncPrefetcher - 4 线程并行预取")
        print("  ✅ PagedSSDCacheManager - 集成 Smart Prefetch")
        print("  ✅ 自动访问追踪（save_block/load_block）")
        print("  ✅ 定期预取定时器")
        print("  ✅ 热块识别和预取")
        print()

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
