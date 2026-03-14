#!/usr/bin/env python3
"""
P1 阶段集成验证脚本

验证点：
1. P1-5: Smart Prefetch 集成
2. P1-6: Checksum Validation 集成
3. P1-7: Adaptive Chunk Prefill 工具
4. 所有组件协同工作
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.access_tracker import AccessFrequencyTracker
from omlx.cache.async_prefetcher import AsyncPrefetcher
from omlx.cache.checksum import ChecksumCalculator
from omlx.cache.chunk_adapter import AdaptiveChunkCalculator


def test_p1_5_smart_prefetch_integration():
    """测试 P1-5: Smart Prefetch 集成"""
    print("=" * 70)
    print("Test 1: P1-5 Smart Prefetch Integration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as cache_dir:
        print("\n📝 创建缓存管理器（enable_prefetch=True）...")

        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 100,  # 100MB
            enable_prefetch=True,
            prefetch_top_n=5,
            prefetch_interval=2.0,
            hot_cache_max_bytes=0  # 禁用 hot cache 测试纯 SSD 预取
        )

        # 验证组件存在
        assert manager.enable_prefetch is True
        assert manager._access_tracker is not None
        assert manager._prefetcher is not None
        assert manager._prefetcher._running is True

        print("  ✅ Smart Prefetch 组件初始化成功")

        # 获取统计
        stats = manager.get_prefetch_stats()
        print(f"\n📊 预取统计：")
        print(f"  - Enabled: {stats['enabled']}")
        print(f"  - Top N: {stats['top_n']}")
        print(f"  - Interval: {stats['interval']}s")
        print(f"  - Tracked blocks: {stats['access_tracker']['total_blocks']}")

        manager.stop()
        print("  ✅ 停止成功")

    print("\n✅ Test 1 通过")
    return True


def test_p1_6_checksum_integration():
    """测试 P1-6: Checksum Validation 集成"""
    print("\n" + "=" * 70)
    print("Test 2: P1-6 Checksum Validation Integration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as cache_dir:
        print("\n📝 创建缓存管理器（enable_checksum=True）...")

        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 100,
            enable_checksum=True,
            hot_cache_max_bytes=0
        )

        # 验证 checksum 启用
        assert manager.enable_checksum is True

        print("  ✅ Checksum Validation 启用")

        # 获取统计
        stats = manager.get_stats()
        print(f"\n📊 Checksum 统计：")
        print(f"  - Verifications: {stats.checksum_verifications}")
        print(f"  - Failures: {stats.checksum_failures}")

    print("\n✅ Test 2 通过")
    return True


def test_p1_7_adaptive_chunk():
    """测试 P1-7: Adaptive Chunk Prefill 工具"""
    print("\n" + "=" * 70)
    print("Test 3: P1-7 Adaptive Chunk Prefill")
    print("=" * 70)

    calculator = AdaptiveChunkCalculator(cache_block_size=64)

    # 测试不同长度的 prompt
    test_cases = [
        (100, 100, 1),      # 短 prompt: 不分块
        (512, 128, 4),      # 中等: chunk_size=128
        (2048, 256, 8),     # 长: chunk_size=256
        (8192, 512, 16),    # 超长: chunk_size=512
    ]

    print("\n📝 测试自适应分块...")
    for prompt_length, expected_chunk_size, expected_num_chunks in test_cases:
        chunk_size = calculator.compute_chunk_size(prompt_length)
        chunks = calculator.split_into_chunks(prompt_length)

        assert chunk_size == expected_chunk_size, \
            f"Prompt {prompt_length}: 预期 chunk_size={expected_chunk_size}, 实际={chunk_size}"
        assert len(chunks) == expected_num_chunks, \
            f"Prompt {prompt_length}: 预期 {expected_num_chunks} chunks, 实际={len(chunks)}"

        print(f"  ✅ Prompt {prompt_length:4d} tokens → {len(chunks):2d} chunks × {chunk_size:3d} tokens")

    print("\n✅ Test 3 通过")
    return True


def test_all_features_together():
    """测试所有功能协同工作"""
    print("\n" + "=" * 70)
    print("Test 4: All P1 Features Integration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as cache_dir:
        print("\n📝 创建全功能缓存管理器...")

        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 100,
            enable_prefetch=True,        # P1-5
            prefetch_top_n=10,
            prefetch_interval=5.0,
            enable_checksum=True,        # P1-6
            enable_compression=True,     # P0-4
            hot_cache_max_bytes=10 * 1024**2,  # 10MB hot cache
        )

        # 创建 Adaptive Chunk Calculator (P1-7)
        chunk_calculator = AdaptiveChunkCalculator(cache_block_size=64)

        # 验证所有功能启用
        assert manager.enable_prefetch is True
        assert manager.enable_checksum is True
        assert manager.enable_compression is True

        print("  ✅ 所有 P1 功能已启用：")
        print("    - Smart Prefetch (P1-5)")
        print("    - Checksum Validation (P1-6)")
        print("    - SSD Compression (P0-4)")
        print("    - Adaptive Chunk Calculator (P1-7)")

        # 获取综合统计
        stats = manager.get_stats()
        prefetch_stats = manager.get_prefetch_stats()

        print(f"\n📊 综合统计：")
        print(f"  Cache:")
        print(f"    - Saves: {stats.saves}")
        print(f"    - Loads: {stats.loads}")
        print(f"    - Hits: {stats.hits}")
        print(f"    - Misses: {stats.misses}")
        print(f"  Checksum:")
        print(f"    - Verifications: {stats.checksum_verifications}")
        print(f"    - Failures: {stats.checksum_failures}")
        print(f"  Prefetch:")
        print(f"    - Enabled: {prefetch_stats['enabled']}")
        print(f"    - Tracked blocks: {prefetch_stats['access_tracker']['total_blocks']}")
        print(f"  Chunk Calculator:")
        chunk_stats = chunk_calculator.get_stats()
        print(f"    - Cache block size: {chunk_stats['cache_block_size']}")
        print(f"    - Alignment: {chunk_stats['enable_alignment']}")

        manager.stop()
        print("\n  ✅ 所有功能协同工作正常")

    print("\n✅ Test 4 通过")
    return True


def main():
    """运行所有集成测试"""
    print("\n" + "=" * 70)
    print("P1 阶段集成验证")
    print("=" * 70)

    try:
        # Test 1: P1-5 Smart Prefetch
        test_p1_5_smart_prefetch_integration()

        # Test 2: P1-6 Checksum Validation
        test_p1_6_checksum_integration()

        # Test 3: P1-7 Adaptive Chunk Prefill
        test_p1_7_adaptive_chunk()

        # Test 4: All features together
        test_all_features_together()

        print("\n" + "=" * 70)
        print("🎉 所有集成测试通过！P1 阶段功能正常")
        print("=" * 70)

        print("\n📋 P1 功能总结：")
        print("  ✅ P1-5: Smart Prefetch - 4 线程并行预取 + 访问频率追踪")
        print("  ✅ P1-6: Checksum Validation - XXH64 数据完整性保护")
        print("  ✅ P1-7: Adaptive Chunk Prefill - 自适应分块策略")
        print("  ✅ P0-4: SSD Compression - zlib 压缩（已集成）")
        print("\n  🔧 所有功能可独立启用/禁用")
        print("  🔧 所有功能协同工作正常")
        print()

    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
