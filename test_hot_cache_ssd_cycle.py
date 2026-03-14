#!/usr/bin/env python3
"""
验证改进后的 hot cache hits 统计

场景：
1. Save → Hot cache（不计入 hot_cache_hits）
2. Evict → SSD（加入 index）
3. Load → 从 SSD 加载回 hot cache（应计入 hot_cache_hits）
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    import numpy as np


def create_mock_kv_cache(num_tokens, num_layers=32, embd_dim=128):
    """创建模拟的 KV cache 数据"""
    if HAS_MLX:
        cache_data = []
        for _ in range(num_layers):
            keys = mx.random.normal(shape=(num_tokens, embd_dim))
            values = mx.random.normal(shape=(num_tokens, embd_dim))
            cache_data.append((keys, values))
        return cache_data
    else:
        cache_data = []
        for _ in range(num_layers):
            keys = np.random.randn(num_tokens, embd_dim).astype(np.float16)
            values = np.random.randn(num_tokens, embd_dim).astype(np.float16)
            cache_data.append((keys, values))
        return cache_data


def test_hot_cache_ssd_cycle():
    """测试 Hot cache → SSD → Hot cache 循环"""
    print("=" * 70)
    print("测试 Hot Cache → SSD → Hot Cache 统计准确性")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  MLX 未安装，使用 NumPy 模拟")

    with tempfile.TemporaryDirectory() as cache_dir:
        # 创建小容量 hot cache（2MB），方便触发 eviction
        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=100 * 1024 * 1024,  # 100MB SSD
            hot_cache_max_bytes=2 * 1024 * 1024,  # 2MB hot cache（小容量，容易触发 eviction）
            enable_prefetch=False,  # 关闭预取，简化测试
            enable_checksum=False,  # 关闭 checksum，加快速度
            enable_compression=False,  # 关闭压缩，加快速度
        )

        print(f"\n✅ 缓存管理器创建完成（Hot cache: 2MB）")

        # 1. 保存第一个块（进入 hot cache）
        print("\n📝 步骤 1: 保存第一个块（512 tokens）")
        block1_cache = create_mock_kv_cache(512)
        block1_hash = b"block_1_hash_" + b"x" * 20

        manager.save_block(
            block_hash=block1_hash,
            cache_data=block1_cache,
            token_count=512
        )

        # 检查统计
        stats = manager.get_stats()
        print(f"  - Saves: {stats.saves}")
        print(f"  - Hot cache hits: {stats.hot_cache_hits}")
        assert stats.hot_cache_hits == 0, "首次保存不应计入 hot_cache_hits"

        # 2. 加载第一个块（从 hot cache）
        print("\n📝 步骤 2: 加载第一个块（从 hot cache，未落盘）")
        loaded1 = manager.load_block(block1_hash)
        assert loaded1 is not None, "应该能从 hot cache 加载"

        stats = manager.get_stats()
        print(f"  - Loads: {stats.loads}")
        print(f"  - Hot cache hits: {stats.hot_cache_hits}")
        assert stats.hot_cache_hits == 0, "从 hot cache 加载未落盘的块不应计入 hot_cache_hits"

        # 3. 保存足够多的块，触发 eviction（将 block1 evict 到 SSD）
        print("\n📝 步骤 3: 保存多个块，触发 eviction")
        num_evict_blocks = 5
        for i in range(num_evict_blocks):
            block_cache = create_mock_kv_cache(512)  # ~2MB per block
            block_hash = f"evict_block_{i}".encode() + b"x" * 10
            manager.save_block(
                block_hash=block_hash,
                cache_data=block_cache,
                token_count=512
            )
            print(f"  - 保存块 {i+1}/{num_evict_blocks}")

        # 等待后台写入完成
        print("\n⏳ 等待后台写入完成...")
        manager.flush(timeout=10.0)

        stats = manager.get_stats()
        print(f"  - Hot cache evictions: {stats.hot_cache_evictions}")
        assert stats.hot_cache_evictions > 0, "应该触发了 eviction"

        # 4. 重新加载 block1（从 SSD 加载回 hot cache）
        print("\n📝 步骤 4: 第一次重新加载 block1（从 SSD → hot cache）")
        loaded1_again = manager.load_block(block1_hash)
        assert loaded1_again is not None, "应该能从 SSD 加载"

        stats = manager.get_stats()
        print(f"  - Loads: {stats.loads}")
        print(f"  - Hot cache hits: {stats.hot_cache_hits}")
        print(f"  ℹ️  第一次从 SSD 加载不计入 hot_cache_hits（符合预期）")

        # 5. 再次加载 block1（现在应该从 hot cache 读取）
        print("\n📝 步骤 5: 第二次加载 block1（从 hot cache，已落盘）")
        loaded1_third = manager.load_block(block1_hash)
        assert loaded1_third is not None, "应该能从 hot cache 加载"

        stats = manager.get_stats()
        print(f"  - Loads: {stats.loads}")
        print(f"  - Hot cache hits: {stats.hot_cache_hits}")

        # 验收标准：从 hot cache 加载已落盘的块应该计入 hot_cache_hits
        if stats.hot_cache_hits > 0:
            print(f"\n✅ 测试通过！从 hot cache 加载已落盘块计入 hot_cache_hits（{stats.hot_cache_hits} 次）")
            return True
        else:
            print(f"\n❌ 测试失败！从 hot cache 加载已落盘块未计入 hot_cache_hits")
            return False


if __name__ == "__main__":
    success = test_hot_cache_ssd_cycle()
    exit(0 if success else 1)
