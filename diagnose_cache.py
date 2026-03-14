#!/usr/bin/env python3
"""诊断缓存保存和加载"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not available")
    sys.exit(1)


def create_mock_cache_data(num_layers=4, seq_len=32, embd_dim=32):
    """创建小型模拟数据"""
    cache_data = []
    for _ in range(num_layers):
        keys = mx.random.normal(shape=(seq_len, embd_dim))
        values = mx.random.normal(shape=(seq_len, embd_dim))
        cache_data.append((keys, values))
    return cache_data


def main():
    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"Cache dir: {cache_dir}")

        # 创建管理器
        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 100,
            enable_prefetch=False,
            enable_checksum=False,
            hot_cache_max_bytes=0
        )

        # 保存 3 个块
        print("\n保存 3 个块...")
        block_hashes = []
        for i in range(3):
            block_hash = f"test_{i:02d}".encode().ljust(32, b'\x00')
            cache_data = create_mock_cache_data()
            result = manager.save_block(block_hash, cache_data, token_count=32)
            print(f"  Block {i}: save_block returned {result}")
            block_hashes.append(block_hash)

        # Flush
        print("\nFlush...")
        if manager.flush(timeout=10.0):
            print("  ✅ Flush 成功")
        else:
            print("  ❌ Flush 超时")

        # 检查统计
        stats = manager.get_stats()
        print(f"\n统计:")
        print(f"  Saves: {stats.saves}")
        print(f"  Num files: {stats.num_files}")

        # 检查文件
        print(f"\n文件系统检查:")
        cache_files = list(Path(cache_dir).rglob("*.safetensors*"))
        print(f"  找到 {len(cache_files)} 个缓存文件")
        for f in cache_files:
            print(f"    - {f.name} ({f.stat().st_size} bytes)")

        # 尝试加载
        print(f"\n加载 3 个块...")
        for i, block_hash in enumerate(block_hashes):
            data = manager.load_block(block_hash)
            if data:
                print(f"  Block {i}: ✅ 加载成功（{len(data)} layers）")
            else:
                print(f"  Block {i}: ❌ 加载失败")

        manager.stop()


if __name__ == "__main__":
    main()
