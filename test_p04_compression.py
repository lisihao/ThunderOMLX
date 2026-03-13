#!/usr/bin/env python3
"""
P0-4 SSD Compression 验证脚本

验证点：
1. 压缩功能可开关
2. 压缩后文件大小减少 2-4x
3. 解压缩后数据正确
4. 向后兼容（旧文件仍可加载）
"""

import hashlib
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from src.omlx.cache.paged_ssd_cache import PagedSSDCacheManager


def create_test_kv_cache(num_layers: int = 4, seq_len: int = 256, d_model: int = 2048):
    """创建模拟 KV cache 数据"""
    cache_data = []
    for _ in range(num_layers):
        # 模拟 KV cache: (batch=1, num_heads=32, seq_len, head_dim=64)
        keys = mx.random.normal((1, 32, seq_len, 64))
        values = mx.random.normal((1, 32, seq_len, 64))
        cache_data.append((keys, values))
    return cache_data


def compute_hash(cache_data):
    """计算 cache_data 的 hash（简化版）"""
    h = hashlib.sha256()
    for keys, values in cache_data:
        h.update(keys.tobytes())
        h.update(values.tobytes())
    return h.digest()


def verify_data_correctness(original, loaded):
    """验证加载的数据与原始数据一致"""
    assert len(original) == len(loaded), "Layer count mismatch"

    for i, (orig_kv, load_kv) in enumerate(zip(original, loaded)):
        orig_k, orig_v = orig_kv
        load_k, load_v = load_kv

        assert orig_k.shape == load_k.shape, f"Layer {i} keys shape mismatch"
        assert orig_v.shape == load_v.shape, f"Layer {i} values shape mismatch"

        # 使用 allclose 检查数值一致性（允许浮点误差）
        assert mx.allclose(orig_k, load_k, atol=1e-5).item(), f"Layer {i} keys data mismatch"
        assert mx.allclose(orig_v, load_v, atol=1e-5).item(), f"Layer {i} values data mismatch"


def test_compression_enabled():
    """测试1: 压缩功能开启"""
    print("\n=== Test 1: Compression Enabled ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cache = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1 * 1024**3,  # 1GB
            enable_compression=True,
            compression_level=6,
        )

        # 创建测试数据
        cache_data = create_test_kv_cache(num_layers=4, seq_len=256)
        block_hash = compute_hash(cache_data)

        # 保存
        success = cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=256,
            model_name="test_model",
        )
        assert success, "Save failed"

        # 等待后台写入完成
        import time
        time.sleep(2)

        # 检查文件扩展名
        file_path = cache._get_file_path(block_hash)
        print(f"  File path: {file_path}")
        assert file_path.suffix == '.zst', f"Expected .zst, got {file_path.suffix}"
        assert file_path.exists(), "File not found"

        # 检查文件大小
        compressed_size = file_path.stat().st_size
        print(f"  Compressed size: {compressed_size / (1024**2):.2f} MB")

        # 加载并验证
        loaded_data = cache.load_block(block_hash)
        assert loaded_data is not None, "Load failed"

        verify_data_correctness(cache_data, loaded_data)
        print("  ✅ Data integrity verified")

        # 清理
        cache.shutdown(wait=True)


def test_compression_disabled():
    """测试2: 压缩功能关闭（向后兼容）"""
    print("\n=== Test 2: Compression Disabled ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cache = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1 * 1024**3,
            enable_compression=False,  # 关闭压缩
        )

        cache_data = create_test_kv_cache(num_layers=4, seq_len=256)
        block_hash = compute_hash(cache_data)

        success = cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=256,
        )
        assert success, "Save failed"

        import time
        time.sleep(2)

        # 检查文件扩展名（应该是 .safetensors）
        file_path = cache._get_file_path(block_hash)
        print(f"  File path: {file_path}")
        assert file_path.suffix == '.safetensors', f"Expected .safetensors, got {file_path.suffix}"
        assert file_path.exists(), "File not found"

        uncompressed_size = file_path.stat().st_size
        print(f"  Uncompressed size: {uncompressed_size / (1024**2):.2f} MB")

        # 加载并验证
        loaded_data = cache.load_block(block_hash)
        assert loaded_data is not None, "Load failed"

        verify_data_correctness(cache_data, loaded_data)
        print("  ✅ Data integrity verified")

        cache.shutdown(wait=True)


def test_compression_ratio():
    """测试3: 验证压缩比（应达到 2-4x）"""
    print("\n=== Test 3: Compression Ratio ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # 保存压缩版本
        cache_compressed = PagedSSDCacheManager(
            cache_dir=cache_dir / "compressed",
            max_size_bytes=1 * 1024**3,
            enable_compression=True,
            compression_level=6,
        )

        cache_data = create_test_kv_cache(num_layers=8, seq_len=512)  # 更大数据
        block_hash = compute_hash(cache_data)

        cache_compressed.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=512,
        )

        import time
        time.sleep(2)

        compressed_path = cache_compressed._get_file_path(block_hash)
        compressed_size = compressed_path.stat().st_size
        cache_compressed.shutdown(wait=True)

        # 保存未压缩版本
        cache_uncompressed = PagedSSDCacheManager(
            cache_dir=cache_dir / "uncompressed",
            max_size_bytes=1 * 1024**3,
            enable_compression=False,
        )

        cache_uncompressed.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=512,
        )

        time.sleep(2)

        uncompressed_path = cache_uncompressed._get_file_path(block_hash)
        uncompressed_size = uncompressed_path.stat().st_size
        cache_uncompressed.shutdown(wait=True)

        # 计算压缩比
        ratio = uncompressed_size / compressed_size
        print(f"  Uncompressed: {uncompressed_size / (1024**2):.2f} MB")
        print(f"  Compressed:   {compressed_size / (1024**2):.2f} MB")
        print(f"  Compression ratio: {ratio:.2f}x")

        assert ratio >= 2.0, f"Compression ratio {ratio:.2f}x < 2.0x (expected 2-4x)"
        assert ratio <= 5.0, f"Compression ratio {ratio:.2f}x > 5.0x (suspiciously high)"

        print(f"  ✅ Compression ratio {ratio:.2f}x is within expected range (2-4x)")


if __name__ == "__main__":
    print("Starting P0-4 SSD Compression Tests...")

    try:
        test_compression_enabled()
        test_compression_disabled()
        test_compression_ratio()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
