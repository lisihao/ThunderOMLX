#!/usr/bin/env python3
"""
P1-7: Adaptive Chunk Prefill 功能测试
"""

import pytest
from omlx.cache.chunk_adapter import AdaptiveChunkCalculator


def test_short_prompt_no_chunking():
    """测试短 prompt 不分块"""
    calculator = AdaptiveChunkCalculator()

    chunk_size = calculator.compute_chunk_size(prompt_length=100)
    assert chunk_size == 100  # 不分块

    chunks = calculator.split_into_chunks(prompt_length=100)
    assert len(chunks) == 1
    assert chunks[0] == (0, 100)


def test_medium_prompt_chunking():
    """测试中等 prompt 分块"""
    calculator = AdaptiveChunkCalculator()

    chunk_size = calculator.compute_chunk_size(prompt_length=512)
    assert chunk_size == 128  # 基准 chunk size

    chunks = calculator.split_into_chunks(prompt_length=512)
    assert len(chunks) == 4  # 512 / 128 = 4
    assert chunks[0] == (0, 128)
    assert chunks[1] == (128, 256)
    assert chunks[2] == (256, 384)
    assert chunks[3] == (384, 512)


def test_large_prompt_chunking():
    """测试长 prompt 分块"""
    calculator = AdaptiveChunkCalculator()

    chunk_size = calculator.compute_chunk_size(prompt_length=2048)
    assert chunk_size == 256  # 长 prompt 使用更大的 chunk

    chunks = calculator.split_into_chunks(prompt_length=2048)
    assert len(chunks) == 8  # 2048 / 256 = 8


def test_very_large_prompt_chunking():
    """测试超长 prompt 分块"""
    calculator = AdaptiveChunkCalculator()

    chunk_size = calculator.compute_chunk_size(prompt_length=8192)
    assert chunk_size == 512  # 超长 prompt 使用最大 chunk

    chunks = calculator.split_into_chunks(prompt_length=8192)
    assert len(chunks) == 16  # 8192 / 512 = 16


def test_alignment():
    """测试缓存块对齐"""
    calculator = AdaptiveChunkCalculator(cache_block_size=64)

    # 128 已经是 64 的倍数
    chunk_size = calculator.compute_chunk_size(prompt_length=512)
    assert chunk_size % 64 == 0

    # 256 也是 64 的倍数
    chunk_size = calculator.compute_chunk_size(prompt_length=2048)
    assert chunk_size % 64 == 0


def test_alignment_disabled():
    """测试禁用对齐"""
    calculator = AdaptiveChunkCalculator(
        cache_block_size=64,
        enable_alignment=False
    )

    # 不对齐时直接使用基准 chunk size
    chunk_size = calculator.compute_chunk_size(prompt_length=512)
    assert chunk_size == 128  # 不一定是 64 的倍数（虽然恰好是）


def test_memory_limit():
    """测试内存限制"""
    calculator = AdaptiveChunkCalculator()

    # 模拟低内存场景
    small_memory = 100 * 1024**2  # 100MB
    chunk_size = calculator.compute_chunk_size(
        prompt_length=4096,
        available_memory=small_memory
    )

    # Chunk size 应该被限制
    chunk_memory = chunk_size * 4096  # bytes per token
    assert chunk_memory <= small_memory * 0.5


def test_memory_check_disabled():
    """测试禁用内存检查"""
    calculator = AdaptiveChunkCalculator(enable_memory_check=False)

    # 即使内存很小，也不限制 chunk size
    small_memory = 1 * 1024**2  # 1MB
    chunk_size = calculator.compute_chunk_size(
        prompt_length=4096,
        available_memory=small_memory
    )

    # 应该使用标准的超长 prompt chunk size
    assert chunk_size == 512


def test_edge_case_exact_multiple():
    """测试边界情况：prompt 长度正好是 chunk size 的倍数"""
    calculator = AdaptiveChunkCalculator()

    chunks = calculator.split_into_chunks(prompt_length=256, chunk_size=128)
    assert len(chunks) == 2
    assert chunks[0] == (0, 128)
    assert chunks[1] == (128, 256)


def test_edge_case_not_exact_multiple():
    """测试边界情况：prompt 长度不是 chunk size 的倍数"""
    calculator = AdaptiveChunkCalculator()

    chunks = calculator.split_into_chunks(prompt_length=300, chunk_size=128)
    assert len(chunks) == 3
    assert chunks[0] == (0, 128)
    assert chunks[1] == (128, 256)
    assert chunks[2] == (256, 300)  # 最后一个 chunk 只有 44 tokens


def test_get_stats():
    """测试获取统计信息"""
    calculator = AdaptiveChunkCalculator(cache_block_size=64)

    stats = calculator.get_stats()
    assert stats['cache_block_size'] == 64
    assert stats['enable_alignment'] is True
    assert stats['enable_memory_check'] is True
    assert stats['short_prompt_threshold'] == 128
    assert stats['medium_chunk_size'] == 128
    assert stats['large_chunk_size'] == 256
    assert stats['very_large_chunk_size'] == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
