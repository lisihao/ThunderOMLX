"""
Cache-Aligned Chunking 单元测试

验证 Phase 2.5-1: chunk 边界对齐到 KV Cache block_size 倍数
以及 align_mode 功能开关（默认 "none"，MLX 不需要 block 对齐）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from omlx.chunking.dynamic_chunker import DynamicChunker
from omlx.chunking.types import ContentType, Boundary, BoundaryType


# === 对齐算法单元测试（直接测试 _align_to_block_boundary 方法） ===

def test_align_up_normal():
    """向上对齐：4089 -> 4096 (256*16)"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256)
    result = chunker._align_to_block_boundary(4089, 0, 3584, 4608)
    assert result == 4096, f"Expected 4096, got {result}"


def test_align_already_aligned():
    """已对齐的不变：4096 -> 4096"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256)
    result = chunker._align_to_block_boundary(4096, 0, 3584, 4608)
    assert result == 4096


def test_align_up_with_offset():
    """带偏移的向上对齐：start=4096, end=8185 -> 8192"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256)
    result = chunker._align_to_block_boundary(8185, 4096, 3584, 4608)
    assert result == 8192, f"Expected 8192, got {result}"


def test_align_down_when_up_exceeds_max():
    """向上超 max_size 时向下对齐"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256)
    result = chunker._align_to_block_boundary(4600, 0, 3584, 4608)
    assert result == 4608, f"Expected 4608, got {result}"

    result2 = chunker._align_to_block_boundary(4609, 0, 3584, 4608)
    assert result2 == 4608, f"Expected 4608, got {result2}"


def test_align_fallback():
    """极端情况：aligned_up 在范围内则返回"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256)
    result = chunker._align_to_block_boundary(4050, 0, 4000, 4100)
    assert result == 4096


def test_align_disabled_with_block_size_1():
    """block_size=1 时禁用对齐"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=1)
    result = chunker._align_to_block_boundary(4089, 0, 3584, 4608)
    assert result == 4089


# === align_mode 功能开关测试 ===

def test_default_align_mode_is_none():
    """默认 align_mode='none'：纯语义分块，不做 block 对齐"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125)
    assert chunker.align_mode == "none"

    tokens = list(range(20000))
    boundaries = [
        Boundary(token_offset=4089, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
        Boundary(token_offset=8200, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
    ]
    chunks = chunker.chunk(tokens, boundaries, ContentType.GENERIC)

    # align_mode="none" 时，第一个 chunk 应切在语义边界 4089，不会 snap 到 4096
    assert chunks[0].end == 4089, f"Expected 4089 (semantic boundary), got {chunks[0].end}"
    print(f"Chunk 0: semantic boundary preserved at {chunks[0].end}")


def test_hard_align_mode():
    """align_mode='hard'：强制对齐到 block_size 倍数"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256, align_mode="hard")

    tokens = list(range(20000))
    chunks = chunker.chunk(tokens, [], ContentType.GENERIC)

    for i, chunk in enumerate(chunks[:-1]):
        chunk_size = chunk.end - chunk.start
        assert chunk_size % 256 == 0, (
            f"Chunk {i}: size {chunk_size} not aligned to 256 "
            f"(start={chunk.start}, end={chunk.end})"
        )

    last = chunks[-1]
    print(f"Last chunk: start={last.start}, end={last.end}, size={last.size}")
    assert last.end == 20000


def test_hard_align_with_boundaries():
    """align_mode='hard' + 语义边界：边界 snap 到 256 倍数"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256, align_mode="hard")

    boundaries = [
        Boundary(token_offset=4089, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
        Boundary(token_offset=8200, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
    ]

    tokens = list(range(12000))
    chunks = chunker.chunk(tokens, boundaries, ContentType.GENERIC)

    # 语义边界在 4089，hard 对齐后 snap 到 4096
    assert chunks[0].end == 4096, f"Expected 4096, got {chunks[0].end}"
    print(f"Chunk 0: boundary at 4089 -> aligned to {chunks[0].end}")

    for i, chunk in enumerate(chunks[:-1]):
        chunk_size = chunk.end - chunk.start
        assert chunk_size % 256 == 0, f"Chunk {i} not aligned: size={chunk_size}"


def test_soft_align_mode():
    """align_mode='soft'：仅 drift < block_size/4 时对齐"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256, align_mode="soft")

    # drift = 4096 - 4089 = 7 < 64 (256/4) → 应该对齐
    boundaries = [
        Boundary(token_offset=4089, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
    ]
    tokens = list(range(8000))
    chunks = chunker.chunk(tokens, boundaries, ContentType.GENERIC)
    assert chunks[0].end == 4096, f"Soft align should snap 4089->4096, got {chunks[0].end}"
    print(f"Soft align: 4089 -> {chunks[0].end} (drift=7, threshold=64)")


# === API 兼容性与质量测试 ===

def test_api_backward_compatibility():
    """API 向后兼容：默认值检查"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125)
    assert chunker.block_size == 256
    assert chunker.align_mode == "none"

    chunker2 = DynamicChunker(target_size=4096, flexibility=0.125, block_size=512, align_mode="hard")
    assert chunker2.block_size == 512
    assert chunker2.align_mode == "hard"


def test_chunk_quality_maintained():
    """纯语义分块质量应 >= 0.80"""
    from omlx.chunking.quality_validator import QualityValidator

    chunker = DynamicChunker(target_size=4096, flexibility=0.125)
    validator = QualityValidator()

    boundaries = [
        Boundary(token_offset=4000, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
        Boundary(token_offset=8100, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
        Boundary(token_offset=12050, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
        Boundary(token_offset=16200, text_offset=0, type=BoundaryType.PARAGRAPH, strength=1.0),
    ]

    tokens = list(range(20000))
    chunks = chunker.chunk(tokens, boundaries, ContentType.GENERIC)

    quality = validator.validate(chunks, boundaries, len(tokens))
    print(f"Quality (align_mode=none): {quality}")
    assert quality.overall_score >= 0.80, f"Quality too low: {quality.overall_score:.2f}"


def test_128k_scenario():
    """128K tokens 场景：默认 none 模式下 token 总数正确"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125)
    tokens = list(range(121388))

    chunks = chunker.chunk(tokens, [], ContentType.GENERIC)

    total_tokens = sum(c.size for c in chunks)
    assert total_tokens == 121388, f"Token count mismatch: {total_tokens}"
    print(f"128K (align_mode=none): {len(chunks)} chunks, total={total_tokens}")


def test_128k_hard_align():
    """128K tokens + hard 对齐：所有非末尾 chunk 对齐到 256"""
    chunker = DynamicChunker(target_size=4096, flexibility=0.125, block_size=256, align_mode="hard")
    tokens = list(range(121388))

    chunks = chunker.chunk(tokens, [], ContentType.GENERIC)

    total_tokens = sum(c.size for c in chunks)
    assert total_tokens == 121388, f"Token count mismatch: {total_tokens}"

    aligned_count = sum(1 for c in chunks[:-1] if (c.end - c.start) % 256 == 0)
    alignment_rate = aligned_count / max(len(chunks) - 1, 1)
    print(f"128K (align_mode=hard): {len(chunks)} chunks, alignment rate={alignment_rate:.1%}")
    assert alignment_rate == 1.0, f"Not all chunks aligned: {alignment_rate:.1%}"


if __name__ == "__main__":
    tests = [
        # 对齐算法单元测试
        test_align_up_normal,
        test_align_already_aligned,
        test_align_up_with_offset,
        test_align_down_when_up_exceeds_max,
        test_align_fallback,
        test_align_disabled_with_block_size_1,
        # align_mode 功能开关
        test_default_align_mode_is_none,
        test_hard_align_mode,
        test_hard_align_with_boundaries,
        test_soft_align_mode,
        # API 兼容性与质量
        test_api_backward_compatibility,
        test_chunk_quality_maintained,
        test_128k_scenario,
        test_128k_hard_align,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"结果: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} tests failed")
