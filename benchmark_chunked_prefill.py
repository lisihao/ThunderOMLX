#!/usr/bin/env python3
"""
Benchmark script for Chunked Prefill MVP.

This script demonstrates:
1. How to use ChunkedPrefillEngine
2. Memory usage comparison
3. Timing measurement
4. Cache merging verification

Usage:
    python3 benchmark_chunked_prefill.py

Environment variables:
    OMLX_ENABLE_CHUNKED_PREFILL=true  - Enable chunking
    OMLX_CHUNK_SIZE=512                - Chunk size
    OMLX_MIN_TOKENS_FOR_CHUNKING=1024  - Minimum tokens to chunk
"""

import os
import time
import logging
from typing import Tuple

import mlx.core as mx

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omlx.chunked_prefill import ChunkedPrefillEngine, ChunkedPrefillConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MockModel:
    """Simple mock model for benchmarking cache operations."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def forward(
        self,
        tokens: mx.array,
        cache: list = None,
    ) -> Tuple[mx.array, list]:
        """Mock forward pass that creates synthetic cache."""
        seq_len = tokens.shape[0] if tokens.ndim == 1 else tokens.shape[1]
        batch_size = 1

        # Simulate attention layers creating KV caches
        # Each layer has (key, value) with shape (batch, seq_len, hidden_dim)
        new_cache = []
        for _ in range(12):  # 12 layers (like typical LLM)
            k = mx.random.normal((batch_size, seq_len, self.hidden_dim))
            v = mx.random.normal((batch_size, seq_len, self.hidden_dim))
            new_cache.append((k, v))

        # Simulate logits output
        logits = mx.random.normal((batch_size, seq_len, 32000))

        # Simulate actual computation
        mx.eval(logits)

        return logits, new_cache


def benchmark_traditional_vs_chunked(prompt_length: int = 2048):
    """Compare traditional vs chunked prefill."""
    logger.info("=" * 80)
    logger.info(f"Benchmark: Prompt Length = {prompt_length} tokens")
    logger.info("=" * 80)

    model = MockModel()
    tokens = mx.array([1] * prompt_length)

    # Test 1: Traditional prefill
    logger.info("\n--- Traditional Prefill ---")
    start = time.time()
    logits_trad, cache_trad = model.forward(tokens, None)
    mx.eval(logits_trad)
    traditional_time = time.time() - start

    logger.info(f"Time: {traditional_time:.3f}s")
    logger.info(f"Output shape: {logits_trad.shape}")
    logger.info(f"Cache layers: {len(cache_trad)}")
    logger.info(f"Cache memory: ~{len(cache_trad) * 2 * prompt_length * 128 / 1e6:.1f} MB")

    # Test 2: Chunked prefill (disabled)
    logger.info("\n--- Chunked Prefill (DISABLED) ---")
    config = ChunkedPrefillConfig(enable_chunking=False)
    engine = ChunkedPrefillEngine(model, config)

    def forward_wrapper(m, t, c):
        return m.forward(t, cache=c)

    start = time.time()
    logits_no_chunk, cache_no_chunk = engine.prefill(
        model, tokens, None, prefill_fn=forward_wrapper
    )
    mx.eval(logits_no_chunk)
    no_chunk_time = time.time() - start

    logger.info(f"Time: {no_chunk_time:.3f}s")
    logger.info(f"Output shape: {logits_no_chunk.shape}")
    logger.info(f"Cache layers: {len(cache_no_chunk)}")

    # Test 3: Chunked prefill (enabled)
    logger.info("\n--- Chunked Prefill (ENABLED, chunk_size=512) ---")
    config = ChunkedPrefillConfig(
        enable_chunking=True,
        chunk_size=512,
        min_tokens_for_chunking=1024,
    )
    engine = ChunkedPrefillEngine(model, config)

    start = time.time()
    logits_chunked, cache_chunked = engine.prefill(
        model, tokens, None, prefill_fn=forward_wrapper
    )
    mx.eval(logits_chunked)
    chunked_time = time.time() - start

    num_chunks = (prompt_length + 512 - 1) // 512
    logger.info(f"Time: {chunked_time:.3f}s ({num_chunks} chunks)")
    logger.info(f"Output shape: {logits_chunked.shape}")
    logger.info(f"Cache layers: {len(cache_chunked)}")
    logger.info(f"Cache memory: ~{len(cache_chunked) * 2 * prompt_length * 128 / 1e6:.1f} MB")

    # Comparison
    logger.info("\n--- Comparison ---")
    logger.info(f"Traditional:  {traditional_time:.3f}s")
    logger.info(f"No chunking:  {no_chunk_time:.3f}s (should be ~equal)")
    logger.info(f"Chunked:      {chunked_time:.3f}s")
    logger.info(f"Overhead:     {(chunked_time / traditional_time - 1) * 100:.1f}%")

    # Verify cache correctness
    logger.info("\n--- Cache Verification ---")
    if len(cache_chunked) == len(cache_trad):
        logger.info("✓ Cache layer count matches")
    else:
        logger.warning(f"✗ Cache layer count mismatch: {len(cache_chunked)} vs {len(cache_trad)}")

    # Check cache shapes
    for i, (c_chunked, c_trad) in enumerate(zip(cache_chunked, cache_trad)):
        if isinstance(c_chunked, tuple):
            k_chunked, v_chunked = c_chunked
            k_trad, v_trad = c_trad
            if k_chunked.shape == k_trad.shape and v_chunked.shape == v_trad.shape:
                logger.info(
                    f"✓ Layer {i}: Cache shapes match {k_chunked.shape} "
                    f"(key), {v_chunked.shape} (value)"
                )
            else:
                logger.warning(
                    f"✗ Layer {i}: Shape mismatch - "
                    f"chunked: {k_chunked.shape}, traditional: {k_trad.shape}"
                )


def benchmark_chunk_sizes(prompt_length: int = 2048):
    """Benchmark different chunk sizes."""
    logger.info("\n" + "=" * 80)
    logger.info(f"Benchmark: Different Chunk Sizes (prompt_length={prompt_length})")
    logger.info("=" * 80)

    model = MockModel()
    tokens = mx.array([1] * prompt_length)

    def forward_wrapper(m, t, c):
        return m.forward(t, cache=c)

    chunk_sizes = [256, 512, 1024, 2048]

    for chunk_size in chunk_sizes:
        config = ChunkedPrefillConfig(
            enable_chunking=True,
            chunk_size=chunk_size,
            min_tokens_for_chunking=1024,
        )
        engine = ChunkedPrefillEngine(model, config)

        start = time.time()
        logits, cache = engine.prefill(model, tokens, None, prefill_fn=forward_wrapper)
        mx.eval(logits)
        elapsed = time.time() - start

        num_chunks = (prompt_length + chunk_size - 1) // chunk_size
        logger.info(
            f"Chunk size: {chunk_size:4d} | Time: {elapsed:.3f}s | "
            f"Chunks: {num_chunks:2d}"
        )


def benchmark_prompt_lengths():
    """Benchmark different prompt lengths."""
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark: Different Prompt Lengths")
    logger.info("=" * 80)

    model = MockModel()
    config = ChunkedPrefillConfig(
        enable_chunking=True,
        chunk_size=512,
        min_tokens_for_chunking=1024,
    )
    engine = ChunkedPrefillEngine(model, config)

    def forward_wrapper(m, t, c):
        return m.forward(t, cache=c)

    prompt_lengths = [512, 1024, 2048, 4096, 8192]

    for length in prompt_lengths:
        tokens = mx.array([1] * length)

        start = time.time()
        logits, cache = engine.prefill(model, tokens, None, prefill_fn=forward_wrapper)
        mx.eval(logits)
        elapsed = time.time() - start

        num_chunks = (length + 512 - 1) // 512
        logger.info(
            f"Prompt length: {length:5d} | Time: {elapsed:.3f}s | "
            f"Chunks: {num_chunks:2d}"
        )


def benchmark_cache_merging():
    """Benchmark cache merging overhead."""
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark: Cache Merging Overhead")
    logger.info("=" * 80)

    # Create synthetic caches
    num_layers = 12
    hidden_dim = 128

    for cache_size in [512, 1024, 2048, 4096]:
        k1 = mx.random.normal((1, cache_size, hidden_dim))
        v1 = mx.random.normal((1, cache_size, hidden_dim))
        cache1 = [(k1, v1) for _ in range(num_layers)]

        k2 = mx.random.normal((1, cache_size, hidden_dim))
        v2 = mx.random.normal((1, cache_size, hidden_dim))
        cache2 = [(k2, v2) for _ in range(num_layers)]

        start = time.time()
        merged = ChunkedPrefillEngine._concatenate_caches(cache1, cache2)
        mx.eval(merged[0][0] if merged else mx.array([]))
        elapsed = time.time() - start

        logger.info(
            f"Cache merge (cache_size={cache_size}): {elapsed*1000:.2f}ms "
            f"({num_layers} layers)"
        )


if __name__ == "__main__":
    logger.info("Chunked Prefill MVP Benchmark")
    logger.info("=" * 80)

    # Run benchmarks
    benchmark_traditional_vs_chunked(prompt_length=2048)
    benchmark_chunk_sizes(prompt_length=2048)
    benchmark_prompt_lengths()
    benchmark_cache_merging()

    logger.info("\n" + "=" * 80)
    logger.info("Benchmark complete!")
    logger.info("=" * 80)
