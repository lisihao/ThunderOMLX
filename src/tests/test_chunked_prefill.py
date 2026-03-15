# SPDX-License-Identifier: Apache-2.0
"""
Tests for Chunked Prefill MVP.

This test module verifies:
1. Configuration loading from environment variables
2. Chunking decision logic (should_use_chunking)
3. Chunked prefill execution with mock prefill function
4. Cache merging behavior
5. Fallback to traditional prefill
"""

import os
import pytest
import mlx.core as mx
from unittest.mock import Mock, patch, MagicMock

from omlx.chunked_prefill import ChunkedPrefillConfig, ChunkedPrefillEngine


class TestChunkedPrefillConfig:
    """Test ChunkedPrefillConfig initialization and env loading."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkedPrefillConfig()
        assert config.chunk_size == 512
        assert config.enable_chunking is False
        assert config.min_tokens_for_chunking == 1024

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkedPrefillConfig(
            chunk_size=256, enable_chunking=True, min_tokens_for_chunking=512
        )
        assert config.chunk_size == 256
        assert config.enable_chunking is True
        assert config.min_tokens_for_chunking == 512

    def test_env_loading_enabled(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "OMLX_ENABLE_CHUNKED_PREFILL": "true",
                "OMLX_CHUNK_SIZE": "256",
                "OMLX_MIN_TOKENS_FOR_CHUNKING": "2048",
            },
        ):
            config = ChunkedPrefillConfig.from_env()
            assert config.enable_chunking is True
            assert config.chunk_size == 256
            assert config.min_tokens_for_chunking == 2048

    def test_env_loading_disabled(self):
        """Test that chunking is disabled by default via env."""
        with patch.dict(
            os.environ,
            {
                "OMLX_ENABLE_CHUNKED_PREFILL": "false",
            },
            clear=True,
        ):
            # Reload env vars to clear any set values
            config = ChunkedPrefillConfig.from_env()
            assert config.enable_chunking is False

    def test_env_loading_invalid_values(self):
        """Test handling of invalid environment values."""
        with patch.dict(
            os.environ,
            {
                "OMLX_CHUNK_SIZE": "invalid",  # Will raise ValueError
            },
        ):
            with pytest.raises(ValueError):
                ChunkedPrefillConfig.from_env()

    def test_config_validation(self):
        """Test that invalid config values are corrected."""
        # Negative chunk size should be clamped to 1
        config = ChunkedPrefillConfig(chunk_size=-1)
        assert config.chunk_size == 1

        # Zero should also be clamped
        config = ChunkedPrefillConfig(min_tokens_for_chunking=0)
        assert config.min_tokens_for_chunking == 1


class TestChunkedPrefillDecision:
    """Test the decision logic for when to use chunking."""

    def test_disabled_no_chunking(self):
        """When chunking is disabled, should never chunk."""
        config = ChunkedPrefillConfig(enable_chunking=False)
        engine = ChunkedPrefillEngine(None, config)

        # Even very long tokens should not be chunked
        long_tokens = mx.array([1] * 10000)
        assert engine.should_use_chunking(long_tokens) is False

    def test_short_tokens_no_chunking(self):
        """Short prompts should not be chunked."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        short_tokens = mx.array([1] * 512)
        assert engine.should_use_chunking(short_tokens) is False

    def test_long_tokens_with_chunking(self):
        """Long prompts should trigger chunking when enabled."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        long_tokens = mx.array([1] * 2048)
        assert engine.should_use_chunking(long_tokens) is True

    def test_exact_threshold(self):
        """Tokens exactly at threshold should trigger chunking."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        threshold_tokens = mx.array([1] * 1024)
        assert engine.should_use_chunking(threshold_tokens) is True

        # Just below threshold
        below_threshold = mx.array([1] * 1023)
        assert engine.should_use_chunking(below_threshold) is False

    def test_2d_token_handling(self):
        """Test handling of 2D token arrays (batch dimension)."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        # 2D array: shape [batch_size, seq_len]
        tokens_2d = mx.array([[1] * 2048])
        assert engine.should_use_chunking(tokens_2d) is True

        tokens_2d_short = mx.array([[1] * 512])
        assert engine.should_use_chunking(tokens_2d_short) is False


class TestChunkedPrefillExecution:
    """Test chunked prefill execution with mock prefill function."""

    def test_missing_prefill_fn(self):
        """Should raise ValueError if prefill_fn is not provided."""
        config = ChunkedPrefillConfig(enable_chunking=True)
        engine = ChunkedPrefillEngine(None, config)

        tokens = mx.array([1] * 2048)
        with pytest.raises(ValueError, match="prefill_fn is required"):
            engine.prefill(None, tokens, None, prefill_fn=None)

    def test_fallback_short_prompt(self):
        """Short prompts should use traditional prefill."""
        config = ChunkedPrefillConfig(enable_chunking=True)
        engine = ChunkedPrefillEngine(None, config)

        mock_model = Mock()
        tokens = mx.array([1] * 512)  # Below threshold
        expected_logits = mx.array([[0.1, 0.2]])
        expected_cache = [mx.array([1, 2, 3])]

        prefill_fn = Mock(return_value=(expected_logits, expected_cache))

        logits, cache = engine.prefill(mock_model, tokens, None, prefill_fn)

        # Should call prefill_fn exactly once
        assert prefill_fn.call_count == 1
        assert mx.array_equal(logits, expected_logits)
        assert mx.array_equal(cache[0], expected_cache[0])

    def test_chunked_execution(self):
        """Test actual chunked execution."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, chunk_size=512, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        mock_model = Mock()
        tokens = mx.array(list(range(1536)))  # 1536 tokens = 3 chunks of 512

        # Mock prefill function that returns dummy logits and cache
        def mock_prefill(model, chunk_tokens, cache):
            # Return dummy logits and a synthetic cache
            batch_size = chunk_tokens.shape[0] if chunk_tokens.ndim > 1 else 1
            seq_len = chunk_tokens.shape[-1]
            logits = mx.random.normal((batch_size, seq_len, 128))
            new_cache = [mx.random.normal((2, seq_len, 64))]
            return logits, new_cache

        logits, cache = engine.prefill(mock_model, tokens, None, mock_prefill)

        # Verify results are returned
        assert logits is not None
        assert cache is not None
        assert len(cache) > 0

    def test_chunking_with_existing_cache(self):
        """Test chunked prefill with pre-existing cache."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, chunk_size=512, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        mock_model = Mock()
        tokens = mx.array(list(range(1024)))

        # Pre-existing cache from previous request
        existing_cache = [mx.array([[1, 2, 3, 4]])]

        call_count = [0]

        def mock_prefill(model, chunk_tokens, cache):
            call_count[0] += 1
            logits = mx.random.normal((1, chunk_tokens.shape[-1], 128))
            new_cache = [mx.random.normal((1, chunk_tokens.shape[-1], 64))]
            return logits, new_cache

        logits, cache = engine.prefill(
            mock_model, tokens, existing_cache, mock_prefill
        )

        # Should have called prefill multiple times (one per chunk)
        assert call_count[0] > 1
        assert cache is not None

    def test_fallback_on_prefill_error(self):
        """Test fallback to traditional prefill on error."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, chunk_size=512, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        mock_model = Mock()
        tokens = mx.array(list(range(2048)))

        fallback_logits = mx.array([[0.5, 0.5]])
        fallback_cache = [mx.array([9, 9, 9])]

        call_count = [0]

        def mock_prefill(model, chunk_tokens, cache):
            call_count[0] += 1
            # Fail on first chunk
            if call_count[0] == 1:
                raise RuntimeError("Simulated prefill error")
            # Return fallback result
            return fallback_logits, fallback_cache

        logits, cache = engine.prefill(mock_model, tokens, None, mock_prefill)

        # Should call prefill twice: once for chunk (error), once for fallback
        assert call_count[0] == 2
        assert mx.array_equal(logits, fallback_logits)


class TestCacheMerging:
    """Test cache concatenation utilities."""

    def test_concatenate_caches_tuple_format(self):
        """Test concatenating caches in tuple (key, value) format."""
        # Create mock KV cache tuples
        # Shape: (1, seq_len, hidden_dim) representing (batch, seq_len, hidden)
        k1 = mx.array(1.0 * mx.ones((1, 4, 64)))  # 4 tokens
        v1 = mx.array(2.0 * mx.ones((1, 4, 64)))
        cache1 = [(k1, v1)]

        k2 = mx.array(3.0 * mx.ones((1, 4, 64)))  # 4 more tokens
        v2 = mx.array(4.0 * mx.ones((1, 4, 64)))
        cache2 = [(k2, v2)]

        result = ChunkedPrefillEngine._concatenate_caches(cache1, cache2)

        # Should have concatenated along seq_len (axis -2)
        assert len(result) == 1
        merged_k, merged_v = result[0]
        assert merged_k.shape[1] == 8  # 4 + 4 tokens
        assert merged_v.shape[1] == 8

    def test_concatenate_caches_empty(self):
        """Test concatenating with empty caches."""
        cache1 = []
        cache2 = [mx.ones((1, 4, 64))]

        result = ChunkedPrefillEngine._concatenate_caches(cache1, cache2)
        assert len(result) == 1

    def test_concatenate_caches_length_mismatch(self):
        """Test that mismatched cache lengths raise error."""
        cache1 = [mx.ones((1, 4, 64))]
        cache2 = [mx.ones((1, 4, 64)), mx.ones((1, 4, 64))]

        with pytest.raises(ValueError, match="Cache length mismatch"):
            ChunkedPrefillEngine._concatenate_caches(cache1, cache2)


class TestIntegrationWithScheduler:
    """Integration tests simulating scheduler usage."""

    def test_engine_with_scheduler_flow(self):
        """Test that chunked prefill engine can be used in scheduler context."""
        config = ChunkedPrefillConfig(
            enable_chunking=True, chunk_size=512, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        # Simulate scheduler calling prefill
        tokens = mx.array([1] * 2048)
        mock_model = Mock()

        def scheduler_prefill(model, toks, cache):
            # Simulate a real prefill operation
            return mx.random.normal((1, toks.shape[-1], 128)), []

        logits, cache = engine.prefill(mock_model, tokens, None, scheduler_prefill)
        assert logits is not None


# Benchmark tests (can be run separately)
class TestChunkedPrefillBenchmark:
    """Benchmark tests for chunked prefill performance.

    These tests measure performance metrics but don't assert specific values
    (too dependent on hardware). They print results for manual inspection.
    """

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_benchmark_memory_usage(self):
        """Benchmark memory usage with chunked vs traditional prefill."""
        import gc
        import psutil

        config = ChunkedPrefillConfig(
            enable_chunking=True, chunk_size=512, min_tokens_for_chunking=1024
        )
        engine = ChunkedPrefillEngine(None, config)

        # This would require a real model and memory profiling
        # Placeholder for manual benchmarking
        print("Benchmark: Memory usage comparison would go here")

    @pytest.mark.skip(reason="Benchmark test - run manually")
    def test_benchmark_first_token_latency(self):
        """Benchmark first-token latency improvement."""
        # This would require:
        # 1. A real MLX model
        # 2. Long prompt (2000+ tokens)
        # 3. Timing instrumentation in prefill_fn
        print("Benchmark: First-token latency would be measured here")


if __name__ == "__main__":
    # Run tests with: pytest src/tests/test_chunked_prefill.py -v
    pytest.main([__file__, "-v"])
