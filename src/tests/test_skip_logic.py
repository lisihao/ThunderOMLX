# SPDX-License-Identifier: Apache-2.0
"""
Tests for Approximate Skip Logic in BlockAwarePrefixCache.

Tests the match_cache_with_skip_logic() method which implements:
- Full Skip (100% cache hit)
- Approximate Skip (90%+ cache hit with configurable threshold)
- No Skip (< threshold)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_cache import BlockTable
from omlx.cache.prefix_cache import BlockAwarePrefixCache


class TestSkipLogic:
    """Tests for match_cache_with_skip_logic() Approximate Skip feature."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock BlockAwarePrefixCache for testing."""
        cache = MagicMock(spec=BlockAwarePrefixCache)
        cache.match_cache_with_skip_logic = BlockAwarePrefixCache.match_cache_with_skip_logic.__get__(cache)
        return cache

    def test_full_skip_100_percent_hit(self, mock_cache):
        """Test Full Skip: 100% cache hit, no remaining tokens."""
        # Mock fetch_cache to return full match
        mock_block_table = BlockTable(request_id="test-001", block_ids=[1, 2, 3], num_tokens=100)
        mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, []))

        # Call with 100 tokens
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            approx_threshold=0.95
        )

        # Assertions
        assert result['can_skip_prefill'] is True
        assert result['skip_reason'] == 'full'
        assert result['cache_hit_ratio'] == 1.0
        assert result['approx_zero_fill_count'] == 0
        assert result['block_table'] == mock_block_table
        assert result['remaining_tokens'] == []

    def test_approximate_skip_95_percent_hit(self, mock_cache):
        """Test Approximate Skip: 95% cache hit (default threshold)."""
        # Mock fetch_cache to return 95 tokens cached, 5 remaining
        mock_block_table = BlockTable(request_id="test-002", block_ids=[1, 2], num_tokens=95)
        remaining_tokens = list(range(95, 100))  # 5 tokens
        mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, remaining_tokens))

        # Call with 100 tokens
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            approx_threshold=0.95  # Default 95%
        )

        # Assertions
        assert result['can_skip_prefill'] is True
        assert result['skip_reason'] == 'approximate'
        assert result['cache_hit_ratio'] == 0.95
        assert result['approx_zero_fill_count'] == 5
        assert result['block_table'] == mock_block_table
        assert len(result['remaining_tokens']) == 5

    def test_approximate_skip_90_percent_hit_custom_threshold(self, mock_cache):
        """Test Approximate Skip: 90% cache hit with custom 0.90 threshold."""
        # Mock fetch_cache to return 90 tokens cached, 10 remaining
        mock_block_table = BlockTable(request_id="test-003", block_ids=[1], num_tokens=90)
        remaining_tokens = list(range(90, 100))  # 10 tokens
        mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, remaining_tokens))

        # Call with 100 tokens, threshold=0.90
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            approx_threshold=0.90  # Custom threshold
        )

        # Assertions
        assert result['can_skip_prefill'] is True
        assert result['skip_reason'] == 'approximate'
        assert result['cache_hit_ratio'] == 0.90
        assert result['approx_zero_fill_count'] == 10
        assert result['block_table'] == mock_block_table
        assert len(result['remaining_tokens']) == 10

    def test_no_skip_below_threshold(self, mock_cache):
        """Test No Skip: 89% cache hit < 90% threshold."""
        # Mock fetch_cache to return 89 tokens cached, 11 remaining
        mock_block_table = BlockTable(request_id="test-004", block_ids=[1], num_tokens=89)
        remaining_tokens = list(range(89, 100))  # 11 tokens
        mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, remaining_tokens))

        # Call with 100 tokens, threshold=0.90
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            approx_threshold=0.90
        )

        # Assertions
        assert result['can_skip_prefill'] is False
        assert result['skip_reason'] == 'none'
        assert result['cache_hit_ratio'] == 0.89
        assert result['approx_zero_fill_count'] == 0  # No skip, no zero fill
        assert result['block_table'] == mock_block_table
        assert len(result['remaining_tokens']) == 11

    def test_no_skip_zero_cache_hit(self, mock_cache):
        """Test No Skip: 0% cache hit (cold start)."""
        # Mock fetch_cache to return no blocks, all tokens remaining
        mock_cache.fetch_cache = MagicMock(return_value=(None, list(range(100))))

        # Call with 100 tokens
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            approx_threshold=0.95
        )

        # Assertions
        assert result['can_skip_prefill'] is False
        assert result['skip_reason'] == 'none'
        assert result['cache_hit_ratio'] == 0.0
        assert result['approx_zero_fill_count'] == 0
        assert result['block_table'] is None
        assert len(result['remaining_tokens']) == 100

    def test_zero_fill_count_accuracy(self, mock_cache):
        """Test zero_fill_count matches remaining_tokens length."""
        test_cases = [
            (100, 90, 10, 0.85),   # 90% hit, use 85% threshold
            (200, 192, 8, 0.95),   # 96% hit, use 95% threshold
            (256, 240, 16, 0.90),  # 93.75% hit, use 90% threshold
        ]

        for total, cached, expected_zero_fill, threshold in test_cases:
            mock_block_table = BlockTable(request_id=f"test-{total}", num_tokens=cached)
            mock_block_table.block_ids = [1, 2]  # Add block_ids for logging
            remaining = list(range(cached, total))
            mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, remaining))

            hit_ratio = cached / total
            result = mock_cache.match_cache_with_skip_logic(
                tokens=list(range(total)),
                approx_threshold=threshold
            )

            # Verify hit_ratio >= threshold AND < 1.0 AND has remaining tokens
            assert hit_ratio >= threshold and hit_ratio < 1.0 and len(remaining) > 0, \
                f"Test case setup error: hit={hit_ratio:.2%}, threshold={threshold:.2%}"

            assert result['can_skip_prefill'] is True, \
                f"Failed for {total} total, {cached} cached, hit={hit_ratio:.2%}, threshold={threshold:.2%}"
            assert result['skip_reason'] == 'approximate'
            assert result['approx_zero_fill_count'] == expected_zero_fill
            assert len(result['remaining_tokens']) == expected_zero_fill

    def test_boundary_condition_exact_threshold(self, mock_cache):
        """Test boundary: cache_hit_ratio exactly equals threshold."""
        # Mock 95 tokens cached, 5 remaining (95% hit rate)
        mock_block_table = BlockTable(request_id="test-boundary", num_tokens=95)
        mock_block_table.block_ids = [1, 2, 3]  # Add block_ids for logging
        remaining_tokens = list(range(95, 100))
        mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, remaining_tokens))

        # Test with threshold = 0.95 (exact match)
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            approx_threshold=0.95
        )

        # At exact threshold (>= 0.95), should trigger approximate skip
        # (condition: >= threshold AND < 1.0 AND remaining > 0)
        assert result['can_skip_prefill'] is True
        assert result['skip_reason'] == 'approximate'
        assert result['cache_hit_ratio'] == 0.95
        assert result['approx_zero_fill_count'] == 5

    def test_extra_keys_isolation(self, mock_cache):
        """Test that extra_keys are passed to fetch_cache for cache isolation."""
        mock_block_table = BlockTable(request_id="test-keys", num_tokens=50)
        mock_block_table.block_ids = [1]  # Add block_ids
        remaining = list(range(50, 100))
        mock_cache.fetch_cache = MagicMock(return_value=(mock_block_table, remaining))

        extra_keys = ("image_hash_123", "config_v2")
        result = mock_cache.match_cache_with_skip_logic(
            tokens=list(range(100)),
            extra_keys=extra_keys,
            approx_threshold=0.50  # 50% threshold to match 50% hit rate
        )

        # Verify fetch_cache was called with extra_keys
        mock_cache.fetch_cache.assert_called_once_with(
            "_skip_check",
            list(range(100)),
            extra_keys
        )
        # 50% hit rate >= 50% threshold, should trigger approximate skip
        assert result['can_skip_prefill'] is True
        assert result['skip_reason'] == 'approximate'
        assert result['cache_hit_ratio'] == 0.50
        assert result['approx_zero_fill_count'] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
