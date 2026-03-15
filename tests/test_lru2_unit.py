"""
Unit tests for LRU-2 hot cache implementation.

Tests the internal _hot_cache_* methods directly to verify LRU-2 logic.
"""

import pytest
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from pathlib import Path
import tempfile
import shutil


def create_fake_entry(block_id: int, size_bytes: int = 1000):
    """Create a fake cache entry for testing."""
    # Simulate raw tensor bytes
    fake_tensor_data = b"x" * size_bytes
    return {
        'tensors_raw': {
            f"cache.{block_id}.k": (fake_tensor_data, None, None),
            f"cache.{block_id}.v": (fake_tensor_data, None, None)
        },
        'file_metadata': {},
        'block_metadata': None
    }


@pytest.fixture
def cache_manager():
    """Create a minimal PagedSSDCacheManager for testing."""
    temp_dir = tempfile.mkdtemp()
    manager = PagedSSDCacheManager(
        cache_dir=Path(temp_dir),
        max_size_bytes=10 * 1024 * 1024,    # 10MB SSD
        hot_cache_max_bytes=5000,            # 5KB hot cache (very small for testing)
        enable_compression=False,
        enable_checksum=False,
        enable_prefetch=False
    )
    yield manager
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestLRU2BasicLogic:
    """Test basic LRU-2 queue logic."""

    def test_first_put_goes_to_cold(self, cache_manager):
        """First _hot_cache_put should add entry to COLD queue."""
        entry = create_fake_entry(1, size_bytes=1000)
        block_hash = b"block_1"

        cache_manager._hot_cache_put(block_hash, entry)

        # Should be in COLD, not in HOT
        assert block_hash in cache_manager._hot_cache_cold
        assert block_hash not in cache_manager._hot_cache_hot

    def test_second_put_promotes_to_hot(self, cache_manager):
        """Second _hot_cache_put should promote from COLD to HOT."""
        entry = create_fake_entry(1, size_bytes=1000)
        block_hash = b"block_1"

        # First put → COLD
        cache_manager._hot_cache_put(block_hash, entry)
        assert block_hash in cache_manager._hot_cache_cold

        # Second put → promote to HOT
        cache_manager._hot_cache_put(block_hash, entry)
        assert block_hash not in cache_manager._hot_cache_cold
        assert block_hash in cache_manager._hot_cache_hot

        # Check stats
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_cold_hits"] == 1
        assert stats["hot_cache_promotions"] == 1

    def test_third_put_stays_in_hot(self, cache_manager):
        """Third+ _hot_cache_put should keep entry in HOT."""
        entry = create_fake_entry(1, size_bytes=1000)
        block_hash = b"block_1"

        # First → COLD
        cache_manager._hot_cache_put(block_hash, entry)
        # Second → promote to HOT
        cache_manager._hot_cache_put(block_hash, entry)
        # Third → stay in HOT
        cache_manager._hot_cache_put(block_hash, entry)

        assert block_hash in cache_manager._hot_cache_hot
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_hot_hits"] == 1

    def test_get_from_cold_promotes(self, cache_manager):
        """_hot_cache_get from COLD should promote to HOT."""
        entry = create_fake_entry(1, size_bytes=1000)
        block_hash = b"block_1"

        # Put in COLD
        cache_manager._hot_cache_put(block_hash, entry)
        assert block_hash in cache_manager._hot_cache_cold

        # Get from COLD → should promote
        result = cache_manager._hot_cache_get(block_hash)
        assert result is not None
        assert block_hash not in cache_manager._hot_cache_cold
        assert block_hash in cache_manager._hot_cache_hot

        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_promotions"] == 1

    def test_get_from_hot_moves_to_end(self, cache_manager):
        """_hot_cache_get from HOT should move to end of HOT queue."""
        entry = create_fake_entry(1, size_bytes=1000)
        block_hash = b"block_1"

        # Put twice → HOT
        cache_manager._hot_cache_put(block_hash, entry)
        cache_manager._hot_cache_put(block_hash, entry)

        # Get from HOT
        result = cache_manager._hot_cache_get(block_hash)
        assert result is not None
        assert block_hash in cache_manager._hot_cache_hot

        # Should be at the end (most recently used)
        last_key = list(cache_manager._hot_cache_hot.keys())[-1]
        assert last_key == block_hash


class TestLRU2EvictionPriority:
    """Test that eviction prioritizes COLD over HOT."""

    def test_evict_from_cold_first(self, cache_manager):
        """When cache is full, should evict from COLD before HOT."""
        # Add blocks to fill cache (5KB limit)
        # Each block is 1KB, so 5 blocks = 5KB (at limit)

        # Add 3 blocks to COLD
        for i in range(3):
            entry = create_fake_entry(i, size_bytes=1000)
            cache_manager._hot_cache_put(f"cold_{i}".encode(), entry)

        # Add 1 block to HOT (accessed twice)
        hot_entry = create_fake_entry(100, size_bytes=1000)
        hot_hash = b"hot_block"
        cache_manager._hot_cache_put(hot_hash, hot_entry)
        cache_manager._hot_cache_put(hot_hash, hot_entry)

        # Verify hot is in HOT
        assert hot_hash in cache_manager._hot_cache_hot

        # Add more blocks to trigger eviction
        for i in range(10, 15):
            entry = create_fake_entry(i, size_bytes=1000)
            cache_manager._hot_cache_put(f"new_{i}".encode(), entry)

        # Check that we evicted from COLD
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_cold_evictions"] > 0

        # Hot block should still be there
        assert hot_hash in cache_manager._hot_cache_hot

    def test_evict_from_hot_when_cold_empty(self, cache_manager):
        """When COLD is empty and a HOT block grows, should evict from HOT."""
        # Scenario: HOT queue full, COLD empty, then update a HOT block with larger data
        # This will trigger eviction from HOT (since COLD is empty)

        # Step 1: Fill HOT queue (2 blocks × 2000 bytes = 4000 bytes, within 5000 limit)
        for i in range(2):
            entry = create_fake_entry(i, size_bytes=1000)  # 2000 bytes each
            block_hash = f"hot_{i}".encode()
            cache_manager._hot_cache_put(block_hash, entry)  # → COLD
            cache_manager._hot_cache_put(block_hash, entry)  # → HOT

        assert len(cache_manager._hot_cache_hot) == 2
        assert len(cache_manager._hot_cache_cold) == 0

        # Step 2: Update hot_0 with MUCH larger data
        # This will make total > 5000, triggering eviction
        large_entry = create_fake_entry(0, size_bytes=2000)  # 4000 bytes (was 2000)
        cache_manager._hot_cache_put(b"hot_0", large_entry)  # Already in HOT, update in-place

        # Now total = 4000 (hot_0) + 2000 (hot_1) = 6000 > 5000
        # Should evict hot_1 from HOT

        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_hot_evictions"] > 0, "Should have evicted from HOT"
        assert len(cache_manager._hot_cache_hot) == 1, "Should have 1 block left in HOT"


class TestLRU2ScanResistance:
    """Test that LRU-2 resists scan pollution."""

    def test_scan_does_not_evict_hot(self, cache_manager):
        """Sequential one-time accesses should not evict truly hot data."""
        # Create a truly hot block
        hot_entry = create_fake_entry(999, size_bytes=500)
        hot_hash = b"truly_hot"

        # Access 5 times to ensure in HOT
        for _ in range(5):
            cache_manager._hot_cache_put(hot_hash, hot_entry)

        assert hot_hash in cache_manager._hot_cache_hot

        # Sequential scan: 50 one-time accesses (each 50 bytes)
        for i in range(50):
            scan_entry = create_fake_entry(i, size_bytes=50)
            cache_manager._hot_cache_put(f"scan_{i}".encode(), scan_entry)

        # Hot block should STILL be in HOT
        assert hot_hash in cache_manager._hot_cache_hot

        # Stats should show COLD evictions
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_cold_evictions"] > 0


class TestLRU2MemoryAccounting:
    """Test that memory accounting is correct."""

    def test_total_equals_cold_plus_hot(self, cache_manager):
        """Total bytes should always equal COLD + HOT."""
        # Add some blocks
        for i in range(3):
            entry = create_fake_entry(i, size_bytes=800)
            cache_manager._hot_cache_put(f"block_{i}".encode(), entry)
            if i % 2 == 0:
                # Promote even blocks
                cache_manager._hot_cache_put(f"block_{i}".encode(), entry)

        # Verify total = cold + hot
        total = cache_manager._hot_cache_total_bytes
        cold = cache_manager._hot_cache_cold_bytes
        hot = cache_manager._hot_cache_hot_bytes

        assert total == cold + hot
        assert total <= cache_manager._hot_cache_max_bytes

    def test_no_overflow(self, cache_manager):
        """Total bytes should never exceed max."""
        # Add many blocks
        for i in range(20):
            entry = create_fake_entry(i, size_bytes=500)
            cache_manager._hot_cache_put(f"block_{i}".encode(), entry)

        # Should not exceed max
        assert cache_manager._hot_cache_total_bytes <= cache_manager._hot_cache_max_bytes


class TestLRU2Statistics:
    """Test LRU-2 statistics accuracy."""

    def test_promotion_count(self, cache_manager):
        """Promotions should count COLD → HOT transitions."""
        for i in range(3):
            entry = create_fake_entry(i, size_bytes=500)
            block_hash = f"block_{i}".encode()
            cache_manager._hot_cache_put(block_hash, entry)  # → COLD
            cache_manager._hot_cache_put(block_hash, entry)  # → HOT (promotion)

        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_promotions"] == 3

    def test_cold_vs_hot_hits(self, cache_manager):
        """COLD hits and HOT hits should be counted separately."""
        entry = create_fake_entry(1, size_bytes=500)
        block_hash = b"test"

        # First put → COLD (no hits)
        cache_manager._hot_cache_put(block_hash, entry)
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_cold_hits"] == 0
        assert stats["hot_cache_hot_hits"] == 0

        # Second put → COLD hit + promotion
        cache_manager._hot_cache_put(block_hash, entry)
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_cold_hits"] == 1
        assert stats["hot_cache_hot_hits"] == 0

        # Third put → HOT hit
        cache_manager._hot_cache_put(block_hash, entry)
        stats = cache_manager.get_stats_dict()
        assert stats["hot_cache_cold_hits"] == 1
        assert stats["hot_cache_hot_hits"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
