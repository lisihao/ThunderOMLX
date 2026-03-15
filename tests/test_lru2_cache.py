"""
Tests for LRU-2 Block-Level Cache implementation.

P2: LRU-2 algorithm maintains two queues (COLD/HOT) to better identify truly hot data
and resist "scan pollution" where sequential one-time accesses evict hot data.
"""

import pytest
import mlx.core as mx
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def lru2_cache(temp_cache_dir):
    """Create PagedSSDCacheManager with small hot cache for testing LRU-2."""
    cache = PagedSSDCacheManager(
        cache_dir=Path(temp_cache_dir),
        max_size_bytes=100 * 1024 * 1024,  # 100MB SSD cache
        hot_cache_max_bytes=1024 * 1024,    # 1MB hot cache
        enable_compression=False,
        enable_checksum=False,
        enable_prefetch=False  # Disable prefetch for isolated testing
    )
    yield cache
    cache.shutdown()


def create_test_block(block_id: int, size_kb: int = 100):
    """Create a test KV cache block."""
    # Create test tensors (~size_kb total)
    num_elements = (size_kb * 1024) // 4  # 4 bytes per float32
    k = mx.random.normal((num_elements,))
    v = mx.random.normal((num_elements,))

    kv_data = {
        f"cache.{block_id}.k": k,
        f"cache.{block_id}.v": v
    }

    return kv_data


class TestLRU2BasicFlow:
    """Test basic LRU-2 flow: first access → COLD, second access → HOT."""

    def test_first_access_goes_to_cold(self, lru2_cache):
        """First access should add block to COLD queue."""
        block_data = create_test_block(1, size_kb=100)
        block_hash = b"block_hash_1"

        # Save block (will go to hot cache)
        lru2_cache.save_block(
            block_hash=block_hash,
            kv_data=block_data,
            metadata={"prompt": "test"},
            file_path=Path("/tmp/test.safetensors")
        )

        # Check that block is in COLD queue (not HOT yet)
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_hits"] == 0, "Should not have COLD hits yet"
        assert stats["hot_cache_hot_hits"] == 0, "Should not have HOT hits yet"
        assert stats["hot_cache_promotions"] == 0, "Should not have promotions yet"

        # Verify internal state
        assert block_hash in lru2_cache._hot_cache_cold
        assert block_hash not in lru2_cache._hot_cache_hot

    def test_second_access_promotes_to_hot(self, lru2_cache):
        """Second access should promote block from COLD to HOT."""
        block_data = create_test_block(1, size_kb=100)
        block_hash = b"block_hash_1"

        # First access → COLD
        lru2_cache.save_block(
            block_hash=block_hash,
            kv_data=block_data,
            metadata={"prompt": "test"},
            file_path=Path("/tmp/test.safetensors")
        )

        # Second access → should promote to HOT
        lru2_cache.save_block(
            block_hash=block_hash,
            kv_data=block_data,
            metadata={"prompt": "test"},
            file_path=Path("/tmp/test.safetensors")
        )

        # Check stats
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_hits"] == 1, "Should have 1 COLD hit"
        assert stats["hot_cache_promotions"] == 1, "Should have 1 promotion"

        # Verify internal state
        assert block_hash not in lru2_cache._hot_cache_cold, "Should no longer be in COLD"
        assert block_hash in lru2_cache._hot_cache_hot, "Should be in HOT"

    def test_third_access_stays_in_hot(self, lru2_cache):
        """Third+ access should keep block in HOT and update LRU order."""
        block_data = create_test_block(1, size_kb=100)
        block_hash = b"block_hash_1"

        # First access → COLD
        lru2_cache.save_block(block_hash, block_data, {"prompt": "test"}, Path("/tmp/test.safetensors"))

        # Second access → promote to HOT
        lru2_cache.save_block(block_hash, block_data, {"prompt": "test"}, Path("/tmp/test.safetensors"))

        # Third access → stay in HOT, move to end
        lru2_cache.save_block(block_hash, block_data, {"prompt": "test"}, Path("/tmp/test.safetensors"))

        # Check stats
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_hot_hits"] == 1, "Should have 1 HOT hit"
        assert stats["hot_cache_promotions"] == 1, "Should still have 1 promotion"

        # Verify still in HOT
        assert block_hash in lru2_cache._hot_cache_hot


class TestLRU2EvictionPriority:
    """Test that COLD queue has eviction priority over HOT queue."""

    def test_evict_from_cold_first(self, lru2_cache):
        """When evicting, should prioritize COLD queue over HOT."""
        # Create blocks that fill the cache
        blocks_cold = []
        for i in range(5):
            block_data = create_test_block(i, size_kb=100)
            block_hash = f"cold_{i}".encode()
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"cold{i}"}, Path(f"/tmp/cold{i}.safetensors"))
            blocks_cold.append(block_hash)

        # Create a hot block (accessed twice)
        hot_block_data = create_test_block(100, size_kb=100)
        hot_block_hash = b"hot_block"
        lru2_cache.save_block(hot_block_hash, hot_block_data, {"prompt": "hot"}, Path("/tmp/hot.safetensors"))
        lru2_cache.save_block(hot_block_hash, hot_block_data, {"prompt": "hot"}, Path("/tmp/hot.safetensors"))

        # Verify hot block is in HOT queue
        assert hot_block_hash in lru2_cache._hot_cache_hot

        # Add more blocks to trigger eviction
        for i in range(10, 20):
            block_data = create_test_block(i, size_kb=100)
            block_hash = f"new_{i}".encode()
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"new{i}"}, Path(f"/tmp/new{i}.safetensors"))

        # Check eviction stats
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_evictions"] > 0, "Should have evicted from COLD"

        # Hot block should still be in HOT queue (not evicted)
        assert hot_block_hash in lru2_cache._hot_cache_hot, "Hot block should not be evicted"

    def test_evict_from_hot_when_cold_empty(self, lru2_cache):
        """When COLD is empty, should evict from HOT."""
        # Create only hot blocks (all accessed twice)
        for i in range(10):
            block_data = create_test_block(i, size_kb=100)
            block_hash = f"hot_{i}".encode()
            # First access
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"hot{i}"}, Path(f"/tmp/hot{i}.safetensors"))
            # Second access → promote to HOT
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"hot{i}"}, Path(f"/tmp/hot{i}.safetensors"))

        # Verify COLD is empty
        assert len(lru2_cache._hot_cache_cold) == 0, "COLD queue should be empty"

        # Add more blocks to trigger eviction from HOT
        for i in range(20, 30):
            block_data = create_test_block(i, size_kb=100)
            block_hash = f"new_hot_{i}".encode()
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"new{i}"}, Path(f"/tmp/new{i}.safetensors"))
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"new{i}"}, Path(f"/tmp/new{i}.safetensors"))

        # Check eviction stats
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_hot_evictions"] > 0, "Should have evicted from HOT when COLD empty"


class TestLRU2ScanPollutionResistance:
    """Test that LRU-2 resists scan pollution from sequential one-time accesses."""

    def test_scan_does_not_evict_hot_data(self, lru2_cache):
        """Sequential scan of one-time blocks should not evict truly hot data."""
        # Create a truly hot block (accessed many times)
        hot_block_data = create_test_block(999, size_kb=100)
        hot_block_hash = b"truly_hot"

        # Access 5 times to ensure it's in HOT queue
        for _ in range(5):
            lru2_cache.save_block(hot_block_hash, hot_block_data, {"prompt": "hot"}, Path("/tmp/hot.safetensors"))

        # Verify in HOT queue
        assert hot_block_hash in lru2_cache._hot_cache_hot

        # Sequential scan: many one-time accesses
        for i in range(100):
            scan_block_data = create_test_block(i, size_kb=50)
            scan_block_hash = f"scan_{i}".encode()
            lru2_cache.save_block(scan_block_hash, scan_block_data, {"prompt": f"scan{i}"}, Path(f"/tmp/scan{i}.safetensors"))

        # Hot block should STILL be in HOT queue
        assert hot_block_hash in lru2_cache._hot_cache_hot, "Truly hot block should not be evicted by scan"

        # Stats should show COLD evictions (scan blocks evicted)
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_evictions"] > 0, "Scan blocks should be evicted from COLD"


class TestLRU2MemoryAccounting:
    """Test that memory accounting is accurate for COLD + HOT."""

    def test_total_bytes_accurate(self, lru2_cache):
        """Total bytes should equal COLD bytes + HOT bytes."""
        # Add some blocks
        for i in range(5):
            block_data = create_test_block(i, size_kb=100)
            block_hash = f"block_{i}".encode()
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"test{i}"}, Path(f"/tmp/test{i}.safetensors"))
            if i % 2 == 0:
                # Promote even blocks to HOT
                lru2_cache.save_block(block_hash, block_data, {"prompt": f"test{i}"}, Path(f"/tmp/test{i}.safetensors"))

        # Verify total = cold + hot
        total = lru2_cache._hot_cache_total_bytes
        cold = lru2_cache._hot_cache_cold_bytes
        hot = lru2_cache._hot_cache_hot_bytes

        assert total == cold + hot, f"Total {total} != cold {cold} + hot {hot}"
        assert total <= lru2_cache._hot_cache_max_bytes, "Total should not exceed max"


class TestLRU2Statistics:
    """Test that LRU-2 statistics are accurate."""

    def test_promotion_count(self, lru2_cache):
        """Promotions should count COLD → HOT transitions."""
        blocks = []
        for i in range(5):
            block_data = create_test_block(i, size_kb=100)
            block_hash = f"block_{i}".encode()
            # First access
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"test{i}"}, Path(f"/tmp/test{i}.safetensors"))
            # Second access → promote
            lru2_cache.save_block(block_hash, block_data, {"prompt": f"test{i}"}, Path(f"/tmp/test{i}.safetensors"))
            blocks.append(block_hash)

        stats = lru2_cache.get_stats()
        assert stats["hot_cache_promotions"] == 5, "Should have 5 promotions"
        assert stats["hot_cache_cold_hits"] == 5, "Should have 5 COLD hits"

    def test_cold_hits_vs_hot_hits(self, lru2_cache):
        """COLD hits and HOT hits should be counted separately."""
        block_data = create_test_block(1, size_kb=100)
        block_hash = b"test_block"

        # First access → add to COLD (no hits)
        lru2_cache.save_block(block_hash, block_data, {"prompt": "test"}, Path("/tmp/test.safetensors"))
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_hits"] == 0
        assert stats["hot_cache_hot_hits"] == 0

        # Second access → COLD hit + promote
        lru2_cache.save_block(block_hash, block_data, {"prompt": "test"}, Path("/tmp/test.safetensors"))
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_hits"] == 1
        assert stats["hot_cache_hot_hits"] == 0

        # Third access → HOT hit
        lru2_cache.save_block(block_hash, block_data, {"prompt": "test"}, Path("/tmp/test.safetensors"))
        stats = lru2_cache.get_stats()
        assert stats["hot_cache_cold_hits"] == 1
        assert stats["hot_cache_hot_hits"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
