# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Cache for Async I/O optimization.

This module implements a LRU cache for storing pre-decompressed file data,
enabling overlap of I/O operations with GPU computation.

Key features:
- Stores raw decompressed bytes (not tensors) to avoid Metal API on background thread
- LRU eviction policy with configurable size
- Thread-safe operations
- Designed for sequential prefetching based on Scheduler predictions
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class PrefetchCache:
    """
    LRU cache for pre-decompressed file data.

    Stores decompressed safetensors file bytes, ready for mx.load() on main thread.
    Separate from hot_cache to avoid confusion and enable different eviction policies.

    Example:
        >>> cache = PrefetchCache(max_size=5)
        >>> cache.put(block_hash, decompressed_bytes)
        >>> data = cache.get(block_hash)  # Returns bytes or None
    """

    def __init__(self, max_size: int = 5):
        """
        Initialize prefetch cache.

        Args:
            max_size: Maximum number of entries to cache (default 5).
                Each entry is typically ~10MB compressed → ~40MB decompressed.
                Total memory: ~200MB with default settings.
        """
        self._cache: OrderedDict[bytes, bytes] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
            'total_bytes': 0,
        }

        logger.info(f"✅ PrefetchCache initialized: max_size={max_size} entries")

    def get(self, block_hash: bytes) -> Optional[bytes]:
        """
        Get pre-decompressed data if available.

        Args:
            block_hash: Block content hash.

        Returns:
            Decompressed bytes ready for mx.load(), or None if not cached.
        """
        with self._lock:
            if block_hash in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(block_hash)
                self._stats['hits'] += 1

                logger.debug(
                    f"Prefetch HIT: {block_hash.hex()[:16]}... "
                    f"({len(self._cache.get(block_hash))} bytes)"
                )
                return self._cache[block_hash]

            self._stats['misses'] += 1
            return None

    def put(self, block_hash: bytes, decompressed_data: bytes) -> None:
        """
        Store pre-decompressed data.

        Args:
            block_hash: Block content hash.
            decompressed_data: Decompressed safetensors file bytes.
        """
        with self._lock:
            # Remove if already exists (update scenario)
            if block_hash in self._cache:
                old_data = self._cache.pop(block_hash)
                self._stats['total_bytes'] -= len(old_data)

            # Add new entry
            self._cache[block_hash] = decompressed_data
            self._stats['puts'] += 1
            self._stats['total_bytes'] += len(decompressed_data)

            # LRU eviction
            while len(self._cache) > self._max_size:
                # Remove oldest (first in OrderedDict)
                evicted_hash, evicted_data = self._cache.popitem(last=False)
                self._stats['evictions'] += 1
                self._stats['total_bytes'] -= len(evicted_data)

                logger.debug(
                    f"Prefetch EVICT: {evicted_hash.hex()[:16]}... "
                    f"(freed {len(evicted_data)} bytes)"
                )

            logger.debug(
                f"Prefetch PUT: {block_hash.hex()[:16]}... "
                f"({len(decompressed_data)} bytes, cache_size={len(self._cache)})"
            )

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._stats['total_bytes'] = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'total_bytes': self._stats['total_bytes'],
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'puts': self._stats['puts'],
                'evictions': self._stats['evictions'],
                'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses']),
            }
