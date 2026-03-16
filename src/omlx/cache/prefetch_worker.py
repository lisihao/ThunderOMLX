# SPDX-License-Identifier: Apache-2.0
"""
Prefetch Worker for Async I/O optimization.

This module implements a background worker thread that pre-fetches and decompresses
cache blocks, overlapping I/O with GPU computation.

Key features:
- Background I/O thread (avoids blocking inference)
- File read + decompression only (no Metal API calls)
- Non-blocking prefetch queue
- Integrated with PrefetchCache for storage
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .paged_ssd_cache import PagedSSDCacheManager
    from .prefetch_cache import PrefetchCache

logger = logging.getLogger(__name__)


class PrefetchWorker:
    """
    Background worker for prefetching cache blocks.

    Runs in a separate thread, performing I/O and decompression operations
    without touching Metal GPU resources (safe for background execution).

    Example:
        >>> worker = PrefetchWorker(cache_manager, prefetch_cache)
        >>> worker.prefetch(next_block_hash)  # Non-blocking
        >>> # ... later in main thread ...
        >>> data = prefetch_cache.get(next_block_hash)  # Fast!
    """

    def __init__(
        self,
        cache_manager: 'PagedSSDCacheManager',
        prefetch_cache: 'PrefetchCache',
        queue_size: int = 10,
    ):
        """
        Initialize prefetch worker.

        Args:
            cache_manager: PagedSSDCacheManager instance for index access.
            prefetch_cache: PrefetchCache instance for storing results.
            queue_size: Maximum size of prefetch request queue (default 10).
        """
        self._manager = cache_manager
        self._prefetch_cache = prefetch_cache
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_size)
        self._shutdown = threading.Event()

        # Statistics
        self._stats = {
            'requests': 0,
            'completed': 0,
            'dropped': 0,  # Queue full
            'not_found': 0,  # Block not in index
            'errors': 0,
            'total_io_time_ms': 0,
            'total_decompress_time_ms': 0,
        }
        self._stats_lock = threading.Lock()

        # Background thread
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="cache-prefetch-worker",
            daemon=True,
        )
        self._thread.start()

        logger.info("✅ PrefetchWorker started")

    def prefetch(self, block_hash: bytes) -> None:
        """
        Request prefetch of a block (non-blocking).

        Args:
            block_hash: Block content hash to prefetch.
        """
        try:
            self._queue.put_nowait(block_hash)
            with self._stats_lock:
                self._stats['requests'] += 1

            logger.debug(f"Prefetch QUEUE: {block_hash.hex()[:16]}... (queue_size={self._queue.qsize()})")
        except queue.Full:
            # Queue full - drop request (don't block main thread)
            with self._stats_lock:
                self._stats['dropped'] += 1
            logger.debug(f"Prefetch DROPPED (queue full): {block_hash.hex()[:16]}...")

    def _worker_loop(self) -> None:
        """
        Background worker loop.

        Continuously processes prefetch requests:
        1. Read file from SSD
        2. Decompress (if needed)
        3. Store in prefetch_cache
        """
        while not self._shutdown.is_set():
            try:
                # Wait for next request (0.5s timeout for shutdown check)
                block_hash = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Check if already in prefetch cache
            if self._prefetch_cache.get(block_hash) is not None:
                logger.debug(f"Prefetch SKIP (already cached): {block_hash.hex()[:16]}...")
                continue

            # Check if block exists in index
            metadata = self._manager._index.get(block_hash)
            if metadata is None:
                with self._stats_lock:
                    self._stats['not_found'] += 1
                logger.debug(f"Prefetch NOT_FOUND: {block_hash.hex()[:16]}...")
                continue

            file_path = metadata.file_path
            if not file_path.exists():
                with self._stats_lock:
                    self._stats['not_found'] += 1
                logger.debug(f"Prefetch MISSING_FILE: {file_path}")
                continue

            try:
                # --- Phase 1: File I/O ---
                io_start = time.time()
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                io_elapsed_ms = (time.time() - io_start) * 1000

                # --- Phase 2: Decompression (if needed) ---
                decompress_elapsed_ms = 0
                if file_path.suffix in ('.lz4', '.zst'):
                    decompress_start = time.time()

                    if file_path.suffix == '.lz4':
                        import lz4.frame
                        decompressed = lz4.frame.decompress(file_data)
                    else:
                        # Legacy .zst files use zlib
                        import zlib
                        decompressed = zlib.decompress(file_data)

                    decompress_elapsed_ms = (time.time() - decompress_start) * 1000
                else:
                    # Uncompressed file
                    decompressed = file_data

                # --- Phase 3: Store in prefetch cache ---
                self._prefetch_cache.put(block_hash, decompressed)

                with self._stats_lock:
                    self._stats['completed'] += 1
                    self._stats['total_io_time_ms'] += io_elapsed_ms
                    self._stats['total_decompress_time_ms'] += decompress_elapsed_ms

                logger.debug(
                    f"Prefetch DONE: {block_hash.hex()[:16]}... "
                    f"(I/O={io_elapsed_ms:.1f}ms, decompress={decompress_elapsed_ms:.1f}ms, "
                    f"size={len(decompressed)} bytes)"
                )

            except Exception as e:
                with self._stats_lock:
                    self._stats['errors'] += 1
                logger.debug(f"Prefetch ERROR for {block_hash.hex()[:16]}: {e}")

    def shutdown(self) -> None:
        """Stop the prefetch worker thread."""
        self._shutdown.set()
        self._thread.join(timeout=1.0)
        logger.info("PrefetchWorker stopped")

    def get_stats(self) -> dict:
        """Get prefetch worker statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
            if stats['completed'] > 0:
                stats['avg_io_time_ms'] = stats['total_io_time_ms'] / stats['completed']
                stats['avg_decompress_time_ms'] = stats['total_decompress_time_ms'] / stats['completed']
            else:
                stats['avg_io_time_ms'] = 0
                stats['avg_decompress_time_ms'] = 0
            return stats
