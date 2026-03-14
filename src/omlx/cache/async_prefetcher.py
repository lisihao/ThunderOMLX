# SPDX-License-Identifier: Apache-2.0
"""
异步块预取器，使用线程池并行从 SSD 加载块。

P1-5: Smart Prefetch - Async Prefetching Component
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Any

logger = logging.getLogger(__name__)


class AsyncPrefetcher:
    """
    异步块预取器，使用线程池并行从 SSD 加载块。

    关键特性：
    - 4 线程并行 I/O（参考 ThunderLLAMA 配置）
    - 非阻塞预取（不影响主流程）
    - 自动容量管理

    设计参考：ThunderLLAMA thunder-lmcache-storage.cpp:prefetch_hot_chunks()
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 100
    ):
        """
        初始化异步预取器。

        Args:
            num_workers: 工作线程数（默认 4，与 ThunderLLAMA 一致）
            max_queue_size: 最大预取队列大小
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

        self._executor: Optional[ThreadPoolExecutor] = None
        self._queue_lock = threading.Lock()
        self._running = False
        self._active_tasks = 0  # 当前活跃的预取任务数

        logger.info(
            f"AsyncPrefetcher initialized: {num_workers} workers, "
            f"max queue size {max_queue_size}"
        )

    def start(self) -> None:
        """启动预取器（初始化线程池）。"""
        if self._running:
            logger.warning("Prefetcher already running")
            return

        self._executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="prefetch-worker-"
        )
        self._running = True

        logger.info(f"AsyncPrefetcher started with {self.num_workers} workers")

    def stop(self) -> None:
        """停止预取器并等待所有任务完成。"""
        if not self._running:
            return

        self._running = False

        if self._executor:
            logger.info("Shutting down prefetch workers...")
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info(f"AsyncPrefetcher stopped (completed {self._active_tasks} tasks)")

    def prefetch_blocks(
        self,
        block_hashes: List[bytes],
        load_fn: Callable[[bytes], Optional[Any]],
        on_loaded: Optional[Callable[[bytes, Any], None]] = None
    ) -> None:
        """
        异步预取块列表（并行 I/O）。

        Args:
            block_hashes: 要预取的块哈希列表
            load_fn: 加载函数 (block_hash) -> block_data
            on_loaded: 加载完成回调 (block_hash, block_data) -> None
        """
        if not self._running:
            logger.warning("Prefetcher not running, call start() first")
            return

        if not block_hashes:
            logger.debug("No blocks to prefetch")
            return

        logger.info(
            f"🔥 Prefetching {len(block_hashes)} blocks "
            f"(parallel I/O with {self.num_workers} threads)"
        )

        # 提交并行任务
        with self._queue_lock:
            for block_hash in block_hashes:
                if not self._running:
                    break  # 如果正在停止，停止提交新任务

                self._executor.submit(
                    self._load_and_callback,
                    block_hash,
                    load_fn,
                    on_loaded
                )
                self._active_tasks += 1

        logger.debug(f"Submitted {len(block_hashes)} prefetch tasks")

    def _load_and_callback(
        self,
        block_hash: bytes,
        load_fn: Callable[[bytes], Optional[Any]],
        on_loaded: Optional[Callable[[bytes, Any], None]]
    ) -> None:
        """
        加载块并调用回调（在工作线程中执行）。

        Args:
            block_hash: 块哈希
            load_fn: 加载函数
            on_loaded: 完成回调
        """
        try:
            # 从 SSD 加载
            block_data = load_fn(block_hash)

            if block_data is None:
                logger.debug(
                    f"Block {block_hash.hex()[:8]}... not found on SSD"
                )
                return

            # 调用完成回调（通常是插入内存缓存）
            if on_loaded:
                on_loaded(block_hash, block_data)

            logger.debug(
                f"✅ Prefetched block {block_hash.hex()[:8]}... from SSD"
            )

        except Exception as e:
            logger.error(
                f"Failed to prefetch block {block_hash.hex()[:8]}...: {e}"
            )

    def get_status(self) -> dict:
        """
        获取预取器状态。

        Returns:
            状态字典，包含：
            - running: 是否运行中
            - num_workers: 工作线程数
            - active_tasks: 活跃任务数
        """
        return {
            'running': self._running,
            'num_workers': self.num_workers,
            'active_tasks': self._active_tasks
        }

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"AsyncPrefetcher("
            f"running={status['running']}, "
            f"workers={status['num_workers']}, "
            f"active={status['active_tasks']})"
        )
