# SPDX-License-Identifier: Apache-2.0
"""
访问频率追踪器，用于智能预取决策。

P1-5: Smart Prefetch - Access Frequency Tracking Component
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class AccessFrequencyTracker:
    """
    追踪块访问频率，用于智能预取决策。

    关键特性：
    - 线程安全，支持高并发访问统计
    - 自动访问次数衰减，避免旧数据累积
    - 热块识别（按访问频率降序排序）

    设计参考：ThunderLLAMA thunder-lmcache-storage.cpp
    """

    def __init__(self, decay_interval: float = 3600.0):
        """
        初始化访问追踪器。

        Args:
            decay_interval: 访问次数衰减周期（秒），默认 1 小时
        """
        self._access_count: Dict[bytes, int] = defaultdict(int)
        self._last_access: Dict[bytes, float] = {}
        self._lock = threading.Lock()
        self._decay_interval = decay_interval
        self._last_decay = time.time()

        logger.info(
            f"AccessFrequencyTracker initialized: decay_interval={decay_interval}s"
        )

    def track_access(self, block_hash: bytes) -> None:
        """
        记录块访问。

        Args:
            block_hash: 块哈希值
        """
        with self._lock:
            self._access_count[block_hash] += 1
            self._last_access[block_hash] = time.time()

            # 定期衰减（避免旧访问数据累积）
            if time.time() - self._last_decay > self._decay_interval:
                self._apply_decay()

    def get_hot_blocks(
        self,
        top_n: int,
        min_access_count: int = 2
    ) -> List[Tuple[bytes, int]]:
        """
        获取热块列表（按访问频率降序）。

        Args:
            top_n: 返回的最大块数
            min_access_count: 最小访问次数阈值

        Returns:
            [(block_hash, access_count), ...] 列表，按访问次数降序
        """
        with self._lock:
            # 过滤并排序
            candidates = [
                (hash, count)
                for hash, count in self._access_count.items()
                if count >= min_access_count
            ]

            # 按访问次数降序排序（热块优先）
            candidates.sort(key=lambda x: x[1], reverse=True)

            return candidates[:top_n]

    def _apply_decay(self) -> None:
        """
        应用访问次数衰减（减半），避免旧数据累积。

        内部方法，需要在持有锁的情况下调用。
        """
        decay_count = 0

        for hash in list(self._access_count.keys()):
            self._access_count[hash] //= 2

            # 移除衰减到 0 的记录
            if self._access_count[hash] == 0:
                del self._access_count[hash]
                if hash in self._last_access:
                    del self._last_access[hash]
                decay_count += 1

        self._last_decay = time.time()

        if decay_count > 0:
            logger.debug(
                f"Access frequency decay: removed {decay_count} cold blocks"
            )

    def reset(self) -> None:
        """重置所有访问统计。"""
        with self._lock:
            count_before = len(self._access_count)
            self._access_count.clear()
            self._last_access.clear()
            self._last_decay = time.time()

            logger.info(f"Access tracker reset: cleared {count_before} blocks")

    def get_stats(self) -> Dict[str, int]:
        """
        获取统计信息。

        Returns:
            统计字典，包含：
            - total_blocks: 追踪的块总数
            - total_accesses: 总访问次数
            - avg_access_per_block: 平均每块访问次数
        """
        with self._lock:
            total_blocks = len(self._access_count)
            total_accesses = sum(self._access_count.values())

            return {
                'total_blocks': total_blocks,
                'total_accesses': total_accesses,
                'avg_access_per_block': (
                    total_accesses // total_blocks
                    if total_blocks > 0 else 0
                )
            }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"AccessFrequencyTracker("
            f"blocks={stats['total_blocks']}, "
            f"accesses={stats['total_accesses']}, "
            f"avg={stats['avg_access_per_block']})"
        )
