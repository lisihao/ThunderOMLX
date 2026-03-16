# SPDX-License-Identifier: Apache-2.0
"""
异步缓存保存执行器

Phase 2: 使 store_cache() 调用非阻塞，减少推理线程等待时间
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class CacheSaveExecutor:
    """
    单线程执行器用于异步 store_cache 操作。

    Phase 2 优化：
    - 推理线程调用 submit_save() 后立即返回
    - 后台线程执行 store_cache()
    - 优雅降级：队列满时丢弃，不阻塞推理

    设计原则：
    - 单 worker 确保顺序执行（Metal 线程安全）
    - max_pending 限制防止内存溢出
    - 统计接口用于监控和调试
    """

    def __init__(self, max_pending: int = 20):
        """
        初始化异步执行器。

        Args:
            max_pending: 最大待处理任务数（默认 20）
        """
        self._executor = ThreadPoolExecutor(
            max_workers=1,  # 单 worker（Metal 安全）
            thread_name_prefix="cache-save-worker"
        )
        self._pending_count = 0
        self._pending_lock = threading.Lock()
        self._max_pending = max_pending

        # 统计信息
        self._stats_lock = threading.Lock()
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "dropped": 0,
            "errors": 0,
        }

        logger.info(
            f"✅ CacheSaveExecutor initialized (max_pending={max_pending})"
        )

    def submit_save(
        self,
        cache_manager,
        request_id: str,
        tokens: list,
        cache_data,
        **kwargs
    ) -> bool:
        """
        提交 store_cache 任务（非阻塞）。

        Args:
            cache_manager: BlockAwarePrefixCache 实例
            request_id: 请求 ID
            tokens: Token 序列
            cache_data: 缓存数据
            **kwargs: 传递给 store_cache 的其他参数

        Returns:
            True if submitted, False if queue full (dropped)
        """
        with self._pending_lock:
            if self._pending_count >= self._max_pending:
                # 队列满，优雅降级：丢弃此次保存
                with self._stats_lock:
                    self._stats["dropped"] += 1
                logger.warning(
                    f"⚠️ Cache save queue full ({self._pending_count}/{self._max_pending}), "
                    f"dropping save for request {request_id}"
                )
                return False

            self._pending_count += 1
            with self._stats_lock:
                self._stats["submitted"] += 1

        def _save_task():
            """后台保存任务"""
            try:
                # 调用 store_cache（在后台线程执行）
                cache_manager.store_cache(
                    request_id, tokens, cache_data, **kwargs
                )
                with self._stats_lock:
                    self._stats["completed"] += 1
                logger.debug(
                    f"✅ Background cache save completed for request {request_id}"
                )
            except Exception as e:
                with self._stats_lock:
                    self._stats["errors"] += 1
                logger.error(
                    f"❌ Background cache save failed for request {request_id}: {e}",
                    exc_info=True
                )
            finally:
                with self._pending_lock:
                    self._pending_count -= 1

        # 提交到后台线程
        self._executor.submit(_save_task)
        return True

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self._stats_lock:
            return self._stats.copy()

    def get_pending_count(self) -> int:
        """获取待处理任务数"""
        with self._pending_lock:
            return self._pending_count

    def shutdown(self, wait: bool = True):
        """
        关闭执行器。

        Args:
            wait: 是否等待所有任务完成
        """
        logger.info(
            f"🛑 Shutting down CacheSaveExecutor (wait={wait}, "
            f"pending={self._pending_count})"
        )
        self._executor.shutdown(wait=wait)

        with self._stats_lock:
            logger.info(f"📊 Final stats: {self._stats}")
