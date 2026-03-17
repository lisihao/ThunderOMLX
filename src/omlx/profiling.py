"""
通用性能分析框架

基于 paged_ssd_cache.py 中的 stats 框架扩展而来。
提供可重用的性能计时和分析功能。

使用方式：
    from omlx.profiling import PerformanceProfiler

    profiler = PerformanceProfiler()

    with profiler.section("my_operation"):
        # 执行操作
        pass

    # 获取统计
    stats = profiler.get_stats()
"""

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """单个操作的timing统计"""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        """平均时间（毫秒）"""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def update(self, elapsed_ms: float):
        """更新统计"""
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'count': self.count,
            'total_ms': round(self.total_ms, 3),
            'avg_ms': round(self.avg_ms, 3),
            'min_ms': round(self.min_ms, 3) if self.min_ms != float('inf') else 0.0,
            'max_ms': round(self.max_ms, 3),
            'percent': 0.0  # 由 get_stats() 计算
        }


class PerformanceProfiler:
    """性能分析器

    特性：
    1. 线程安全
    2. 嵌套timing支持
    3. 自动百分比计算
    4. 最小开销（仅在启用时计时）

    Examples:
        # 方式1: Context manager
        with profiler.section("operation"):
            do_work()

        # 方式2: 手动开始/结束
        profiler.start("operation")
        do_work()
        profiler.end("operation")

        # 方式3: 直接记录时间
        profiler.record("operation", elapsed_ms=123.4)
    """

    def __init__(self, enabled: bool = True):
        """初始化性能分析器

        Args:
            enabled: 是否启用profiling（禁用时开销几乎为0）
        """
        self.enabled = enabled
        self._stats: Dict[str, TimingStats] = defaultdict(TimingStats)
        self._lock = threading.Lock()

        # 当前活动的timing（支持嵌套）
        self._current: Dict[int, List[tuple]] = defaultdict(list)  # thread_id -> [(name, start_time)]

    @contextmanager
    def section(self, name: str):
        """Context manager for timing a section

        Args:
            name: Section name (支持层级，如 "prefill.qkv_proj")

        Example:
            with profiler.section("model.forward"):
                with profiler.section("model.forward.qkv"):
                    # ...
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(name, elapsed_ms)

    def start(self, name: str):
        """手动开始计时

        Args:
            name: Operation name
        """
        if not self.enabled:
            return

        thread_id = threading.get_ident()
        start_time = time.perf_counter()
        self._current[thread_id].append((name, start_time))

    def end(self, name: Optional[str] = None):
        """手动结束计时

        Args:
            name: Operation name (可选，用于验证)
        """
        if not self.enabled:
            return

        end_time = time.perf_counter()
        thread_id = threading.get_ident()

        if not self._current[thread_id]:
            logger.warning(f"end() called without matching start() for '{name}'")
            return

        recorded_name, start_time = self._current[thread_id].pop()

        if name and name != recorded_name:
            logger.warning(
                f"Mismatched end(): expected '{recorded_name}', got '{name}'"
            )

        elapsed_ms = (end_time - start_time) * 1000
        self.record(recorded_name, elapsed_ms)

    def record(self, name: str, elapsed_ms: float):
        """直接记录一个timing

        Args:
            name: Operation name
            elapsed_ms: Elapsed time in milliseconds
        """
        if not self.enabled:
            return

        with self._lock:
            self._stats[name].update(elapsed_ms)

    def get_stats(self, top_n: Optional[int] = None) -> Dict:
        """获取统计信息

        Args:
            top_n: 只返回前N个最耗时的操作

        Returns:
            {
                'total_time_ms': float,
                'operations': {
                    'op_name': {
                        'count': int,
                        'total_ms': float,
                        'avg_ms': float,
                        'min_ms': float,
                        'max_ms': float,
                        'percent': float
                    },
                    ...
                },
                'top_operations': [
                    ('op_name', stats_dict),
                    ...
                ]
            }
        """
        with self._lock:
            # 计算总时间（使用最大的操作时间作为基准）
            # 通常是以".total"结尾的顶层操作
            total_time_ms = max(
                (stats.total_ms for stats in self._stats.values()),
                default=0.0
            )

            # 转换为字典并计算百分比
            operations = {}
            for name, stats in self._stats.items():
                op_dict = stats.to_dict()
                if total_time_ms > 0:
                    op_dict['percent'] = (stats.total_ms / total_time_ms) * 100
                operations[name] = op_dict

            # 排序：按 total_ms 降序
            sorted_ops = sorted(
                operations.items(),
                key=lambda x: x[1]['total_ms'],
                reverse=True
            )

            if top_n:
                sorted_ops = sorted_ops[:top_n]

            return {
                'total_time_ms': round(total_time_ms, 3),
                'operations': operations,
                'top_operations': sorted_ops
            }

    def print_stats(self, top_n: int = 20, min_percent: float = 1.0):
        """打印统计信息

        Args:
            top_n: 显示前N个最耗时的操作
            min_percent: 只显示占比 >= min_percent 的操作
        """
        stats = self.get_stats()

        print("=" * 80)
        print("Performance Profiling Results")
        print("=" * 80)
        print(f"Total Time: {stats['total_time_ms']:.1f} ms\n")

        print(f"{'Operation':<50} {'Count':>8} {'Avg (ms)':>10} {'Total (ms)':>12} {'%':>6}")
        print("-" * 80)

        count = 0
        for name, op_stats in stats['top_operations']:
            if op_stats['percent'] < min_percent:
                continue

            if count >= top_n:
                break

            print(
                f"{name:<50} {op_stats['count']:>8} "
                f"{op_stats['avg_ms']:>10.2f} {op_stats['total_ms']:>12.1f} "
                f"{op_stats['percent']:>5.1f}%"
            )
            count += 1

        print("=" * 80)

    def reset(self):
        """重置所有统计"""
        with self._lock:
            self._stats.clear()
            self._current.clear()

    def merge_from(self, other: 'PerformanceProfiler'):
        """合并另一个profiler的统计

        Args:
            other: 另一个 PerformanceProfiler 实例
        """
        with self._lock:
            for name, stats in other._stats.items():
                self._stats[name].count += stats.count
                self._stats[name].total_ms += stats.total_ms
                self._stats[name].min_ms = min(self._stats[name].min_ms, stats.min_ms)
                self._stats[name].max_ms = max(self._stats[name].max_ms, stats.max_ms)


# 全局单例（可选使用）
_global_profiler: Optional[PerformanceProfiler] = None
_global_profiler_lock = threading.Lock()


def get_global_profiler() -> PerformanceProfiler:
    """获取全局profiler单例"""
    global _global_profiler

    if _global_profiler is None:
        with _global_profiler_lock:
            if _global_profiler is None:
                # 从环境变量读取是否启用
                import os
                enabled = os.getenv('OMLX_ENABLE_PROFILING', 'false').lower() == 'true'
                _global_profiler = PerformanceProfiler(enabled=enabled)

    return _global_profiler


def reset_global_profiler():
    """重置全局profiler"""
    global _global_profiler

    with _global_profiler_lock:
        if _global_profiler is not None:
            _global_profiler.reset()


# 便捷函数
def profile_section(name: str):
    """便捷的context manager"""
    return get_global_profiler().section(name)


def print_profiling_stats(top_n: int = 20, min_percent: float = 1.0):
    """打印全局profiler的统计"""
    get_global_profiler().print_stats(top_n=top_n, min_percent=min_percent)
