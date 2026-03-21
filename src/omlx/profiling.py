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

import json
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


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class MemorySnapshot:
    """内存快照"""
    label: str
    mlx_mb: float
    timestamp_ms: float

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "mlx_mb": round(self.mlx_mb, 2),
            "timestamp_ms": round(self.timestamp_ms, 3),
        }


@dataclass
class RequestStats:
    """请求统计"""
    request_id: str
    ttft_ms: float
    tpot_ms: float
    total_tokens: int

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "ttft_ms": round(self.ttft_ms, 3),
            "tpot_ms": round(self.tpot_ms, 3),
            "total_tokens": self.total_tokens,
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

        # 扩展统计（ThunderOMLX 增强）
        self._cache_stats: Dict[str, CacheStats] = defaultdict(CacheStats)
        self._memory_snapshots: List[MemorySnapshot] = []
        self._request_stats: List[RequestStats] = []
        self._start_time = time.perf_counter()

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
            # 重置扩展统计
            self._cache_stats.clear()
            self._memory_snapshots.clear()
            self._request_stats.clear()
            self._start_time = time.perf_counter()

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

    # ====== ThunderOMLX 扩展功能 ======

    def _elapsed_ms(self) -> float:
        """从启动到现在的毫秒数"""
        return (time.perf_counter() - self._start_time) * 1000

    def record_cache_hit(self, cache_type: str):
        """记录缓存命中

        Args:
            cache_type: 缓存类型（如 "l2", "l3", "prefix"）
        """
        if not self.enabled:
            return

        with self._lock:
            self._cache_stats[cache_type].hits += 1

    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中

        Args:
            cache_type: 缓存类型（如 "l2", "l3", "prefix"）
        """
        if not self.enabled:
            return

        with self._lock:
            self._cache_stats[cache_type].misses += 1

    def snapshot_memory(self, label: str, mlx_mb: Optional[float] = None):
        """记录内存快照

        Args:
            label: 快照标签
            mlx_mb: MLX 张量内存（MB），如果为 None 则尝试自动获取
        """
        if not self.enabled:
            return

        if mlx_mb is None:
            # 尝试获取 MLX 内存使用
            try:
                import mlx.core as mx
                # MLX 没有直接的内存统计 API，使用占位值
                mlx_mb = 0.0
            except ImportError:
                mlx_mb = 0.0

        with self._lock:
            snapshot = MemorySnapshot(
                label=label,
                mlx_mb=mlx_mb,
                timestamp_ms=self._elapsed_ms(),
            )
            self._memory_snapshots.append(snapshot)

    def record_request(self, request_id: str, ttft_ms: float, tpot_ms: float, total_tokens: int):
        """记录请求统计

        Args:
            request_id: 请求 ID
            ttft_ms: Time To First Token（毫秒）
            tpot_ms: Time Per Output Token（毫秒）
            total_tokens: 总 token 数
        """
        if not self.enabled:
            return

        with self._lock:
            stats = RequestStats(
                request_id=request_id,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_tokens=total_tokens,
            )
            self._request_stats.append(stats)

    def save_json(self, path: str):
        """保存统计数据到 JSON 文件

        Args:
            path: 输出文件路径
        """
        if not self.enabled:
            return

        # 构建完整统计数据
        stats = self.get_stats()

        # 添加扩展统计
        with self._lock:
            # Cache 统计
            cache_stats_dict = {
                name: cstats.to_dict()
                for name, cstats in self._cache_stats.items()
            }

            # 整体缓存命中率
            total_hits = sum(cs.hits for cs in self._cache_stats.values())
            total_misses = sum(cs.misses for cs in self._cache_stats.values())
            total_cache = total_hits + total_misses
            overall_hit_rate = total_hits / total_cache if total_cache > 0 else 0.0

            # 请求统计摘要
            request_summary = {}
            if self._request_stats:
                total_requests = len(self._request_stats)
                avg_ttft = sum(r.ttft_ms for r in self._request_stats) / total_requests
                avg_tpot = sum(r.tpot_ms for r in self._request_stats) / total_requests
                total_tokens = sum(r.total_tokens for r in self._request_stats)

                request_summary = {
                    "total_requests": total_requests,
                    "avg_ttft_ms": round(avg_ttft, 3),
                    "avg_tpot_ms": round(avg_tpot, 3),
                    "total_tokens": total_tokens,
                }

            # 构建输出
            output = {
                "summary": {
                    "total_time_ms": stats.get("total_time_ms", 0.0),
                    "total_requests": len(self._request_stats),
                    "avg_ttft_ms": request_summary.get("avg_ttft_ms", 0.0),
                    "avg_tpot_ms": request_summary.get("avg_tpot_ms", 0.0),
                    "cache_hit_rate": round(overall_hit_rate, 4),
                },
                "timers": stats.get("operations", {}),
                "cache_stats": cache_stats_dict,
                "memory_snapshots": [snap.to_dict() for snap in self._memory_snapshots],
                "request_stats": [req.to_dict() for req in self._request_stats],
            }

        with open(path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✅ Profiler report saved to: {path}")


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
