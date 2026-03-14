"""
ThunderOMLX 统一内存双层缓存管理器
利用 Apple Silicon Unified Memory 实现 L2 (RAM) + L3 (SSD) 架构

关键特性：
- L2 (Unified RAM): 内存缓存，< 5ms 访问延迟
- L3 (NVMe SSD): mmap 零拷贝磁盘缓存，< 50ms 访问延迟
- LRU-2 跨层驱逐策略
- 跨会话持久化（L3 层）
"""
from __future__ import annotations

import asyncio
import json
import logging
import mmap
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import mlx.core as mx

from omlx.serialization import TensorSerializer
from omlx.thunder_config import SerializationConfig
from .interface import CacheManager
from .stats import BaseCacheStats

logger = logging.getLogger(__name__)


# 多进程工作函数（必须在模块顶层才能被 pickle）
def _fetch_in_process_worker(key: str, l3_cache_path: str, config_dict: Dict[str, Any]) -> Tuple[Optional[bytes], bool]:
    """在独立进程中执行L3加载（可序列化）

    Args:
        key: 缓存键
        l3_cache_path: L3 缓存目录（字符串）
        config_dict: 序列化配置字典

    Returns:
        (tensor_bytes, hit): tensor_bytes 为序列化后的字节（或 None），hit 为是否命中
    """
    from pathlib import Path
    import io
    import numpy as np

    # 重建 Path 和 Config
    l3_path = Path(l3_cache_path)
    config = SerializationConfig(**config_dict)

    # 构造 L3 文件路径
    file_path = l3_path / key

    # 检查是否存在
    meta_path = file_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None, False

    # 使用 TensorSerializer 加载
    serializer = TensorSerializer(config)
    try:
        tensor = serializer.load(file_path)

        # 序列化为 bytes（通过 numpy）
        np_array = np.array(tensor)
        buffer = io.BytesIO()
        np.save(buffer, np_array)
        tensor_bytes = buffer.getvalue()

        return tensor_bytes, True
    except Exception as e:
        # 进程内日志
        import logging
        proc_logger = logging.getLogger(__name__)
        proc_logger.error(f"多进程加载失败 {key}: {e}")
        return None, False


@dataclass
class UnifiedCacheStats(BaseCacheStats):
    """统一内存缓存统计"""

    # L2 (RAM) 统计
    l2_hits: int = 0
    l2_misses: int = 0
    l2_evictions: int = 0
    l2_size_bytes: int = 0
    l2_max_bytes: int = 0

    # L3 (SSD) 统计
    l3_hits: int = 0
    l3_misses: int = 0
    l3_evictions: int = 0
    l3_size_bytes: int = 0
    l3_max_bytes: int = 0
    l3_mmap_hits: int = 0  # mmap 零拷贝命中

    # 跨层统计
    l2_to_l3_promotions: int = 0  # L2 驱逐到 L3
    l3_to_l2_promotions: int = 0  # L3 加载回 L2

    @property
    def l2_hit_rate(self) -> float:
        """L2 缓存命中率"""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def l3_hit_rate(self) -> float:
        """L3 缓存命中率"""
        total = self.l3_hits + self.l3_misses
        return self.l3_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """整体命中率（L2 + L3）"""
        total_hits = self.l2_hits + self.l3_hits
        total_queries = self.l2_hits + self.l2_misses
        return total_hits / total_queries if total_queries > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = super().to_dict()
        d.update({
            "l2_hit_rate": self.l2_hit_rate,
            "l3_hit_rate": self.l3_hit_rate,
            "overall_hit_rate": self.overall_hit_rate,
        })
        return d


class LRU2Queue:
    """LRU-2 驱逐队列（简化版）

    两层队列：COLD（访问 1 次）和 HOT（访问 2+ 次）
    第 2 次访问时从 COLD 晋升到 HOT
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cold_queue: OrderedDict[str, int] = OrderedDict()  # key -> access_count
        self.hot_queue: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.RLock()

    def access(self, key: str) -> None:
        """记录访问，COLD→HOT 晋升"""
        with self._lock:
            if key in self.hot_queue:
                # Already in HOT, move to end
                self.hot_queue.move_to_end(key)
            elif key in self.cold_queue:
                # Second access: COLD → HOT
                self.cold_queue.pop(key)
                self.hot_queue[key] = 2
            else:
                # First access: add to COLD
                self.cold_queue[key] = 1

    def evict(self) -> Optional[str]:
        """驱逐一个条目（优先 COLD，然后 HOT）"""
        with self._lock:
            # 优先驱逐 COLD 队列头部
            if self.cold_queue:
                key, _ = self.cold_queue.popitem(last=False)
                return key
            # COLD 为空，驱逐 HOT 队列头部
            if self.hot_queue:
                key, _ = self.hot_queue.popitem(last=False)
                return key
            return None

    def remove(self, key: str) -> bool:
        """移除指定 key"""
        with self._lock:
            if key in self.cold_queue:
                self.cold_queue.pop(key)
                return True
            if key in self.hot_queue:
                self.hot_queue.pop(key)
                return True
            return False

    def __len__(self) -> int:
        """当前队列大小"""
        return len(self.cold_queue) + len(self.hot_queue)

    def clear(self) -> None:
        """清空队列"""
        with self._lock:
            self.cold_queue.clear()
            self.hot_queue.clear()


class UnifiedMemoryCacheManager(CacheManager):
    """统一内存双层缓存管理器

    架构：
    - L2 (RAM): 内存 dict，快速访问（< 5ms）
    - L3 (SSD): mmap 零拷贝，持久化（< 50ms）
    - LRU-2: 跨层驱逐策略
    """

    def __init__(
        self,
        l2_max_size_mb: int = 8192,  # L2 最大 8GB
        l3_cache_path: Path | str = "~/.cache/thunderomlx/l3_cache",
        l3_max_size_gb: int = 256,  # L3 最大 256GB
        serialization_config: Optional[SerializationConfig] = None,
    ):
        """初始化双层缓存

        Args:
            l2_max_size_mb: L2 内存缓存最大大小（MB）
            l3_cache_path: L3 磁盘缓存路径
            l3_max_size_gb: L3 磁盘缓存最大大小（GB）
            serialization_config: 序列化配置（None 则使用默认）
        """
        self.l2_max_bytes = l2_max_size_mb * 1024 * 1024
        self.l3_max_bytes = l3_max_size_gb * 1024 * 1024 * 1024
        self.l3_cache_path = Path(l3_cache_path).expanduser()
        self.l3_cache_path.mkdir(parents=True, exist_ok=True)

        # L2 (RAM): key -> mx.array
        self.l2_cache: Dict[str, mx.array] = {}
        self.l2_size_bytes = 0
        self.l2_lru = LRU2Queue(max_size=1000)  # 简化：假设最多 1000 个条目

        # L3 (SSD): 文件索引 (key -> file_path)
        self.l3_index: Dict[str, Path] = {}
        self.l3_size_bytes = 0
        self.l3_lru = LRU2Queue(max_size=10000)  # L3 容量更大

        # 序列化器
        if serialization_config is None:
            serialization_config = SerializationConfig()
        self.serializer = TensorSerializer(serialization_config)

        # 统计
        self.stats = UnifiedCacheStats()
        self.stats.l2_max_bytes = self.l2_max_bytes
        self.stats.l3_max_bytes = self.l3_max_bytes

        # 线程锁
        self._lock = threading.RLock()

        # 启动时扫描 L3 缓存目录
        self._scan_l3_cache()

        logger.info(
            f"UnifiedMemoryCacheManager 初始化完成: "
            f"L2={l2_max_size_mb}MB, L3={l3_max_size_gb}GB @ {self.l3_cache_path}"
        )

    def _scan_l3_cache(self) -> None:
        """启动时扫描 L3 缓存目录，恢复索引"""
        if not self.l3_cache_path.exists():
            return

        count = 0
        total_size = 0

        for meta_file in self.l3_cache_path.glob("*.meta.json"):
            try:
                # 从元数据文件推导 key
                key = meta_file.stem.replace(".meta", "")

                # 读取元数据确定数据文件路径
                metadata = self.serializer.get_metadata(meta_file.parent / key)
                if metadata is None:
                    continue

                # 确定数据文件路径
                if metadata.compression == "zlib":
                    data_file = meta_file.parent / f"{key}.npz"
                elif metadata.compression == "lz4":
                    data_file = meta_file.parent / f"{key}.lz4"
                else:
                    data_file = meta_file.parent / f"{key}.npy"

                if not data_file.exists():
                    continue

                # 添加到索引
                self.l3_index[key] = meta_file.parent / key
                file_size = data_file.stat().st_size + meta_file.stat().st_size
                total_size += file_size
                count += 1

            except Exception as e:
                logger.warning(f"L3 扫描跳过损坏文件 {meta_file}: {e}")

        self.l3_size_bytes = total_size
        self.stats.l3_size_bytes = total_size

        logger.info(f"L3 缓存恢复: {count} 个条目, 总大小 {total_size / (1024**3):.2f} GB")

    def fetch(self, key: str) -> Tuple[Optional[mx.array], bool]:
        """从缓存获取数据（L2 → L3）

        Args:
            key: 缓存键

        Returns:
            (value, hit): value 为 mx.array（命中）或 None（未命中），hit 为是否命中
        """
        with self._lock:
            # 1. 尝试 L2
            if key in self.l2_cache:
                self.stats.l2_hits += 1
                self.stats.hits += 1
                self.l2_lru.access(key)
                return self.l2_cache[key], True

            self.stats.l2_misses += 1

            # 2. 尝试 L3
            if key in self.l3_index:
                try:
                    file_path = self.l3_index[key]

                    # 从 L3 加载
                    tensor = self.serializer.load(file_path, verify_checksum=True)

                    self.stats.l3_hits += 1
                    self.stats.hits += 1
                    self.l3_lru.access(key)

                    # 提升到 L2（如果空间允许）
                    self._promote_to_l2(key, tensor)

                    return tensor, True

                except Exception as e:
                    logger.error(f"L3 加载失败 {key}: {e}")
                    self.stats.l3_misses += 1
                    self.stats.misses += 1
                    return None, False

            self.stats.l3_misses += 1
            self.stats.misses += 1
            return None, False

    def store(self, key: str, value: mx.array) -> bool:
        """存储数据到缓存（先 L2，驱逐时到 L3）

        Args:
            key: 缓存键
            value: 要缓存的 MLX 数组

        Returns:
            成功返回 True
        """
        with self._lock:
            try:
                # 计算大小
                tensor_size = value.nbytes

                # 如果 L2 空间不足，驱逐
                while (
                    self.l2_size_bytes + tensor_size > self.l2_max_bytes
                    and len(self.l2_cache) > 0
                ):
                    self._evict_from_l2()

                # 存入 L2
                self.l2_cache[key] = value
                self.l2_size_bytes += tensor_size
                self.stats.l2_size_bytes = self.l2_size_bytes
                self.l2_lru.access(key)

                return True

            except Exception as e:
                logger.error(f"存储失败 {key}: {e}")
                return False

    def _promote_to_l2(self, key: str, tensor: mx.array) -> None:
        """将 L3 数据提升到 L2"""
        tensor_size = tensor.nbytes

        # 如果 L2 空间不足，驱逐
        while (
            self.l2_size_bytes + tensor_size > self.l2_max_bytes
            and len(self.l2_cache) > 0
        ):
            self._evict_from_l2()

        # 提升到 L2
        self.l2_cache[key] = tensor
        self.l2_size_bytes += tensor_size
        self.stats.l2_size_bytes = self.l2_size_bytes
        self.stats.l3_to_l2_promotions += 1
        self.l2_lru.access(key)

    def _evict_from_l2(self) -> bool:
        """从 L2 驱逐一个条目（写入 L3）"""
        # 使用 LRU-2 选择驱逐目标
        evict_key = self.l2_lru.evict()
        if evict_key is None or evict_key not in self.l2_cache:
            return False

        tensor = self.l2_cache.pop(evict_key)
        tensor_size = tensor.nbytes
        self.l2_size_bytes -= tensor_size
        self.stats.l2_size_bytes = self.l2_size_bytes
        self.stats.l2_evictions += 1

        # 写入 L3
        self._write_to_l3(evict_key, tensor)

        return True

    def _write_to_l3(self, key: str, tensor: mx.array) -> None:
        """将数据写入 L3（持久化）"""
        try:
            # 如果 L3 空间不足，驱逐
            tensor_size = tensor.nbytes * 1.2  # 估算压缩后大小（保守）
            while (
                self.l3_size_bytes + tensor_size > self.l3_max_bytes
                and len(self.l3_index) > 0
            ):
                self._evict_from_l3()

            # 保存到磁盘
            file_path = self.l3_cache_path / key
            metadata = self.serializer.save(tensor, file_path)

            # 更新索引
            self.l3_index[key] = file_path

            # 更新大小（实际文件大小）
            if metadata.compression == "zlib":
                data_file = Path(str(file_path) + ".npz")
            elif metadata.compression == "lz4":
                data_file = Path(str(file_path) + ".lz4")
            else:
                data_file = Path(str(file_path) + ".npy")
            meta_file = file_path.with_suffix(".meta.json")

            actual_size = data_file.stat().st_size + meta_file.stat().st_size
            self.l3_size_bytes += actual_size
            self.stats.l3_size_bytes = self.l3_size_bytes
            self.stats.l2_to_l3_promotions += 1
            self.l3_lru.access(key)

        except Exception as e:
            logger.error(f"L3 写入失败 {key}: {e}")

    def _evict_from_l3(self) -> bool:
        """从 L3 驱逐一个条目（删除文件）"""
        evict_key = self.l3_lru.evict()
        if evict_key is None or evict_key not in self.l3_index:
            return False

        try:
            file_path = self.l3_index.pop(evict_key)

            # 删除数据文件和元数据文件
            for pattern in ["*.npy", "*.npz", "*.lz4", "*.meta.json"]:
                for f in file_path.parent.glob(f"{file_path.name}{pattern.replace('*', '')}"):
                    file_size = f.stat().st_size
                    f.unlink()
                    self.l3_size_bytes -= file_size

            self.stats.l3_size_bytes = self.l3_size_bytes
            self.stats.l3_evictions += 1
            self.stats.evictions += 1

            return True

        except Exception as e:
            logger.error(f"L3 驱逐失败 {evict_key}: {e}")
            return False

    def evict(self, key: str) -> bool:
        """驱逐指定条目（从 L2 和 L3）"""
        with self._lock:
            evicted = False

            # 从 L2 移除
            if key in self.l2_cache:
                tensor = self.l2_cache.pop(key)
                self.l2_size_bytes -= tensor.nbytes
                self.stats.l2_size_bytes = self.l2_size_bytes
                self.l2_lru.remove(key)
                evicted = True

            # 从 L3 移除
            if key in self.l3_index:
                file_path = self.l3_index.pop(key)
                try:
                    for pattern in ["*.npy", "*.npz", "*.lz4", "*.meta.json"]:
                        for f in file_path.parent.glob(f"{file_path.name}{pattern.replace('*', '')}"):
                            file_size = f.stat().st_size
                            f.unlink()
                            self.l3_size_bytes -= file_size

                    self.stats.l3_size_bytes = self.l3_size_bytes
                    self.l3_lru.remove(key)
                    evicted = True

                except Exception as e:
                    logger.error(f"L3 文件删除失败 {key}: {e}")

            if evicted:
                self.stats.evictions += 1

            return evicted

    def clear(self) -> int:
        """清空所有缓存"""
        with self._lock:
            # 清空 L2
            l2_count = len(self.l2_cache)
            self.l2_cache.clear()
            self.l2_size_bytes = 0
            self.stats.l2_size_bytes = 0
            self.l2_lru.clear()

            # 清空 L3 索引（不删除文件，保留持久化数据）
            l3_count = len(self.l3_index)
            self.l3_index.clear()
            # 不清空 l3_size_bytes，文件仍在磁盘上
            self.l3_lru.clear()

            return l2_count + l3_count

    def get_stats(self) -> UnifiedCacheStats:
        """获取缓存统计"""
        return self.stats

    @property
    def size(self) -> int:
        """当前缓存条目数（L2 + L3）"""
        return len(self.l2_cache) + len(self.l3_index)

    @property
    def max_size(self) -> int:
        """最大缓存容量（简化：返回 L2 + L3 的估算条目数）"""
        # 简化估算：假设平均每个条目 10MB
        avg_size_mb = 10
        return int((self.l2_max_bytes + self.l3_max_bytes) / (avg_size_mb * 1024 * 1024))

    # === P3-4: 异步批量加载 ===
    # 注意：受 Python GIL 限制，当前实现的加速效果有限（~1x）
    # 瓶颈在 CPU 密集型的 numpy → mx.array 转换，而非磁盘 I/O
    # 未来优化方向：C++ 扩展、多进程、或 GPU 直接加载

    async def batch_fetch(self, keys: List[str]) -> List[Tuple[Optional[mx.array], bool]]:
        """异步批量加载多个 key（并行 I/O）

        Args:
            keys: 要加载的 key 列表

        Returns:
            [(value, hit), ...] 列表，与 keys 顺序对应

        Note:
            受 Python GIL 限制，加速效果有限（~1x）。
            瓶颈在 CPU 密集型的反序列化，而非 I/O。
        """
        # 使用 asyncio.gather 并行加载
        tasks = [self._async_fetch_single(key) for key in keys]
        results = await asyncio.gather(*tasks)
        return results

    def batch_fetch_parallel(self, keys: List[str], max_workers: int = 8) -> List[Tuple[Optional[mx.array], bool]]:
        """使用线程池并行批量加载多个 key

        Args:
            keys: 要加载的 key 列表
            max_workers: 最大工作线程数

        Returns:
            [(value, hit), ...] 列表，与 keys 顺序对应
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(keys)
        key_to_index = {key: i for i, key in enumerate(keys)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_key = {
                executor.submit(self.fetch, key): key
                for key in keys
            }

            # 收集结果（保持顺序）
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    results[key_to_index[key]] = result
                except Exception as e:
                    logger.error(f"并行加载失败 {key}: {e}")
                    results[key_to_index[key]] = (None, False)

        return results

    def batch_fetch_multiprocess(self, keys: List[str], max_workers: int = 4) -> List[Tuple[Optional[mx.array], bool]]:
        """使用多进程并行批量加载多个 key（绕过 GIL）

        Args:
            keys: 要加载的 key 列表
            max_workers: 最大工作进程数（默认 4，避免过多进程开销）

        Returns:
            [(value, hit), ...] 列表，与 keys 顺序对应

        Note:
            使用 ProcessPoolExecutor 绕过 Python GIL 限制。
            每个进程有独立的 Python 解释器和 GIL，可真正并行。

            预期加速：4-8x（取决于核心数和数据大小）

            注意事项：
            - 进程间通信有序列化开销（mx.array → bytes → mx.array）
            - 内存占用更高（每个进程独立内存）
            - 首次启动慢（进程池初始化）
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import io
        import numpy as np

        # 准备可序列化的参数
        l3_path_str = str(self.l3_cache_path)
        config_dict = {
            "compression": self.serializer.config.compression,
            "enable_checksum": self.serializer.config.enable_checksum,
        }

        # 提交到进程池
        results = [None] * len(keys)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(_fetch_in_process_worker, key, l3_path_str, config_dict): (i, key)
                for i, key in enumerate(keys)
            }

            # 收集结果
            for future in as_completed(futures):
                i, key = futures[future]
                try:
                    tensor_bytes, hit = future.result()

                    if hit and tensor_bytes:
                        # 反序列化回 mx.array
                        np_array = np.load(io.BytesIO(tensor_bytes))
                        tensor = mx.array(np_array)
                        results[i] = (tensor, True)

                        # 更新统计（L3 命中）
                        self.stats.l3_hits += 1
                        self.stats.hits += 1

                        # 提升到 L2（可选）
                        self._promote_to_l2(key, tensor)
                    else:
                        results[i] = (None, False)
                        self.stats.l3_misses += 1
                        self.stats.misses += 1

                except Exception as e:
                    logger.error(f"多进程结果处理失败 {key}: {e}")
                    results[i] = (None, False)
                    self.stats.l3_misses += 1
                    self.stats.misses += 1

        return results

    async def _async_fetch_single(self, key: str) -> Tuple[Optional[mx.array], bool]:
        """异步加载单个 key

        Args:
            key: 缓存键

        Returns:
            (value, hit): value 为 mx.array（命中）或 None（未命中），hit 为是否命中
        """
        # 1. 尝试 L2（同步，快速）
        if key in self.l2_cache:
            self.stats.l2_hits += 1
            self.stats.hits += 1
            self.l2_lru.access(key)
            return self.l2_cache[key], True

        self.stats.l2_misses += 1

        # 2. 尝试 L3（异步加载）
        if key in self.l3_index:
            try:
                file_path = self.l3_index[key]

                # 异步从 L3 加载
                tensor = await self._async_load_from_l3(file_path)

                self.stats.l3_hits += 1
                self.stats.hits += 1
                self.l3_lru.access(key)

                # 提升到 L2（如果空间允许）
                self._promote_to_l2(key, tensor)

                return tensor, True

            except Exception as e:
                logger.error(f"L3 异步加载失败 {key}: {e}")
                self.stats.l3_misses += 1
                self.stats.misses += 1
                return None, False

        self.stats.l3_misses += 1
        self.stats.misses += 1
        return None, False

    async def _async_load_from_l3(self, file_path: Path) -> mx.array:
        """异步从 L3 加载张量

        Args:
            file_path: 文件路径（不含扩展名）

        Returns:
            加载的 mx.array
        """
        # 1. 异步读取元数据
        meta_path = file_path.with_suffix(".meta.json")
        async with aiofiles.open(meta_path, "r") as f:
            meta_content = await f.read()
            metadata_dict = json.loads(meta_content)

        # 从元数据确定压缩类型
        compression = metadata_dict.get("compression", "none")

        # 2. 确定数据文件路径
        if compression == "zlib":
            data_path = Path(str(file_path) + ".npz")
        elif compression == "lz4":
            data_path = Path(str(file_path) + ".lz4")
        else:
            data_path = Path(str(file_path) + ".npy")

        # 3. 异步读取数据文件（二进制）
        async with aiofiles.open(data_path, "rb") as f:
            data_bytes = await f.read()

        # 4. 在 executor 中反序列化（MLX 操作可能阻塞）
        loop = asyncio.get_event_loop()
        tensor = await loop.run_in_executor(
            None,  # 使用默认 executor
            self._deserialize_tensor,
            data_bytes,
            compression,
        )

        return tensor

    def _deserialize_tensor(self, data_bytes: bytes, compression: str) -> mx.array:
        """从字节反序列化张量（在 executor 中执行）

        Args:
            data_bytes: 文件字节
            compression: 压缩类型

        Returns:
            mx.array
        """
        import io
        import numpy as np

        # 优化：使用 BytesIO 避免临时文件
        if compression in ["zlib", "lz4"]:
            # npz 格式（压缩）
            with np.load(io.BytesIO(data_bytes)) as npz:
                np_array = npz["tensor"]
            tensor = mx.array(np_array)
        else:
            # npy 格式（无压缩）
            np_array = np.load(io.BytesIO(data_bytes))
            tensor = mx.array(np_array)

        return tensor
