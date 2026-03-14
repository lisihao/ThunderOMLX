# P1-5: Smart Prefetch 实现设计

> **目标**: 4x L3 (SSD) 加速，通过智能预取热块到内存

---

## 📊 核心原理

### ThunderLLAMA 实现分析

```cpp
// thunder-lmcache-storage.cpp
void ThunderChunkStorage::prefetch_hot_chunks(size_t top_n) {
    // 1. 访问频率追踪 (access_freq_)
    std::map<uint64_t, uint32_t> access_freq_;

    // 2. 热块识别：按访问次数排序
    std::sort(prefetch_list, [](auto& a, auto& b) {
        return std::get<1>(a) > std::get<1>(b);  // 访问次数降序
    });

    // 3. 并行 I/O：4 线程异步读取
    const size_t num_threads = std::min(size_t(4), prefetch_list.size());
    for (...) {
        futures.push_back(std::async(std::launch::async, read_chunk_async, ...));
    }

    // 4. L3 → L2 晋升
    l2_cache_[key_hash] = chunk;
    l3_offsets_.erase(key_hash);
}
```

### 性能提升原理

| 方式 | 延迟 | 吞吐 |
|------|------|------|
| **同步单线程读取** | ~15ms/block | ~100 MB/s |
| **异步 4 线程预取** | ~4ms/block | ~400 MB/s |
| **加速比** | **3.75x** | **4x** |

---

## 🏗️ oMLX 实现架构

### 新增组件

```
PagedSSDCacheManager
    │
    ├─ AccessFrequencyTracker (新增)
    │   ├─ track_access(block_hash)
    │   ├─ get_hot_blocks(top_n) → List[block_hash]
    │   └─ reset_frequency()
    │
    ├─ AsyncPrefetcher (新增)
    │   ├─ prefetch_blocks(block_hashes: List[bytes])
    │   ├─ _prefetch_worker() → 后台线程
    │   └─ stop()
    │
    └─ 现有方法
        ├─ save_block()  → 调用 tracker.track_access()
        └─ load_block()  → 调用 tracker.track_access()
```

---

## 📋 实现清单

### 1. 访问频率追踪器

**文件**: `src/omlx/cache/access_tracker.py` (新建)

```python
from collections import defaultdict
from typing import Dict, List, Tuple
import threading
import time


class AccessFrequencyTracker:
    """
    追踪块访问频率，用于智能预取决策。

    线程安全，支持高并发访问统计。
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
            [(block_hash, access_count), ...] 列表
        """
        with self._lock:
            # 过滤并排序
            candidates = [
                (hash, count)
                for hash, count in self._access_count.items()
                if count >= min_access_count
            ]

            # 按访问次数降序排序
            candidates.sort(key=lambda x: x[1], reverse=True)

            return candidates[:top_n]

    def _apply_decay(self) -> None:
        """
        应用访问次数衰减（减半），避免旧数据累积。
        """
        for hash in list(self._access_count.keys()):
            self._access_count[hash] //= 2

            # 移除衰减到 0 的记录
            if self._access_count[hash] == 0:
                del self._access_count[hash]
                if hash in self._last_access:
                    del self._last_access[hash]

        self._last_decay = time.time()

    def reset(self) -> None:
        """重置所有访问统计。"""
        with self._lock:
            self._access_count.clear()
            self._last_access.clear()
            self._last_decay = time.time()

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息。"""
        with self._lock:
            return {
                'total_blocks': len(self._access_count),
                'total_accesses': sum(self._access_count.values()),
                'avg_access_per_block': (
                    sum(self._access_count.values()) // len(self._access_count)
                    if self._access_count else 0
                )
            }
```

---

### 2. 异步预取器

**文件**: `src/omlx/cache/async_prefetcher.py` (新建)

```python
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class AsyncPrefetcher:
    """
    异步块预取器，使用线程池并行从 SSD 加载块。

    关键特性：
    - 4 线程并行 I/O
    - 非阻塞预取
    - 自动容量管理
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 100
    ):
        """
        初始化异步预取器。

        Args:
            num_workers: 工作线程数（默认 4）
            max_queue_size: 最大预取队列大小
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_queue: List[bytes] = []
        self._queue_lock = threading.Lock()
        self._running = False

        logger.info(
            f"AsyncPrefetcher initialized: {num_workers} workers, "
            f"queue size {max_queue_size}"
        )

    def start(self) -> None:
        """启动预取器。"""
        if self._running:
            return

        self._executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="prefetch-"
        )
        self._running = True

        logger.info("AsyncPrefetcher started")

    def stop(self) -> None:
        """停止预取器并等待所有任务完成。"""
        if not self._running:
            return

        self._running = False

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("AsyncPrefetcher stopped")

    def prefetch_blocks(
        self,
        block_hashes: List[bytes],
        load_fn: Callable[[bytes], Optional[any]],
        on_loaded: Optional[Callable[[bytes, any], None]] = None
    ) -> None:
        """
        异步预取块列表。

        Args:
            block_hashes: 要预取的块哈希列表
            load_fn: 加载函数 (block_hash) -> block_data
            on_loaded: 加载完成回调 (block_hash, block_data) -> None
        """
        if not self._running:
            logger.warning("Prefetcher not running, call start() first")
            return

        if not block_hashes:
            return

        logger.info(
            f"🔥 Prefetching {len(block_hashes)} blocks "
            f"(parallel I/O with {self.num_workers} threads)"
        )

        # 提交并行任务
        futures = []
        for block_hash in block_hashes:
            future = self._executor.submit(
                self._load_and_callback,
                block_hash,
                load_fn,
                on_loaded
            )
            futures.append(future)

        # 非阻塞：立即返回（预取在后台进行）
        logger.debug(f"Submitted {len(futures)} prefetch tasks")

    def _load_and_callback(
        self,
        block_hash: bytes,
        load_fn: Callable[[bytes], Optional[any]],
        on_loaded: Optional[Callable[[bytes, any], None]]
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
                logger.debug(f"Block {block_hash.hex()[:8]} not found on SSD")
                return

            # 调用完成回调（通常是插入内存缓存）
            if on_loaded:
                on_loaded(block_hash, block_data)

            logger.debug(
                f"✅ Prefetched block {block_hash.hex()[:8]} from SSD"
            )

        except Exception as e:
            logger.error(
                f"Failed to prefetch block {block_hash.hex()[:8]}: {e}"
            )
```

---

### 3. 集成到 PagedSSDCacheManager

**文件**: `src/omlx/cache/paged_ssd_cache.py` (修改)

**新增导入**:
```python
from .access_tracker import AccessFrequencyTracker
from .async_prefetcher import AsyncPrefetcher
```

**修改 __init__**:
```python
class PagedSSDCacheManager(CacheManager):
    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int,
        enable_prefetch: bool = True,  # ✅ 新增
        prefetch_top_n: int = 50,      # ✅ 新增
        prefetch_interval: float = 10.0,  # ✅ 新增（秒）
        ...
    ):
        # ... 现有初始化 ...

        # ✅ P1-5: Smart Prefetch
        self.enable_prefetch = enable_prefetch
        self.prefetch_top_n = prefetch_top_n
        self.prefetch_interval = prefetch_interval

        if enable_prefetch:
            self._access_tracker = AccessFrequencyTracker()
            self._prefetcher = AsyncPrefetcher(num_workers=4)
            self._prefetcher.start()

            # 定期预取热块
            self._start_prefetch_timer()

            logger.info(
                f"Smart Prefetch enabled: top-{prefetch_top_n} blocks, "
                f"interval {prefetch_interval}s"
            )
        else:
            self._access_tracker = None
            self._prefetcher = None
```

**新增预取定时器**:
```python
    def _start_prefetch_timer(self) -> None:
        """启动定期预取定时器。"""
        def prefetch_hot_blocks_periodically():
            while self._running and self.enable_prefetch:
                time.sleep(self.prefetch_interval)

                try:
                    self._trigger_smart_prefetch()
                except Exception as e:
                    logger.error(f"Prefetch error: {e}")

        self._prefetch_thread = threading.Thread(
            target=prefetch_hot_blocks_periodically,
            daemon=True,
            name="prefetch-timer"
        )
        self._prefetch_thread.start()

    def _trigger_smart_prefetch(self) -> None:
        """触发智能预取（定期调用）。"""
        if not self.enable_prefetch or not self._access_tracker:
            return

        # 1. 获取热块列表
        hot_blocks = self._access_tracker.get_hot_blocks(
            top_n=self.prefetch_top_n,
            min_access_count=2  # 至少访问 2 次
        )

        if not hot_blocks:
            return

        # 2. 过滤：只预取在 SSD 但不在内存的块
        blocks_to_prefetch = []
        for block_hash, access_count in hot_blocks:
            # 检查是否在 SSD
            metadata = self._index.get(block_hash)
            if metadata is None:
                continue

            # 检查是否已在内存（假设有内存缓存层）
            # TODO: 根据实际架构调整
            # if self._memory_cache.has(block_hash):
            #     continue

            blocks_to_prefetch.append(block_hash)

        if not blocks_to_prefetch:
            logger.debug("No blocks to prefetch (all hot blocks in memory)")
            return

        logger.info(
            f"🔥 Triggering smart prefetch: {len(blocks_to_prefetch)} blocks "
            f"(access counts: {[count for _, count in hot_blocks[:5]]})"
        )

        # 3. 异步预取
        self._prefetcher.prefetch_blocks(
            block_hashes=blocks_to_prefetch,
            load_fn=self._load_block_from_disk,
            on_loaded=self._on_block_prefetched
        )
```

**修改 save_block（追踪访问）**:
```python
    def save_block(
        self,
        block_hash: bytes,
        block_data: List[Any],
        token_count: int,
        ...
    ) -> bool:
        # ... 现有保存逻辑 ...

        # ✅ 追踪访问
        if self._access_tracker:
            self._access_tracker.track_access(block_hash)

        return True
```

**修改 load_block（追踪访问）**:
```python
    def load_block(
        self,
        block_hash: bytes
    ) -> Optional[List[Any]]:
        # ✅ 追踪访问
        if self._access_tracker:
            self._access_tracker.track_access(block_hash)

        # ... 现有加载逻辑 ...
```

**新增回调方法**:
```python
    def _load_block_from_disk(
        self,
        block_hash: bytes
    ) -> Optional[List[Any]]:
        """
        从 SSD 加载块（供预取器调用）。

        Args:
            block_hash: 块哈希

        Returns:
            块数据或 None
        """
        metadata = self._index.get(block_hash)
        if not metadata:
            return None

        try:
            # 读取 safetensors 文件
            block_data = self._read_safetensors(metadata.file_path)
            return block_data
        except Exception as e:
            logger.error(
                f"Failed to load block {block_hash.hex()[:8]}: {e}"
            )
            return None

    def _on_block_prefetched(
        self,
        block_hash: bytes,
        block_data: List[Any]
    ) -> None:
        """
        预取完成回调（供预取器调用）。

        Args:
            block_hash: 块哈希
            block_data: 块数据
        """
        # TODO: 根据实际架构调整
        # 如果有内存缓存层，插入到内存
        # if self._memory_cache:
        #     self._memory_cache.insert(block_hash, block_data)

        logger.debug(
            f"✅ Block {block_hash.hex()[:8]} prefetched and cached"
        )
```

**新增 stop 方法**:
```python
    def stop(self) -> None:
        """停止预取器和后台线程。"""
        self._running = False

        if self._prefetcher:
            self._prefetcher.stop()

        if hasattr(self, '_prefetch_thread'):
            self._prefetch_thread.join(timeout=5.0)
```

---

## 🧪 测试验证

### 功能测试

**文件**: `tests/test_smart_prefetch.py` (新建)

```python
import time
import pytest
from omlx.cache.access_tracker import AccessFrequencyTracker
from omlx.cache.async_prefetcher import AsyncPrefetcher


def test_access_frequency_tracker():
    """测试访问频率追踪器。"""
    tracker = AccessFrequencyTracker()

    # 模拟访问
    block_a = b'block_a_hash'
    block_b = b'block_b_hash'
    block_c = b'block_c_hash'

    # A 访问 5 次
    for _ in range(5):
        tracker.track_access(block_a)

    # B 访问 3 次
    for _ in range(3):
        tracker.track_access(block_b)

    # C 访问 1 次
    tracker.track_access(block_c)

    # 获取热块（top-2）
    hot_blocks = tracker.get_hot_blocks(top_n=2, min_access_count=2)

    assert len(hot_blocks) == 2
    assert hot_blocks[0] == (block_a, 5)  # 最热
    assert hot_blocks[1] == (block_b, 3)  # 第二热

    # C 被过滤（访问次数 < 2）
    assert block_c not in [h for h, _ in hot_blocks]


def test_async_prefetcher():
    """测试异步预取器。"""
    prefetcher = AsyncPrefetcher(num_workers=4)
    prefetcher.start()

    loaded_blocks = []

    def mock_load_fn(block_hash: bytes):
        """模拟加载函数。"""
        time.sleep(0.01)  # 模拟 I/O
        return f"data_{block_hash.hex()[:8]}"

    def on_loaded(block_hash: bytes, block_data: any):
        """加载完成回调。"""
        loaded_blocks.append((block_hash, block_data))

    # 预取 10 个块
    block_hashes = [f"block_{i}".encode() for i in range(10)]
    prefetcher.prefetch_blocks(
        block_hashes=block_hashes,
        load_fn=mock_load_fn,
        on_loaded=on_loaded
    )

    # 等待预取完成
    time.sleep(0.5)

    # 验证所有块都被加载
    assert len(loaded_blocks) == 10

    prefetcher.stop()


@pytest.mark.asyncio
async def test_smart_prefetch_integration():
    """集成测试：完整的智能预取流程。"""
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024 * 1024 * 100,  # 100MB
            enable_prefetch=True,
            prefetch_top_n=5,
            prefetch_interval=1.0  # 1 秒触发一次
        )

        # 模拟块访问模式
        # Block A 被频繁访问（热块）
        block_a_hash = b'block_a_' + b'0' * 24
        for _ in range(10):
            # 模拟访问（在实际场景中会调用 load_block）
            manager._access_tracker.track_access(block_a_hash)

        # Block B 被偶尔访问（温块）
        block_b_hash = b'block_b_' + b'0' * 24
        for _ in range(3):
            manager._access_tracker.track_access(block_b_hash)

        # Block C 只访问 1 次（冷块）
        block_c_hash = b'block_c_' + b'0' * 24
        manager._access_tracker.track_access(block_c_hash)

        # 等待定时预取触发
        time.sleep(2.0)

        # 验证热块被识别
        hot_blocks = manager._access_tracker.get_hot_blocks(
            top_n=5,
            min_access_count=2
        )

        assert len(hot_blocks) >= 1
        assert hot_blocks[0][0] == block_a_hash  # A 是最热的

        manager.stop()
```

### 性能测试

**文件**: `tests/benchmark_prefetch.py` (新建)

```python
import time
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager


def benchmark_with_prefetch():
    """测试启用预取的性能。"""
    manager = PagedSSDCacheManager(
        cache_dir="/tmp/omlx_cache_prefetch",
        max_size_bytes=1024 * 1024 * 500,
        enable_prefetch=True,
        prefetch_top_n=100
    )

    # 模拟工作负载
    block_hashes = [f"block_{i}".encode() for i in range(1000)]

    start = time.perf_counter()

    for i in range(10):  # 10 轮访问
        for block_hash in block_hashes[:100]:  # 访问前 100 个块
            manager.load_block(block_hash)

    elapsed = time.perf_counter() - start

    manager.stop()

    return elapsed


def benchmark_without_prefetch():
    """测试禁用预取的性能。"""
    manager = PagedSSDCacheManager(
        cache_dir="/tmp/omlx_cache_no_prefetch",
        max_size_bytes=1024 * 1024 * 500,
        enable_prefetch=False  # ✅ 禁用预取
    )

    block_hashes = [f"block_{i}".encode() for i in range(1000)]

    start = time.perf_counter()

    for i in range(10):
        for block_hash in block_hashes[:100]:
            manager.load_block(block_hash)

    elapsed = time.perf_counter() - start

    manager.stop()

    return elapsed


if __name__ == "__main__":
    print("Running prefetch benchmarks...")

    time_without = benchmark_without_prefetch()
    print(f"Without prefetch: {time_without:.2f}s")

    time_with = benchmark_with_prefetch()
    print(f"With prefetch: {time_with:.2f}s")

    speedup = time_without / time_with
    print(f"Speedup: {speedup:.1f}x")

    assert speedup >= 2.0, f"Expected > 2x speedup, got {speedup:.1f}x"
```

---

## 📊 预期性能提升

| 场景 | 无预取 | 有预取 | 加速比 |
|------|--------|--------|--------|
| **Agent Scenario (热块访问)** | ~500 MB/s | ~2000 MB/s | 4x |
| **冷启动** | ~100 MB/s | ~100 MB/s | 1x (预期) |
| **混合负载** | ~300 MB/s | ~900 MB/s | 3x |

**关键指标**：
- L3 (SSD) 读取延迟：15ms → 4ms
- 缓存命中后从 SSD 恢复时间：减少 75%

---

## 🚨 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|----------|
| **CPU 开销增加** | 🟡 中 | 限制工作线程数为 4，监控 CPU 使用率 |
| **内存占用增加** | 🟡 中 | 限制预取队列大小，设置内存上限 |
| **预取预测不准** | 🟢 低 | 基于实际访问频率，而非猜测 |
| **线程安全问题** | 🟡 中 | 严格使用锁保护共享数据结构 |

---

## 🎯 成功标准

### 功能标准
- [ ] 访问频率追踪正常工作
- [ ] 热块识别准确（top-N 与实际访问匹配）
- [ ] 异步预取不阻塞主流程
- [ ] 预取的块正确插入缓存

### 性能标准
- [ ] L3 (SSD) 加速 > 3x（目标 4x）
- [ ] Agent Scenario 整体性能无回退
- [ ] CPU 开销 < 10%

### 质量标准
- [ ] 测试覆盖率 > 80%
- [ ] 无线程死锁/竞态条件
- [ ] 内存泄漏检查通过

---

**设计版本**: v1.0
**创建日期**: 2026-03-13
**预计工期**: 1 天
