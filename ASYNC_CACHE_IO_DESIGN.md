# 异步 Cache I/O 设计方案

**日期**: 2026-03-16 00:30
**预期收益**: 0.8-1.5s（通过 I/O 与计算重叠）

---

## 问题分析

### 当前瓶颈

从 Profiling 数据看，Cache I/O 时间：
- 文件读取：~0.5s
- 解压缩：~0.5s
- mx.load()：~0.4s
- **总计：~1.4s**

### 关键约束

**Metal GPU 限制**：
- `mx.load()` 必须在主线程（推理线程）执行
- 在 worker thread 中调用会导致 Metal 资源竞争 → 死锁
- 之前尝试过异步加载但失败了（见代码注释）

### 优化机会

**I/O 与计算可重叠**：
```
当前（串行）：
  GPU 计算 chunk N → 等待 I/O → 加载 block N+1 → GPU 计算 chunk N+1

优化后（重叠）：
  GPU 计算 chunk N
      ↓（同时）
  后台预取 block N+1（I/O + 解压）
      ↓
  主线程快速 mx.load()（数据已就绪）→ GPU 计算 chunk N+1
```

**预期收益**：
- 如果 I/O 时间 < GPU 计算时间：完全隐藏 I/O
- 如果 I/O 时间 > GPU 计算时间：部分隐藏
- 保守估计：**节省 60-80% 的 I/O 时间 = ~1s**

---

## 设计方案

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    主线程（推理线程）                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  load_block(hash) →                                         │
│    1. Check prefetch_cache ──────┐                          │
│       ├─ Hit: decompressed_data  │                          │
│       │   → mx.load() (快！)     │                          │
│       │                           │                          │
│       └─ Miss: fallback to sync  │                          │
│           → read + decompress    │                          │
│           → mx.load()             │                          │
│                                   │                          │
└───────────────────────────────────┼──────────────────────────┘
                                    │
                         prefetch_cache (LRU, max 3-5 items)
                                    │
┌───────────────────────────────────┼──────────────────────────┐
│              后台预取线程（I/O 线程）                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  while True:                                                │
│    next_hash = predict_next_block()                         │
│    if next_hash in prefetch_cache:                          │
│        continue  # 已预取                                   │
│                                                             │
│    file_data = read_file(next_hash)      # SSD I/O         │
│    decompressed = decompress(file_data)  # CPU 密集        │
│                                                             │
│    prefetch_cache.put(next_hash, decompressed)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. 预取缓存（PrefetchCache）

```python
class PrefetchCache:
    """LRU 缓存，存储已预取的解压数据（纯字节）"""

    def __init__(self, max_size: int = 5):
        self._cache: OrderedDict[bytes, bytes] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, block_hash: bytes) -> Optional[bytes]:
        """获取预取的解压数据（线程安全）"""
        with self._lock:
            if block_hash in self._cache:
                self._cache.move_to_end(block_hash)
                return self._cache[block_hash]
        return None

    def put(self, block_hash: bytes, decompressed_data: bytes) -> None:
        """存储预取的解压数据（线程安全，LRU 淘汰）"""
        with self._lock:
            if block_hash in self._cache:
                self._cache.move_to_end(block_hash)
            else:
                self._cache[block_hash] = decompressed_data
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)  # 淘汰最老的
```

---

### 2. 预取预测器（PrefetchPredictor）

```python
class PrefetchPredictor:
    """预测下一个需要的 block（基于访问模式）"""

    def __init__(self):
        self._last_hash: Optional[bytes] = None
        self._access_pattern: List[bytes] = []

    def record_access(self, block_hash: bytes) -> None:
        """记录访问模式"""
        self._last_hash = block_hash
        self._access_pattern.append(block_hash)
        if len(self._access_pattern) > 100:
            self._access_pattern.pop(0)

    def predict_next(self) -> Optional[bytes]:
        """预测下一个 block（简单策略：假设顺序访问）"""
        # 简单策略：返回 None（让调用者决定）
        # 高级策略：基于历史模式预测（后续优化）
        return None
```

**调用方式**：
- 主线程调用 `load_block(hash)` 后，立即调用 `prefetch_next(hash_of_next_block)`
- 预测器记录模式，预取线程执行实际 I/O

---

### 3. 预取工作线程（PrefetchWorker）

```python
class PrefetchWorker:
    """后台线程，预取下一个 block 的文件数据"""

    def __init__(self, cache_manager: 'PagedSSDCacheManager'):
        self._manager = cache_manager
        self._prefetch_queue: queue.Queue[bytes] = queue.Queue(maxsize=10)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="cache-prefetch-worker",
            daemon=True
        )
        self._thread.start()

    def prefetch(self, block_hash: bytes) -> None:
        """请求预取 block（非阻塞）"""
        try:
            self._prefetch_queue.put_nowait(block_hash)
        except queue.Full:
            pass  # 队列满，丢弃（不阻塞主线程）

    def _worker_loop(self) -> None:
        """后台线程主循环"""
        while not self._shutdown.is_set():
            try:
                block_hash = self._prefetch_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 检查是否已在预取缓存中
            if self._manager._prefetch_cache.get(block_hash) is not None:
                continue

            # 检查是否在索引中
            metadata = self._manager._index.get(block_hash)
            if metadata is None:
                continue

            file_path = metadata.file_path
            if not file_path.exists():
                continue

            try:
                # 读取文件（I/O）
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                # 解压缩（如果需要）
                if file_path.suffix in ('.lz4', '.zst'):
                    decompressed = self._decompress(file_data, file_path.suffix)
                else:
                    decompressed = file_data

                # 存入预取缓存
                self._manager._prefetch_cache.put(block_hash, decompressed)

            except Exception as e:
                logger.debug(f"Prefetch failed for {block_hash.hex()[:16]}: {e}")

    def _decompress(self, data: bytes, suffix: str) -> bytes:
        """解压缩数据"""
        if suffix == '.lz4':
            import lz4.frame
            return lz4.frame.decompress(data)
        elif suffix == '.zst':
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        return data

    def shutdown(self) -> None:
        """关闭预取线程"""
        self._shutdown.set()
        self._thread.join(timeout=1.0)
```

---

### 4. 修改 load_block（主线程）

```python
def load_block(self, block_hash: bytes) -> Optional[List[Any]]:
    """加载 block（带预取加速）"""

    # ... (hot cache 检查，同之前) ...

    # 检查预取缓存
    prefetched_data = self._prefetch_cache.get(block_hash)
    if prefetched_data is not None:
        # 预取命中！直接 mx.load()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as tmp:
            tmp.write(prefetched_data)
            tmp_path = tmp.name

        try:
            tensors = mx.load(tmp_path)
            # ... (reconstruct cache data, 同之前) ...
            self._stats["prefetch_hits"] += 1
            return cache_data
        finally:
            os.unlink(tmp_path)

    # Fallback：同步读取（同之前的逻辑）
    # ...
```

---

## 调用时机

### 何时触发预取？

**选项 1：在 Scheduler 中预测** ⭐ **推荐**

```python
# scheduler.py
def _prefill_chunk(self, chunk_id: int):
    # 计算当前 chunk
    output = model.forward(chunk)

    # 预取下一个 chunk 的 KV cache blocks
    if chunk_id < total_chunks - 1:
        next_chunk_hashes = self._get_chunk_block_hashes(chunk_id + 1)
        for block_hash in next_chunk_hashes:
            cache_manager.prefetch_block(block_hash)
```

**选项 2：在 load_block 中记录模式**

```python
# paged_ssd_cache.py
def load_block(self, block_hash: bytes):
    result = ... # 加载逻辑

    # 记录访问模式，预测下一个
    self._prefetch_predictor.record_access(block_hash)
    next_hash = self._prefetch_predictor.predict_next()
    if next_hash:
        self._prefetch_worker.prefetch(next_hash)

    return result
```

**推荐**：选项 1（Scheduler 预测）更精确，因为 Scheduler 知道 chunk 边界。

---

## 性能分析

### 预期收益

**场景 1：I/O 完全隐藏**（理想情况）

```
GPU 计算 chunk: 200ms
SSD I/O + 解压: 100ms

当前：200ms（计算）+ 100ms（I/O）= 300ms
优化后：max(200ms, 100ms) = 200ms
节省：100ms
```

**场景 2：I/O 部分隐藏**（现实情况）

```
GPU 计算 chunk: 200ms
SSD I/O + 解压: 150ms

当前：200ms + 150ms = 350ms
优化后：200ms + 50ms = 250ms
节省：100ms
```

**总收益**（8192 tokens, ~16 chunks）：
- 每 chunk 节省 50-100ms
- 总计节省：**0.8-1.6s**

---

### 内存开销

**预取缓存大小**：
- 每个 block ~10MB（压缩后）
- 最多缓存 5 个 blocks
- **总内存：~50MB**（可接受）

---

### 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 线程安全问题 | 低 | 高 | 使用锁保护共享数据结构 |
| 预取错误 block | 中 | 低 | 不影响正确性，只是浪费 I/O |
| 内存占用过高 | 低 | 中 | LRU 淘汰，限制缓存大小 |
| 预测失败率高 | 中 | 中 | 保留 fallback，不影响正确性 |

---

## 实施计划

### Phase 1: 基础设施（30 分钟）

1. ✅ 实现 `PrefetchCache`
2. ✅ 实现 `PrefetchWorker`
3. ✅ 修改 `load_block` 检查预取缓存

### Phase 2: 集成预测（15 分钟）

1. ✅ 在 Scheduler 中添加预取调用
2. ✅ 传递下一个 chunk 的 block hashes

### Phase 3: 测试验证（15 分钟）

1. ✅ 功能测试：确保正确性
2. ✅ 性能测试：测量收益
3. ✅ 压力测试：并发安全

---

## 成功标准

- ✅ Prefill TPS 提升 > 10%（~70 tok/s）
- ✅ 预取命中率 > 80%
- ✅ 无死锁或崩溃
- ✅ 内存占用 < 100MB

---

*设计者: Solar (战略家模式)*
*审核: 稳健派 (Gemini 2.5 Pro) 建议*
