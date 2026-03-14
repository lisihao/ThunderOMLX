# P2-9: 块级 LRU 优化设计

> **目标**: 提升缓存命中率，通过 LFU-LRU 混合策略优化块驱逐决策

---

## 问题分析

### 当前 LRU 实现 (PagedSSDCacheIndex)

```python
class PagedSSDCacheIndex:
    def __init__(self, max_size_bytes: int):
        self._index: Dict[bytes, PagedSSDBlockMetadata] = {}
        self._lru: OrderedDict[bytes, float] = OrderedDict()  # block_hash → last_access
        self._total_size: int = 0
        self._max_size: int = max_size_bytes
        self._lock = threading.RLock()

    def touch(self, block_hash: bytes) -> None:
        """Update last access time (move to end of LRU)."""
        self._index[block_hash].touch()
        self._lru.move_to_end(block_hash)
        self._lru[block_hash] = self._index[block_hash].last_access

    def evict_until_size(self, target_size: int) -> List[PagedSSDBlockMetadata]:
        """Evict LRU entries until total size is below target."""
        evicted = []
        while self._total_size > target_size and self._lru:
            block_hash = next(iter(self._lru))  # ❌ 纯 LRU，不考虑访问频率
            metadata = self.remove(block_hash)
            if metadata:
                evicted.append(metadata)
        return evicted
```

### 问题

| 问题 | 影响 | 示例 |
|------|------|------|
| **只看 last_access** | 高频但暂时未访问的 blocks 被驱逐 | System prompt block 访问 100 次，但最近 1 分钟未访问 → 被驱逐 |
| **未集成 AccessFrequencyTracker** | P1-5 的访问频率数据未使用 | AccessFrequencyTracker 已经追踪访问次数，但 LRU 未使用 |
| **所有 blocks 平等对待** | 重要 blocks 和普通 blocks 同等对待 | System prompt blocks vs random query blocks 驱逐优先级相同 |

---

## 优化方案: LFU-LRU 混合策略

### 核心思想

**驱逐优先级 = f(访问频率, 最近访问时间)**

- 高频 + 近期访问 → **最低驱逐优先级** (保留)
- 低频 + 远期访问 → **最高驱逐优先级** (优先驱逐)
- 中等情况 → 混合评分

### 评分公式

```
eviction_score = (1 - normalized_frequency) * 0.6 + normalized_recency * 0.4

其中:
- normalized_frequency = access_count / max_access_count ∈ [0, 1]
  - 0 = 从未访问 (容易驱逐)
  - 1 = 访问最频繁 (难驱逐)

- normalized_recency = (current_time - last_access) / max_age ∈ [0, 1]
  - 0 = 刚访问 (难驱逐)
  - 1 = 很久未访问 (容易驱逐)

- 权重: frequency 60%, recency 40%
  - 偏向频率（长期模式）而非时间（短期波动）

eviction_score ∈ [0, 1]
- 0 = 保留 (高频 + 近期)
- 1 = 优先驱逐 (低频 + 远期)
```

### 驱逐策略

```python
def evict_until_size(self, target_size: int) -> List[PagedSSDBlockMetadata]:
    """Evict blocks using LFU-LRU hybrid strategy."""
    evicted = []

    # Step 1: 计算每个 block 的驱逐分数
    block_scores = []
    current_time = time.time()
    max_access_count = max((self._access_tracker.get_access_count(h) for h in self._index.keys()), default=1)
    max_age = max((current_time - meta.last_access for meta in self._index.values()), default=1.0)

    for block_hash, metadata in self._index.items():
        access_count = self._access_tracker.get_access_count(block_hash)
        age = current_time - metadata.last_access

        normalized_freq = access_count / max_access_count
        normalized_recency = age / max_age

        eviction_score = (1 - normalized_freq) * 0.6 + normalized_recency * 0.4

        block_scores.append((eviction_score, block_hash, metadata))

    # Step 2: 按驱逐分数排序（分数高的优先驱逐）
    block_scores.sort(key=lambda x: x[0], reverse=True)

    # Step 3: 从高分数开始驱逐，直到满足 target_size
    for eviction_score, block_hash, metadata in block_scores:
        if self._total_size <= target_size:
            break

        # 驱逐这个 block
        self.remove(block_hash)
        evicted.append(metadata)

        logger.debug(
            f"Evicted block {block_hash.hex()[:8]} "
            f"(score={eviction_score:.3f}, "
            f"freq={self._access_tracker.get_access_count(block_hash)}, "
            f"age={current_time - metadata.last_access:.1f}s)"
        )

    return evicted
```

---

## 集成 AccessFrequencyTracker (P1-5)

### 修改点

**1. PagedSSDCacheIndex 初始化**

```python
from .access_tracker import AccessFrequencyTracker

class PagedSSDCacheIndex:
    def __init__(self, max_size_bytes: int, access_tracker: Optional[AccessFrequencyTracker] = None):
        self._index: Dict[bytes, PagedSSDBlockMetadata] = {}
        self._lru: OrderedDict[bytes, float] = OrderedDict()
        self._total_size: int = 0
        self._max_size: int = max_size_bytes
        self._lock = threading.RLock()

        # ✅ 新增: 访问频率追踪器
        self._access_tracker = access_tracker or AccessFrequencyTracker(decay_interval=3600.0)
```

**2. touch() 方法更新访问频率**

```python
def touch(self, block_hash: bytes) -> None:
    """Update last access time and access frequency."""
    with self._lock:
        if block_hash in self._index:
            # 更新时间
            self._index[block_hash].touch()
            self._lru.move_to_end(block_hash)
            self._lru[block_hash] = self._index[block_hash].last_access

            # ✅ 新增: 更新访问频率
            self._access_tracker.track_access(block_hash)
```

**3. PagedSSDCacheManager 传递 tracker**

```python
class PagedSSDCacheManager:
    def __init__(
        self,
        cache_dir: Path,
        max_size_bytes: int,
        enable_prefetch: bool = False,
        enable_checksum: bool = False,
        hot_cache_max_bytes: int = 0,
        checksum_verify_on_load: bool = False,
    ):
        # ... 现有初始化 ...

        # ✅ 创建 AccessFrequencyTracker
        from .access_tracker import AccessFrequencyTracker
        self._access_tracker = AccessFrequencyTracker(decay_interval=3600.0)

        # ✅ 传递给 Index
        self._index = PagedSSDCacheIndex(
            max_size_bytes=max_size_bytes,
            access_tracker=self._access_tracker  # 传递 tracker
        )
```

---

## 配置参数

```python
# src/omlx/cache/paged_ssd_cache.py

class PagedSSDCacheIndex:
    # LFU-LRU 混合策略权重
    FREQUENCY_WEIGHT: float = 0.6  # 访问频率权重 (60%)
    RECENCY_WEIGHT: float = 0.4    # 最近访问时间权重 (40%)

    # 最小访问次数阈值（低于此值的 blocks 优先驱逐）
    MIN_ACCESS_COUNT_THRESHOLD: int = 2
```

---

## 预期效果

### 性能指标

| 指标 | 当前 (纯 LRU) | 优化后 (LFU-LRU) | 提升 |
|------|---------------|-----------------|------|
| **缓存命中率** | ~85% | ~92% | +7% |
| **热点 blocks 保留率** | ~70% | ~95% | +25% |
| **驱逐准确性** | 基线 | +15% | - |

### 使用场景

**场景 1: Agent 多轮对话**
- System prompt blocks 被访问 100+ 次
- 纯 LRU: 可能因最近 1 分钟未访问被驱逐
- LFU-LRU: 高频 (100 次) → 保留 ✅

**场景 2: 批量请求**
- 同一 prompt 前缀被 10 个请求共享
- 纯 LRU: 第 11 个请求时可能已被驱逐
- LFU-LRU: 高频 (10 次) → 保留 ✅

---

## 实现计划

### Phase 1: 集成 AccessFrequencyTracker (0.5 天)

- [x] P1-5 已实现 AccessFrequencyTracker
- [ ] 修改 PagedSSDCacheIndex 初始化接受 tracker
- [ ] 修改 touch() 方法调用 tracker.track_access()
- [ ] 修改 PagedSSDCacheManager 创建并传递 tracker

### Phase 2: 实现 LFU-LRU 驱逐策略 (1 天)

- [ ] 实现 eviction_score 计算逻辑
- [ ] 修改 evict_until_size() 方法
- [ ] 添加驱逐日志（分数、频率、时间）
- [ ] 添加配置参数（权重、阈值）

### Phase 3: 测试验证 (0.5 天)

- [ ] 单元测试：驱逐优先级正确性
- [ ] 集成测试：缓存命中率提升
- [ ] 性能测试：驱逐开销 < 5ms

**总计**: 2 天

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **驱逐开销增加** | 中 | 中 | 驱逐时排序 O(N log N)，但 N 通常 < 1000，开销 < 5ms |
| **AccessFrequencyTracker 内存开销** | 低 | 低 | Tracker 已在 P1-5 实现，decay 机制自动清理旧数据 |
| **权重参数调优** | 低 | 低 | 默认 60/40 基于 Redis 经验，可配置 |

---

## 验证标准

### 功能验证

- [ ] 高频 blocks 优先保留（访问 10+ 次的 blocks 驱逐率 < 5%）
- [ ] 低频远期 blocks 优先驱逐
- [ ] touch() 正确更新访问频率
- [ ] 驱逐日志包含分数、频率、时间

### 性能验证

- [ ] 缓存命中率提升 > 5% (Agent scenario)
- [ ] 驱逐开销 < 5ms (1000 blocks)
- [ ] 内存开销增加 < 10MB (AccessFrequencyTracker)

---

## 参考

- **Redis LFU 实现**: https://redis.io/docs/manual/eviction/#the-new-lfu-mode
- **LRU-K 算法**: Database Systems - The Complete Book (Section 13.8)
- **P1-5 AccessFrequencyTracker**: `src/omlx/cache/access_tracker.py`

---

*设计版本: v1.0*
*创建日期: 2026-03-13*
*预计完成: 2 天*
