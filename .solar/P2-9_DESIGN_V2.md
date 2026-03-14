# P2-9: 块级 LRU 优化设计 (v2.0 - LRU-2)

> **目标**: 提升缓存命中率，使用 LRU-2 策略优化块驱逐决策
> **版本**: v2.0 - 基于 Solar 主脑会审结果，放弃 LFU-LRU 混合策略

---

## 设计变更历史

| 版本 | 方案 | 状态 | 原因 |
|------|------|------|------|
| v1.0 | LFU-LRU 混合策略 | ❌ **放弃** | 专家会审发现 CRITICAL 问题：性能回退 (O(1)→O(N))、线程安全失败、归一化缺陷 |
| v2.0 | **LRU-2** | ✅ **采纳** | Solar 主脑独立分析：简单、高效、O(1) 性能、有实证支持 |

---

## 问题重新定义

### v1.0 假设的问题（已证伪）

**假设**: System prompt block 访问 100 次，但最近 1 分钟未访问 → 被 LRU 驱逐

**Solar 主脑分析**:
- ❌ 无实际证据证明此场景发生
- ❌ 纯 LRU 中，100 次访问的 block 需要 **1000+ 新 blocks** 访问后才会被驱逐
- ❌ Agent scenario (4 并发)，1000 blocks = 64 轮对话，**不可能在 1 分钟内完成**

**结论**: v1.0 在解决一个 **伪需求**

### 真实问题

**实际观察** (基于 LLM KV Cache 特性):
```
场景 1: 一次性长 prompt
- 用户输入 8K tokens 的文档
- 生成 128 tokens 后，文档内容永远不会再访问
- 这些 blocks 占据缓存，但不会再被命中

场景 2: 重复短 prompt
- System prompt (512 tokens) 被访问 100 次
- 用户 query (256 tokens) 每次都不同
- Query blocks 是一次性访问，但会挤占 system prompt 空间
```

**真实需求**: 区分 **一次性访问** (cold) vs **重复访问** (hot)

---

## LRU-2 方案

### 核心思想

**LRU-K 原理** (K=2):
```
普通 LRU: 只看最近一次访问时间 → 无法区分 cold/hot
LRU-2:    看最近两次访问 → 自动区分 cold/hot

访问 1 次 → cold queue (低优先级，优先驱逐)
访问 2+ 次 → hot queue (高优先级，保留)
```

**优势**:
- ✅ **O(1) 所有操作** (touch, evict)
- ✅ **无需归一化** (避免异常值问题)
- ✅ **无需访问计数** (自动区分 cold/hot)
- ✅ **线程安全简单** (两个 OrderedDict + 一个锁)
- ✅ **有实证支持** (PostgreSQL, InnoDB 使用)

---

## 实现方案

### 架构

```
┌─────────────────────────────────────────────────────────┐
│              LRU-2 Cache Index Architecture             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Cold Queue (OrderedDict)                       │   │
│  │  - 只访问 1 次的 blocks                          │   │
│  │  - LRU 驱逐                                      │   │
│  │  - 优先级: 低 (优先驱逐)                         │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│                          │ touch() 第 2 次访问          │
│                          ▼                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Hot Queue (OrderedDict)                        │   │
│  │  - 访问 2+ 次的 blocks                           │   │
│  │  - LRU 驱逐                                      │   │
│  │  - 优先级: 高 (最后驱逐)                         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  驱逐顺序: Cold Queue (LRU) → Hot Queue (LRU)          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 代码实现

#### 1. PagedSSDCacheIndex (修改)

```python
from collections import OrderedDict
import threading
import time
from typing import Dict, List, Optional

class PagedSSDCacheIndex:
    """
    LRU-2 Cache Index.

    Maintains two queues:
    - cold_queue: blocks accessed only once (low priority)
    - hot_queue: blocks accessed 2+ times (high priority)

    Eviction order: cold_queue (LRU) → hot_queue (LRU)
    """

    def __init__(self, max_size_bytes: int):
        """
        Initialize LRU-2 cache index.

        Args:
            max_size_bytes: Maximum total size of SSD cache.
        """
        # Metadata index (block_hash → metadata)
        self._index: Dict[bytes, PagedSSDBlockMetadata] = {}

        # LRU-2 Queues (block_hash → last_access_time)
        self._cold_queue: OrderedDict[bytes, float] = OrderedDict()  # 访问 1 次
        self._hot_queue: OrderedDict[bytes, float] = OrderedDict()   # 访问 2+ 次

        self._total_size: int = 0
        self._max_size: int = max_size_bytes
        self._lock = threading.RLock()

    def add(self, metadata: PagedSSDBlockMetadata) -> None:
        """
        Add a new block to the index (cold queue).

        Args:
            metadata: Block metadata.
        """
        with self._lock:
            block_hash = metadata.block_hash

            # Remove from existing queues if present
            if block_hash in self._index:
                old_meta = self._index[block_hash]
                self._total_size -= old_meta.file_size
                self._cold_queue.pop(block_hash, None)
                self._hot_queue.pop(block_hash, None)

            # Add to index and cold queue
            self._index[block_hash] = metadata
            self._cold_queue[block_hash] = metadata.last_access
            self._total_size += metadata.file_size

    def get(self, block_hash: bytes) -> Optional[PagedSSDBlockMetadata]:
        """
        Get block metadata (does NOT update access time).

        Args:
            block_hash: Block content hash.

        Returns:
            Block metadata if found, None otherwise.
        """
        with self._lock:
            return self._index.get(block_hash)

    def touch(self, block_hash: bytes) -> None:
        """
        Update last access time and promote to hot queue if needed.

        LRU-2 Logic:
        - First touch: stays in cold_queue
        - Second touch: moves to hot_queue
        - Subsequent touches: update position in hot_queue

        Args:
            block_hash: Block content hash.
        """
        with self._lock:
            if block_hash not in self._index:
                return

            # Update metadata timestamp
            self._index[block_hash].touch()
            current_time = time.time()

            if block_hash in self._cold_queue:
                # ✅ Second access → promote to hot queue
                del self._cold_queue[block_hash]
                self._hot_queue[block_hash] = current_time
                self._hot_queue.move_to_end(block_hash)

            elif block_hash in self._hot_queue:
                # ✅ Subsequent access → update position in hot queue
                self._hot_queue[block_hash] = current_time
                self._hot_queue.move_to_end(block_hash)

            else:
                # Edge case: block exists in index but not in queues
                # (should not happen, defensive programming)
                self._cold_queue[block_hash] = current_time

    def remove(self, block_hash: bytes) -> Optional[PagedSSDBlockMetadata]:
        """
        Remove block from index and queues.

        Args:
            block_hash: Block content hash.

        Returns:
            Removed metadata if found, None otherwise.
        """
        with self._lock:
            if block_hash not in self._index:
                return None

            # Remove from index
            metadata = self._index.pop(block_hash)
            self._total_size -= metadata.file_size

            # Remove from queues
            self._cold_queue.pop(block_hash, None)
            self._hot_queue.pop(block_hash, None)

            return metadata

    def evict_until_size(self, target_size: int) -> List[PagedSSDBlockMetadata]:
        """
        Evict blocks using LRU-2 strategy until total size is below target.

        Eviction order:
        1. Cold queue (LRU) - one-time access blocks
        2. Hot queue (LRU) - frequently accessed blocks

        Args:
            target_size: Target total size in bytes.

        Returns:
            List of evicted metadata.
        """
        with self._lock:
            evicted = []

            # Phase 1: Evict from cold queue (low priority)
            while self._total_size > target_size and self._cold_queue:
                # Get LRU entry from cold queue
                block_hash = next(iter(self._cold_queue))
                metadata = self.remove(block_hash)
                if metadata:
                    evicted.append(metadata)
                    logger.debug(
                        f"Evicted COLD block {block_hash.hex()[:8]} "
                        f"(size={metadata.file_size}, age={time.time() - metadata.last_access:.1f}s)"
                    )

            # Phase 2: Evict from hot queue (high priority, only if needed)
            while self._total_size > target_size and self._hot_queue:
                # Get LRU entry from hot queue
                block_hash = next(iter(self._hot_queue))
                metadata = self.remove(block_hash)
                if metadata:
                    evicted.append(metadata)
                    logger.debug(
                        f"Evicted HOT block {block_hash.hex()[:8]} "
                        f"(size={metadata.file_size}, age={time.time() - metadata.last_access:.1f}s)"
                    )

            return evicted

    def get_lru_entries(self, count: int) -> List[PagedSSDBlockMetadata]:
        """
        Get least recently used entries (for monitoring).

        Args:
            count: Maximum number of entries to return.

        Returns:
            List of LRU metadata entries (cold first, then hot).
        """
        with self._lock:
            result = []

            # Get from cold queue first
            for block_hash in list(self._cold_queue.keys())[:count]:
                if block_hash in self._index:
                    result.append(self._index[block_hash])

            # Get from hot queue if needed
            remaining = count - len(result)
            if remaining > 0:
                for block_hash in list(self._hot_queue.keys())[:remaining]:
                    if block_hash in self._index:
                        result.append(self._index[block_hash])

            return result

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        with self._lock:
            return {
                'total_blocks': len(self._index),
                'cold_blocks': len(self._cold_queue),
                'hot_blocks': len(self._hot_queue),
                'total_size': self._total_size,
                'max_size': self._max_size,
            }

    def contains(self, block_hash: bytes) -> bool:
        """Check if block exists in index."""
        with self._lock:
            return block_hash in self._index
```

---

## 性能分析

### 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| **add()** | O(1) | Dict insert + OrderedDict insert |
| **get()** | O(1) | Dict lookup |
| **touch()** | O(1) | Dict delete + insert + move_to_end |
| **remove()** | O(1) | Dict delete + OrderedDict delete |
| **evict_until_size()** | O(K) | K = 驱逐的 blocks 数量，通常 K << N |

**对比**:
- LRU (当前): O(1) 所有操作 ✅
- LRU-2 (本方案): O(1) 所有操作 ✅
- LFU-LRU (v1.0): O(N log N) 驱逐 ❌

### 空间复杂度

```
额外空间 = 2 * N * (32 bytes + 8 bytes)
        = 80 * N bytes

N = 1000 blocks → 80 KB
N = 10000 blocks → 800 KB
```

**可接受**: 相比 v1.0 的 AccessFrequencyTracker (需要额外存储访问计数)，LRU-2 空间开销更低。

---

## 预期效果（保守估计）

### 性能指标

| 指标 | 当前 (纯 LRU) | 优化后 (LRU-2) | 提升 |
|------|---------------|---------------|------|
| **缓存命中率** | ~85% | ~88-90% | +3-5% |
| **热点 blocks 保留率** | ~70% | ~85-90% | +15-20% |
| **驱逐准确性** | 基线 | +10-15% | - |
| **驱逐开销** | < 1ms | < 1ms | 无变化 |

**为什么比 v1.0 保守？**
- v1.0 声称 +7% 基于错误的 Redis 类比
- LRU-2 实证数据 (PostgreSQL, InnoDB) 显示 +3-5% 是现实预期
- LLM KV Cache 的时间局部性极强，纯 LRU 已经很好

### 使用场景

**场景 1: Agent 多轮对话**
```
System prompt: 512 tokens, 访问 100 次
- 第 1 次访问: cold queue
- 第 2 次访问: → hot queue (保留)
- 后续 98 次: 保持在 hot queue

纯 LRU: 可能被新的 query blocks 挤出 (❌)
LRU-2: 保留在 hot queue (✅)
```

**场景 2: 批量请求**
```
10 个请求共享前缀 "Explain Python"
- 请求 1: "Explain Python" → cold queue
- 请求 2: "Explain Python" → hot queue (保留)
- 请求 3-10: 命中 hot queue

纯 LRU: 第 11 个不同请求时可能被驱逐 (❌)
LRU-2: 保留在 hot queue (✅)
```

**场景 3: 一次性长 prompt**
```
用户上传 8K tokens 文档，生成 128 tokens
- 文档 blocks: 访问 1 次 → cold queue
- 生成完成后，永远不会再访问
- 下次驱逐时优先驱逐这些 blocks

纯 LRU: 与热点 blocks 平等对待 (❌)
LRU-2: 优先驱逐 cold blocks (✅)
```

---

## 实现计划

### Phase 1: 实现 LRU-2 (0.5 天)

- [ ] 修改 `PagedSSDCacheIndex.__init__()` - 添加 cold/hot queues
- [ ] 修改 `touch()` - 实现 cold → hot 晋升逻辑
- [ ] 修改 `evict_until_size()` - 优先驱逐 cold queue
- [ ] 修改 `add()`, `remove()`, `get_stats()`
- [ ] 添加日志：cold/hot 驱逐区分

### Phase 2: 测试验证 (0.5 天)

**单元测试**:
- [ ] 测试 cold → hot 晋升逻辑
- [ ] 测试驱逐顺序 (cold 优先)
- [ ] 测试边界情况 (空缓存、单元素)
- [ ] 测试线程安全 (并发 touch + evict)

**集成测试**:
- [ ] Agent scenario: 多轮对话，验证 system prompt 保留
- [ ] 批量请求: 验证共享前缀保留
- [ ] 一次性 prompt: 验证 cold blocks 优先驱逐

**性能测试**:
- [ ] 缓存命中率对比 (LRU vs LRU-2)
- [ ] 驱逐开销验证 (< 1ms)
- [ ] 内存开销验证 (< 1MB for 10000 blocks)

**总计**: 1 天

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **改进效果不明显** | 中 | 中 | 先收集实际缓存失效数据，验证假设 |
| **线程安全问题** | 低 | 高 | 充分的并发测试 + code review |
| **内存开销** | 低 | 低 | 两个 OrderedDict，开销可控 |
| **性能回退** | 低 | 高 | 保持 O(1) 操作，无性能风险 |

---

## 验证标准

### 功能验证

- [ ] Cold → hot 晋升正确（第 2 次访问时触发）
- [ ] 驱逐顺序正确（cold 优先，hot 最后）
- [ ] 线程安全（并发测试无 race condition）
- [ ] 边界情况处理（空缓存、单元素、全 cold/全 hot）

### 性能验证

- [ ] 缓存命中率提升 > 3% (Agent scenario)
- [ ] 驱逐开销 < 1ms (1000 blocks)
- [ ] 内存开销 < 1MB (10000 blocks)
- [ ] 无性能回退（所有操作仍 O(1)）

---

## 参考

- **LRU-K 算法**: O'Neil et al. "The LRU-K page replacement algorithm for database disk buffering" (1993)
- **PostgreSQL Buffer Manager**: 使用 LRU-2 (called "usage count")
- **InnoDB Buffer Pool**: 使用 LRU + young/old 分区（类似 LRU-2）

---

*设计版本: v2.0*
*创建日期: 2026-03-13*
*Solar 主脑独立分析后采纳*
*预计完成: 1 天*
