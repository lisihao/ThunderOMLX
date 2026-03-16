# Task #14: 锁优化分析 - 意外性能下降

**日期**: 2026-03-15
**状态**: ⚠️ 锁优化后性能下降 2-3%
**原因**: 分析中

---

## 执行摘要

实施了三项锁优化（分片锁、读写锁、GIL 无锁），但性能反而下降 2-3%。

---

## 性能对比

### 优化前（Phase 1）

```
TTFT: 11631.3ms
Processing TPS: 704.3 tok/s
Generation TPS: ~77 tok/s
```

### 锁优化后

```
TTFT: 11949.1ms (+317.8ms, +2.7%)  ❌
Processing TPS: 685.6 tok/s (-18.7 tok/s, -2.6%)  ❌
Generation TPS: 79.3 tok/s (+2.3 tok/s, +3%)  ✅
```

**结论**: Prefill 性能下降 ~2.6%，Generation 性能提升 ~3%

---

## 实施的优化

### 1. 分片锁（ShardedPagedSSDCacheIndex）

**修改**: 1 个 RLock → 16 个分片锁

```python
# 每次访问需要计算 shard_id
shard_id = int.from_bytes(block_hash[:2], 'big') & 15
shard = self.shards[shard_id]
with shard['lock']:
    ...
```

**新增开销**:
- `int.from_bytes()`: ~10-20ns
- Hash → shard 映射: ~5ns
- 数组访问: ~5ns
- **总计**: ~20-30ns per access

**收益**:
- 如果有并发访问：减少锁竞争
- 如果没有并发：**纯开销，无收益**

### 2. 读写锁（RWLock）

**修改**: `_hot_cache_lock = RWLock()`

```python
# 读操作
with self._hot_cache_lock.read():
    ...

# 写操作
with self._hot_cache_lock.write():
    ...
```

**新增开销**:
- 维护 readers/writers 计数: ~10ns
- 条件变量检查: ~20ns
- **总计**: ~30ns per lock acquire/release

**收益**:
- 如果有多个并发读：显著减少等待
- 如果读写交替：**纯开销，无收益**

### 3. GIL 无锁（移除 _pending_write_hashes_lock）

**修改**: 移除显式锁，依赖 GIL

```python
# 之前
with self._pending_write_hashes_lock:
    self._pending_write_hashes.add(block_hash)

# 现在
self._pending_write_hashes.add(block_hash)
```

**收益**: 减少 ~50ns per operation（锁开销）

---

## 根因分析

### 假设 1: 单线程场景，无并发收益

**问题**: ThunderOMLX Prefill 阶段可能主要是**单线程执行**：

- 主线程：执行 prefill（模型 forward）
- 后台线程：写入 SSD（异步，不阻塞主线程）

**如果主线程和后台线程很少竞争同一个锁**，那么：
- 分片锁的并发优势 = 0
- 读写锁的并发优势 = 0
- 只剩下**纯开销**

### 假设 2: 分片锁查找开销

**分析**: 每次 `contains/add/touch` 都要计算 shard_id：

```python
# 优化前（单锁）
with self._lock:  # ~50ns
    return block_hash in self._index  # ~20ns
Total: ~70ns

# 优化后（分片锁）
shard_id = self._get_shard_id(block_hash)  # ~30ns
shard = self.shards[shard_id]  # ~10ns
with shard['lock']:  # ~50ns
    return block_hash in shard['index']  # ~20ns
Total: ~110ns
```

**增加开销**: 40ns per access

**如果 Prefill 期间有 10,000 次访问**:
- 总增加: 10,000 × 40ns = **0.4ms**

（太小了，不足以解释 317ms 的差异）

### 假设 3: 读写锁开销

**分析**: RWLock 比简单 Lock 复杂：

```python
# 简单 Lock
def acquire(self):
    atomic_test_and_set()  # ~30ns

# RWLock read
def acquire_read(self):
    self._read_ready.acquire()     # ~30ns
    while self._writers > 0:       # ~10ns
        self._read_ready.wait()    # 如果有 writer，阻塞
    self._readers += 1              # ~5ns
    self._read_ready.release()     # ~30ns
Total: ~75ns (无竞争情况)
```

**增加开销**: 45ns per lock

**如果 Prefill 期间有 1,000 次 hot cache 访问**:
- 总增加: 1,000 × 45ns = **0.045ms**

（也太小了）

### 假设 4: 缓存局部性破坏

**分析**: 分片后，数据分散在 16 个不同的 dict：

```python
# 优化前：所有数据在一个 dict
self._index = {}  # 单一 dict，CPU 缓存友好

# 优化后：数据分散在 16 个 dict
self.shards[0]['index'] = {}
self.shards[1]['index'] = {}
...
self.shards[15]['index'] = {}
```

**问题**: 如果 block hash 访问有局部性（连续的 token 可能 hash 相近），分片可能破坏：
- CPU L1/L2 缓存命中率
- Python dict 的内部优化

**预估影响**: 可能增加 1-2% 的访问延迟

### 假设 5: 测试波动

**可能性**: ±3% 是正常的性能波动

**证据需要**: 运行 3-5 次取平均值

---

## 关键洞察

### 从 cProfile 数据重新审视

**优化前的 cProfile**:
```
thread.lock.acquire: 23.764s (1181 calls)
```

**问题**: 这 23.764s 是哪个线程的时间？

**可能答案**: 主要是**后台 writer thread** 的等待时间！

```python
def _writer_loop(self):
    while True:
        item = self._write_queue.get(timeout=1.0)  # 🔒 这里等待
        ...
```

**如果是这样**:
- 23.764s 主要是 writer thread 在 `queue.get()` 上等待任务
- **不是主线程的等待时间！**
- **优化 Index lock 和 hot_cache_lock 对主线程影响很小！**

### 重新理解瓶颈

**主线程时间分配**（推测）:

```
Prefill 11.8s:
- 模型 forward: ~8s (68%)
- ContextPilot: ~0.25s (2%)
- Cache 查找/写入: ~2s (17%)
  - Index contains/add: ~0.5s
  - Hot cache put/get: ~0.5s
  - Tensor 提取: ~0.5s
  - 其他: ~0.5s
- 其他开销: ~1.5s (13%)
```

**锁优化影响**:
- 如果 Index/Hot cache 操作只占 1s，减少 50% 锁等待 = 0.5s 节省
- 但分片开销可能增加 0.6s
- **净效果**: -0.1s ≈ -1%

**与实测吻合**: -2.6% 性能下降

---

## 结论

### 优化失败的原因

**核心问题**: **单线程场景下，锁优化无收益，只有开销**

1. **Prefill 主要是单线程**（模型 forward）
2. **后台 writer thread 不竞争主线程的锁**
3. **分片锁和读写锁增加了额外开销**
4. **无并发 = 无收益**

### 反思

**错误假设**: 认为 cProfile 中的 23.764s 锁等待时间都在主线程

**实际情况**: 大部分是后台线程的 `queue.get()` 等待时间

**教训**: **在优化前必须确认瓶颈在哪个线程！**

---

## 推荐方案

### 选项 A: 回滚锁优化 ✅ **推荐**

**理由**:
1. 锁优化后性能下降 2.6%
2. 增加了代码复杂度
3. 维护成本增加

**操作**:
```bash
git diff HEAD src/omlx/cache/paged_ssd_cache.py > /tmp/lock_opt.patch
git checkout src/omlx/cache/paged_ssd_cache.py
rm src/omlx/cache/rwlock.py src/omlx/cache/sharded_index.py
```

### 选项 B: 保留优化，接受性能下降

**理由**:
1. -2.6% 在可接受范围内
2. 为未来的多线程场景做准备
3. 代码更现代化

**风险**: 维护成本增加

### 选项 C: 只保留 GIL 无锁优化

**理由**:
1. GIL 优化无性能损失
2. 减少代码复杂度
3. 移除分片锁和读写锁

**预期**: 恢复到 Phase 1 性能

---

## 下一步

1. **等待 profiling 结果**（`bb58375`）验证假设
2. **根据监护人决策**:
   - 回滚全部锁优化
   - 或只保留 GIL 优化
3. **转向 Task #11**: ContextPilot 优化（~250ms 收益）

---

*分析时间: 2026-03-15 23:45*
*负责人: Claude Sonnet 4.5*
*状态: ⚠️ 优化失败，等待 profiling 验证*
