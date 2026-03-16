# Phase 1-4 实施总结

> **创建时间**: 2026-03-16
> **状态**: 代码实现完成，性能验证待改进

---

## ✅ 实施完成情况

### 代码修改统计

- **总修改**: +358/-38 行
- **新文件**: 1 个（cache_save_executor.py）
- **修改文件**: 3 个
  - src/omlx/cache/paged_ssd_cache.py
  - src/omlx/cache/prefix_cache.py
  - src/omlx/scheduler.py

### Phase 实施明细

| Phase | 优化内容 | 预期收益 | 实施状态 |
|-------|---------|---------|---------|
| **Phase 1** | 异步 Tensor 提取 | +2.8% | ✅ 完成 |
| **Phase 2** | 异步 save_block 调用 | +2.1% | ✅ 完成 |
| **Phase 3** | 减少调度间隙 | +1.0% | ✅ 完成（instrumentation）|
| **Phase 4** | 批量 Metal 操作 | +0.3% | ✅ 完成 |
| **总计** | - | +6.3% | ✅ 代码完成 |

---

## 🐛 Bug 修复记录

### Bug 1: `tensors_raw` 未定义（CRITICAL）

**问题**:
```python
Failed to prepare block for SSD cache: name 'tensors_raw' is not defined
```

**根因**:
- Phase 1 修改中，将 tensors_raw 提取移到后台线程
- 但忘记了 Hot Cache 启用时仍需要在推理线程提取（用于立即缓存）

**修复**: paged_ssd_cache.py:1468-1477
```python
tensors_raw = None
if self._hot_cache_enabled:
    # Hot Cache 需要立即存储 tensors_raw
    tensors_raw = {}
    for name, arr in arrays.items():
        tensors_raw[name] = _extract_tensor_bytes(arr)
```

---

### Bug 2: `time` 变量冲突

**问题**:
```python
UnboundLocalError: cannot access local variable 'time' where it is not associated with a value
```

**根因**:
- _writer_loop 函数内部有 `import time`（line 1275）
- Python 将 `time` 视为局部变量
- 导致之前使用 `time.time()` 时报错（line 1244）

**修复**: 删除 line 1275 的重复 `import time`（文件开头已 import）

---

### Bug 3: `_hot_cache_entry_size` 不处理 `None`

**问题**:
```python
AttributeError: 'NoneType' object has no attribute 'values'
```

**根因**:
- Hot Cache 禁用时，`tensors_raw` 为 `None`
- 但 `_hot_cache_entry_size` 期望 `Dict` 类型

**修复**: paged_ssd_cache.py:835-839
```python
def _hot_cache_entry_size(tensors_raw: Optional[Dict[str, tuple]]) -> int:
    if tensors_raw is None:
        return 0
    return sum(len(raw) for raw, _, _ in tensors_raw.values())
```

---

## ✅ 功能验证

### test_phase_all_functional.py（5/5 通过）

- ✅ CacheSaveExecutor import
- ✅ skip_eval 参数存在
- ✅ Scheduler 使用 CacheSaveExecutor
- ✅ Prefix Cache 批量 eval 逻辑
- ✅ Phase 3 队列延迟 instrumentation

---

## ⚠️ 性能验证问题

### 当前测试结果（test_processing_tps.py）

```
总请求数: 4
总 tokens: 512
总 walltime: 9.19s

Processing TPS: 55.7 tok/s

单个请求详情:
  R1: 128 tokens in 2.71s (47.1 tok/s)  # 含 warmup
  R2: 128 tokens in 1.49s (85.9 tok/s)
  R3: 128 tokens in 1.49s (85.9 tok/s)
  R4: 128 tokens in 1.50s (85.5 tok/s)
```

**Generation TPS**: 85-90 tok/s（scheduler 内部）
**Processing TPS**: 55.7 tok/s（总体，包含等待时间）

---

### 🚨 关键发现：测试场景不匹配

#### 原始基准（PHASE1_2_ANALYSIS.md）

- **Processing TPS**: 692.7 tok/s
- **Walltime**: 36.606s
- **总 Tokens**: ~25,000 tokens
- **save_block 耗时**: 2.603s（占总时间 7.1%）
- **场景**: 并发 Agent 请求，大量 cache 操作

#### 当前测试（test_processing_tps.py）

- **Processing TPS**: 55.7 tok/s
- **Walltime**: 9.19s
- **总 Tokens**: 512 tokens
- **save_block 触发**: 未知（无日志）
- **场景**: **顺序执行**单一会话请求

#### 差距分析

| 指标 | 原始基准 | 当前测试 | 差距 |
|------|---------|---------|------|
| Processing TPS | 692.7 | 55.7 | **12.4x** |
| 总 Tokens | 25,000 | 512 | **48.8x** |
| 执行方式 | 并发 | 顺序 | - |
| save_block | 大量触发 | 很少/未触发 | - |

---

### 为什么当前测试无法体现优化？

1. **顺序执行 vs 并发执行**
   - 当前：walltime = Σ(所有请求时间) + 等待时间
   - 原始：walltime ≈ max(请求时间)
   - Phase 2 的异步优化在并发场景才有效

2. **tokens 太少**
   - 512 tokens → 很少的 cache save 操作
   - save_block 优化（Phase 1-4）无法体现

3. **没有触发 Phase 3/4**
   - Phase 3 queue latency: 无 warnings（<100ms）
   - Phase 4 batch eval: 无日志（没有多 blocks）

4. **BatchedEngine vs llama-server**
   - 原始基准可能使用 llama-server（并发 HTTP 请求）
   - 当前测试使用 BatchedEngine（单一会话）

---

## 📋 下一步建议

### 选项 1：基于原始基准重现（推荐）

**目标**: 找到或创建与 PHASE1_2_ANALYSIS.md 中相同的测试场景

**步骤**:
1. 确认原始基准的测试方法
   - 是否使用 llama-server？
   - 并发请求数量？
   - 每个请求的 tokens？

2. 重现相同场景
   - 启动 llama-server
   - 使用 benchmark_omlx.py 或类似工具
   - 并发发送 4+ 请求，每个 ~1024 tokens

3. 对比优化前后
   - 基线：save_block 耗时 ~2.6s（7.1%）
   - 优化后：预期节省 ~2.3s（Phase 1-4）
   - 目标：Processing TPS ≥ 730 tok/s（+5.4%）

---

### 选项 2：创建极端 cache save 场景

**目标**: 强制触发大量 save_block 操作

**方法**:
- 生成更多 tokens（8192 prefill + 1024 generation）
- 多个请求，触发 cache eviction 和 save
- 检查 Phase 3/4 日志是否出现

---

### 选项 3：接受当前验证（不推荐）

**理由**:
- 代码逻辑正确（功能测试通过）
- Bug 已修复（3 个关键 bug）
- 但性能提升**未验证**

**风险**:
- 可能引入性能回退（Phase 4 batch eval）
- 无法证明达成目标（+5.4%）

---

## 📊 Phase 3 Instrumentation 详情

### 已添加的日志点

1. **队列延迟测量**（paged_ssd_cache.py:1244-1250）
   ```python
   dequeue_time = time.time()
   queue_latency_ms = (dequeue_time - enqueue_time) * 1000
   if queue_latency_ms > 100:
       logger.warning(f"⚠️ High queue latency: {queue_latency_ms:.1f}ms...")
   ```

2. **入队时间戳**（paged_ssd_cache.py:1540）
   ```python
   enqueue_time = time.time()
   self._write_queue.put_nowait((block_hash, arrays, metadata, file_path, enqueue_time))
   ```

### 当前测试未触发的原因

- 队列延迟 < 100ms（无 warning）
- 可能是因为：
  - save_block 调用次数少
  - 后台 writer 线程处理速度快
  - 没有 GIL 竞争

---

## 📊 Phase 4 Batch Eval 详情

### 已添加的日志点

**prefix_cache.py:708-712**:
```python
logger.info(f"⚡ Phase 4: Batch eval {len(all_tensors_for_batch_eval)} tensors...")
if HAS_MLX:
    import mlx.core as mx
    mx.eval(*all_tensors_for_batch_eval)
    mx.synchronize()
```

### 当前测试未触发的原因

- 没有多个 blocks 需要批量保存
- 可能需要：
  - 更长的 context（触发多 block save）
  - 更多请求（触发 cleanup_finished）

---

## 🎯 成功标准（来自计划）

| Phase | 最低 TPS | _cleanup_finished | 队列延迟 p99 |
|-------|---------|------------------|-------------|
| Phase 1 | ≥ 710 | - | - |
| Phase 2 | ≥ 725 | < 500ms | - |
| Phase 3 | ≥ 732 | - | < 200ms |
| Phase 4 | ≥ 736 | - | - |
| **总体目标** | **≥ 730 tok/s** | - | - |

**当前状态**: 无法测量（测试场景不匹配）

---

## 📁 相关文件

### 代码文件
- src/omlx/cache/paged_ssd_cache.py（Phase 1, 3, 4）
- src/omlx/cache/cache_save_executor.py（Phase 2）
- src/omlx/cache/prefix_cache.py（Phase 4）
- src/omlx/scheduler.py（Phase 2）

### 测试文件
- test_phase_all_functional.py（功能验证）
- test_processing_tps.py（性能测试 - 场景不匹配）
- test_profiling.py（Generation TPS 测试）

### 文档文件
- .solar/PHASE1_2_ANALYSIS.md（原始分析）
- .solar/PHASE3_QUEUE_LATENCY_INSTRUMENTATION.md（Phase 3 详情）
- .solar/PHASE4_BATCH_METAL_OPS.md（Phase 4 详情）
- .solar/PROCESSING_TPS_OPTIMIZATION_SUMMARY.md（总体规划）

---

## 总结

✅ **代码实施**: 完成（+358/-38 行，3 bug 修复）
✅ **功能验证**: 通过（5/5 测试）
⚠️  **性能验证**: **待改进**（测试场景不匹配）

**建议**: 基于原始基准重现测试场景，验证 Processing TPS 从 692.7 → 730 tok/s 的提升。
