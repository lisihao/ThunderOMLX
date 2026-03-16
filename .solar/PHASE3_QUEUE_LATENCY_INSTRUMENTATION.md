# Phase 3: 队列延迟插桩

## 目标

最小化 queue_put → save_block 延迟，识别调度间隙根因。

**预期收益**: +1.0% (+7 tok/s → 734 tok/s 累计)
**风险**: 低
**实施时间**: 2026-03-16

---

## 实施总结

### 修改内容

**文件**: `/Users/lisihao/ThunderOMLX/src/omlx/cache/paged_ssd_cache.py`

#### 修改 1: save_block 入队时记录时间戳 (line 1526-1529)

```python
# Phase 3 优化：添加入队时间戳，用于测量调度延迟
enqueue_time = time.time()
try:
    self._write_queue.put_nowait(
        (block_hash, arrays, metadata, file_path, enqueue_time)  # Phase 3: 添加时间戳
    )
```

#### 修改 2: _writer_loop 出队时计算延迟 (line 1241-1251)

```python
# Phase 1 优化：接收 arrays 而非 tensors_raw
# Phase 3 优化：接收入队时间戳，计算调度延迟
block_hash, arrays, metadata, file_path, enqueue_time = item
temp_path = None

# Phase 3 优化：测量队列延迟（用于诊断调度间隙问题）
dequeue_time = time.time()
queue_latency_ms = (dequeue_time - enqueue_time) * 1000
if queue_latency_ms > 100:
    logger.warning(
        f"⚠️ High queue latency: {queue_latency_ms:.1f}ms for block {block_hash[:8]}... "
        f"(GIL contention?)"
    )
```

---

## 工作原理

### 延迟测量流程

```
推理线程                               Writer 线程
    │                                      │
    ├─ enqueue_time = time.time()         │
    ├─ put_nowait((..enqueue_time))       │
    │                                      │
    │                                  ├─ dequeue_time = time.time()
    │                                  ├─ latency = dequeue - enqueue
    │                                  ├─ if latency > 100ms: warning
    │                                  ├─ extract_tensor_bytes()
    │                                  └─ write to SSD
    ▼                                      ▼
```

### 诊断标准

| 延迟范围 | 判断 | 可能原因 |
|---------|------|----------|
| < 10ms | ✅ 正常 | 队列处理顺畅 |
| 10-100ms | ⚠️ 注意 | 轻微调度延迟 |
| > 100ms | 🔥 警告 | GIL 竞争 / 系统负载高 |
| > 500ms | 🚨 严重 | 需要切换 multiprocessing.Queue |

---

## 下一步分析

### 运行 Benchmark 收集数据

```bash
cd /Users/lisihao/ThunderOMLX
python benchmark_prefill_generation.py \
  --model ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q5_K_M.gguf \
  --input-length 8192 --output-length 128 \
  --warmup 1 --trials 5 2>&1 | tee /tmp/phase3_latency.log
```

### 分析延迟模式

```bash
# 提取高延迟警告
grep "High queue latency" /tmp/phase3_latency.log | \
  awk '{print $5}' | sed 's/ms//' | sort -n

# 统计分布
grep "High queue latency" /tmp/phase3_latency.log | wc -l
```

### 判断是否需要优化

- **如果延迟持续 > 500ms** → GIL 竞争确认，考虑切换到 `multiprocessing.Queue`
- **如果延迟分布均匀 < 100ms** → 调度间隙不是瓶颈，跳过优化
- **如果偶尔出现高延迟** → 系统负载波动，不需要修改代码

---

## 约束验证

✅ **不破坏现有 API**: 队列数据格式变更仅在内部，外部接口不变
✅ **不引入新依赖**: 使用标准库 `time` 模块（已导入）
✅ **向后兼容**: 插桩不影响功能，仅添加监控
✅ **语法检查**: `python3 -m py_compile` 通过

---

## 性能影响

- **入队开销**: `time.time()` ~0.001ms (可忽略)
- **出队开销**: `time.time()` ~0.001ms + 条件判断 ~0.001ms (可忽略)
- **日志开销**: 仅在高延迟时触发 (预期 < 1% 情况)

**总开销**: < 0.01ms per block (可忽略)

---

## 状态

✅ **代码修改完成**
✅ **语法验证通过**
⚠️ **性能测试待运行** (需要实际推理)

---

*Phase 3 实施于: 2026-03-16*
*修改文件: paged_ssd_cache.py (+9 lines)*
