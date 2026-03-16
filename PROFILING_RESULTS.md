# ThunderOMLX 性能分析结果

**日期**: 2026-03-15
**版本**: v0.3.0
**硬件**: M4 Pro 48GB
**模型**: Qwen3.5-35B-A3B (4-bit)

---

## 关键发现

### 1. 性能分析数据（直接测试）

通过在 `scheduler.step()` 中添加性能分析代码，测量了每个 token 生成的耗时：

| 阶段 | step (ms/tok) | batch_gen (ms/tok) | batch_gen 占比 | TPS |
|------|---------------|-------------------|---------------|-----|
| **Token 1-50** (含 prefill) | 249.54 | 234.14 | 93.8% | 4.0 |
| **Token 51-100** (纯 generation) | 12.52 | 12.46 | 99.5% | 79.8 |

**结论**：
- ✅ **纯 generation 性能**: 79.8 tok/s，与 Native MLX (80.1 tok/s) 几乎一致
- ✅ **batch_generator.next() 占比 99.5%**，说明瓶颈在 MLX 层，不在 ThunderOMLX 层
- ✅ **ThunderOMLX 调度开销**：仅 0.06 ms/tok (0.5%)

---

## 2. 与 Benchmark 结果的对比

| 测试方式 | tg TPS | TPOT (ms/tok) | 环境 |
|----------|--------|---------------|------|
| **Admin Benchmark** | 65.1 tok/s | 15.47 | 干净环境，禁用压缩 |
| **直接测试** | 79.8 tok/s | 12.52 | 单请求，直接调用 engine |
| **Native MLX** | 80.1 tok/s | 12.48 | Baseline |

**差异**: 14.7 tok/s (18.5%)

---

## 3. 差异原因分析

### 可能原因 1: 测试方法不同

**Admin Benchmark**:
- 通过 HTTP/SSE API
- 包含网络序列化/反序列化开销
- 可能有多个请求并发（max_num_seqs）
- 包含 ContextPilot 判断逻辑

**直接测试**:
- 直接调用 BatchedEngine
- 单请求运行
- 无 HTTP 开销

### 可能原因 2: 上下文差异

**分析**:
| 因素 | Admin Benchmark | 直接测试 |
|------|----------------|----------|
| 请求数 | 可能并发 | 单请求 |
| API 层 | HTTP/SSE | 直接调用 |
| ContextPilot | 启用 | 启用 |
| 缓存状态 | 干净启动 | 干净启动 |

### 可能原因 3: 长上下文开销

从之前的分析可知：
- pp1024: 78.5 tok/s (ThunderOMLX)
- pp8192: 65.1 tok/s (Admin Benchmark)
- pp8192: 79.8 tok/s (直接测试)

**假设**: 直接测试可能没有正确模拟长上下文的 KV cache 加载开销。

---

## 4. 性能瓶颈定位

### scheduler.step() 分解

```
Total step time: 12.52 ms/tok
├─ Process aborts: ~0.01 ms
├─ Memory check: ~0.01 ms
├─ Schedule waiting: ~0.02 ms
├─ batch_generator.next(): 12.46 ms (99.5%)  ← 瓶颈在这里
├─ Process responses: ~0.01 ms
└─ Cleanup: ~0.01 ms
```

### batch_generator.next() 内部（MLX 层）

无法直接 profile，但可能包含：
- KV Cache 加载 (block-aware cache)
- Attention 计算 (Flash Attention)
- Token sampling
- MLX Metal 调用

---

## 5. 下一步行动

### 高优先级

1. **重现 Admin Benchmark 的 65.1 tok/s**
   - 在直接测试中模拟相同条件
   - 确认差异来源

2. **Profile KV Cache 操作**
   - 统计每个 token 加载的 block 数量
   - 测量 block 加载时间
   - 对比 pp1024 vs pp8192 的 cache 行为

3. **对比不同场景**
   - 单请求 vs 并发请求
   - 有/无 ContextPilot
   - 不同 context 长度

### 中优先级

1. **优化 API 层开销**
   - 如果确认 HTTP/SSE 有开销，优化序列化
   - 减少中间拷贝

2. **优化 ContextPilot**
   - Profile 判断逻辑耗时
   - 优化判断频率

### 低优先级

1. **MLX 层优化**
   - 由于 99.5% 时间在 MLX，优化空间有限
   - 除非能优化 KV cache 或 attention 实现

---

## 6. 技术细节

### 性能分析代码位置

文件: `src/omlx/scheduler.py`

**添加的代码**:

```python
# Line 3841: 开始计时
def step(self) -> SchedulerOutput:
    step_start = time.perf_counter()
    # ...

# Line 3861: 测量 batch_generator.next()
if self.batch_generator is not None and self.running:
    gen_start = time.perf_counter()
    responses = self.batch_generator.next()
    gen_time = (time.perf_counter() - gen_start) * 1000
    # ...

# Line 3960: 每 50 tokens 打印统计
if hasattr(self, '_perf_gen_count') and self._perf_gen_count % 50 == 0:
    avg_gen = sum(self._perf_gen_times) / len(self._perf_gen_times)
    avg_step = sum(self._perf_step_times[-50:]) / min(50, len(self._perf_step_times))
    print(f"⏱️  Perf [{self._perf_gen_count} tokens]: ...")
```

### 运行测试

```bash
cd ~/ThunderOMLX
python3 test_profiling.py
```

---

## 7. 结论

**核心发现**:
- ✅ ThunderOMLX 调度层开销 < 1% (0.06 ms/tok)
- ✅ 瓶颈 99.5% 在 MLX `batch_generator.next()`
- ⚠️ 直接测试 (79.8 tok/s) 与 Benchmark (65.1 tok/s) 存在 18.5% 差异
- ❓ 差异原因待确认：API 开销 vs 长上下文 KV cache 开销

**建议**:
1. 优先确认直接测试与 Benchmark 的差异来源
2. 如果是 API 层开销，优化 HTTP/SSE
3. 如果是 KV cache 开销，优化 block 加载策略

---

*报告生成时间: 2026-03-15 20:30*
*测试工具: test_profiling.py + scheduler.py 性能分析*
*分析者: Claude Sonnet 4.5*
