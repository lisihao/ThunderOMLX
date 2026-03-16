# Token 11-50 性能异常分析

**日期**: 2026-03-15
**问题**: Token 11-50平均298.55ms/tok（比正常慢24×）

---

## 📊 数据对比

### 修改前（Warmup: pp32/tg8）

```
Token 1-10 avg:  1290.27ms  ← 被Token 1和Token 9拉高
Token 11-50 avg:   12.49ms  ← 正常
Token 51-100 avg:  12.55ms  ← 正常
```

### 修改后（Warmup: pp8192/tg16）

```
Token 1-10 avg:  1277.84ms  ← 被Token 1拉高（12666ms）
Token 11-50 avg:  298.55ms  ← ⚠️ 异常！比预期慢24倍
Token 51-100 avg:  12.53ms  ← 正常
```

---

## 🔍 手动验证计算

### Token 1-50 总时间

从日志: `step=501.41ms/tok, TPS=2.0`

```
Token 1-50 总时间 = 501.41 × 50 = 25070.5ms
```

### Token 1-10 总时间

```
Token 1:  12666.01ms
Token 2:     12.74ms
Token 3:     12.44ms
Token 4:     12.44ms
Token 5:     12.47ms
Token 6:     12.51ms
Token 7:     12.46ms
Token 8:     12.58ms
Token 9:     12.37ms
Token 10:    12.42ms
──────────────────────
总计:    12778.43ms
平均:     1277.84ms ✓ 与日志一致
```

### Token 11-50 总时间

```
Token 11-50 总时间 = 25070.5 - 12778.43 = 12292.07ms
Token 11-50 平均   = 12292.07 / 40 = 307.30ms

日志显示: 298.55ms (差异可能是四舍五入)
```

**结论**: Token 11-50确实变慢了，平均~307ms/tok

---

## 🤔 可能根因

### 假设 #1: Warmup 导致 Cache 失效？

**理论**: Warmup使用了8192 tokens，可能导致后续测试的cache失效？

**验证**: ❌ 不太可能
- 每次测试都是新请求，不会复用warmup的cache
- 而且Token 51-100恢复正常（12.53ms），说明cache正常

---

### 假设 #2: Profiling 统计错误？

**理论**: `_perf_all_times` 统计有bug？

**验证**: ❌ 不太可能
- 代码逻辑正确：`self._perf_all_times.append(gen_time)`
- Token 1-10的平均计算正确
- 问题是Token 11-50真的慢，不是统计错误

---

### 假设 #3: 第二次 Prefill？（最可能）

**理论**: Warmup阶段已经完成了一次完整测试，但计数器没有重置，导致实际测试被认为是continuation

**关键问题**:
- Warmup是否真的执行了？
- Warmup的Token打印为什么没有出现在日志中？
- Profiling计数器在warmup后是否重置了？

**需要验证**:
1. 检查warmup是否真的运行了8192 tokens prefill + 16 gen
2. 检查profiling计数器是否在每次请求开始时重置
3. 检查是否有多个BatchGenerator实例

---

### 假设 #4: 批处理调度问题

**理论**: Token 11-50期间scheduler在做额外的工作（调度、内存管理等）

**验证**: 看batch_gen占比
```
修改前 Token 50: batch_gen=268.05ms (99.9%)
修改后 Token 50: batch_gen=494.41ms (98.6%)
```

batch_gen占比从99.9% → 98.6%，说明ThunderOMLX层开销增加了1.3%。

但是batch_gen绝对值从268.05ms → 494.41ms，**增加了84.5%！**

**这是关键！**

---

## 🎯 关键发现

### batch_gen 时间异常

```
修改前:
Token 50: batch_gen=268.05ms, step=268.31ms
Token 100: batch_gen=12.46ms, step=12.55ms

修改后:
Token 50: batch_gen=494.41ms, step=501.41ms
Token 100: batch_gen=12.46ms, step=12.53ms
```

**问题**: 为什么Token 50的batch_gen从268ms变成494ms（+84.5%），而Token 100保持12.46ms不变？

**可能解释**:
1. **Token 1-50包含了额外的overhead**（不在Token 1-10中）
2. **MLX层在Token 11-50期间有额外工作**
3. **Metal缓冲区管理、内存分配延迟分散在Token 11-50**

---

## 🔧 下一步调查

### 1. 打印Token 11-20的单独时间

修改scheduler.py，打印前20个token而不是10个：

```python
# Print first 20 tokens individually for detailed analysis
if self._perf_gen_count <= 20:
    logger.info(f"🔍 Token {self._perf_gen_count}: {gen_time:.2f}ms")
    print(f"🔍 Token {self._perf_gen_count}: {gen_time:.2f}ms", flush=True)
```

### 2. 添加Token 50-60的打印

```python
# Also print Token 50-60 to compare with Token 11-20
if 50 <= self._perf_gen_count <= 60:
    logger.info(f"🔍 Token {self._perf_gen_count}: {gen_time:.2f}ms")
    print(f"🔍 Token {self._perf_gen_count}: {gen_time:.2f}ms", flush=True)
```

### 3. 检查warmup是否真的运行

在benchmark.py的warmup后添加日志：

```python
logger.info(f"Benchmark: warmup complete (pp{max_pp_len}, tg16)")
logger.info(f"Warmup generated tokens, profiling counter should be reset")
```

### 4. 检查profiling计数器重置

在scheduler.py的add_request或相关位置，确认每次新请求是否重置计数器。

---

## 📌 当前结论

**Token 9瓶颈成功消除** ✅
- 从11785ms → 12.37ms
- Warmup改用长上下文确实有效

**但引入了新问题** ⚠️
- Token 11-50变慢（12.49ms → 298.55ms）
- Token 50的batch_gen从268ms → 494ms（+84.5%）
- 需要进一步调查根因

**暂时结论**:
- Generation TPS提升了3.7%（64.9 → 67.3 tok/s），说明整体有改进
- 但还有优化空间，需要解决Token 11-50的异常

---

*分析时间: 2026-03-15 23:25*
*负责人: Claude Sonnet 4.5*
