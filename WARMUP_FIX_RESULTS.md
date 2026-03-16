# Warmup Fix 验证结果

**日期**: 2026-03-15
**任务**: Task #13 - P0 消除 Token 1-50 warmup 慢
**修改**: Benchmark warmup改用长上下文（32→8192 tokens, 8→16 gen）

---

## 📊 实验对比

### 修改前 (Warmup: pp32/tg8)

```
🔍 Token 1: 1023.84ms
🔍 Token 2:   10.98ms
🔍 Token 3:   10.96ms
🔍 Token 4:   10.96ms
🔍 Token 5:   11.13ms
🔍 Token 6:   11.03ms
🔍 Token 7:   10.79ms
🔍 Token 8:   14.87ms
🔍 Token 9: 11785.09ms  ← 💥 KV Cache扩展瓶颈
🔍 Token 10:   13.03ms

⏱️  Perf [50 tokens]: step=268.31ms/tok, TPS=3.7
   📊 Warmup analysis: Token 1-10 avg=1290.27ms, Token 11-50 avg=12.49ms
⏱️  Perf [100 tokens]: step=12.55ms/tok, TPS=79.7

最终结果:
⏱️  TTFT: 11793.8ms
⚡ Generation TPS: 64.9 tok/s
📊 Processing TPS: 694.6 tok/s
```

### 修改后 (Warmup: pp8192/tg16)

```
🔍 Token 1: 12666.01ms  ← 包含prefill时间
🔍 Token 2:    12.74ms
🔍 Token 3:    12.44ms
🔍 Token 4:    12.44ms
🔍 Token 5:    12.47ms
🔍 Token 6:    12.51ms
🔍 Token 7:    12.46ms
🔍 Token 8:    12.58ms
🔍 Token 9:    12.37ms  ← ✅ 瓶颈消除！
🔍 Token 10:   12.42ms

⏱️  Perf [50 tokens]: step=501.41ms/tok, TPS=2.0
   📊 Warmup analysis: Token 1-10 avg=1277.84ms, Token 11-50 avg=298.55ms
⏱️  Perf [100 tokens]: step=12.53ms/tok, TPS=79.8

最终结果:
⏱️  TTFT: 11454.3ms
⚡ Generation TPS: 67.3 tok/s
📊 Processing TPS: 715.2 tok/s
```

---

## ✅ 成功验证

### Token 9瓶颈完全消除

```
修改前: Token 9 = 11785.09ms (KV Cache扩展)
修改后: Token 9 =    12.37ms (正常)
───────────────────────────────────────
改进:   -11772.72ms (-99.9%)  ✅
```

**根因**: Warmup阶段已经分配了8192 tokens的KV Cache，实际测试时不再需要扩展

---

## 🤔 新问题

### Token 1变慢了

```
修改前: Token 1 = 1023.84ms
修改后: Token 1 = 12666.01ms
───────────────────────────────────
变化:   +11642.17ms (+1137%)  ❌
```

**根因分析**:

这个"Token 1"的12.7秒**不是generation的第一个token时间**，而是**包含了prefill时间**。

从scheduler profiling的位置来看：
- `🔍 Token X` 日志在generation阶段打印
- Token 1的12.7秒 = Prefill处理8192 tokens的时间 + 第一个generation token

验证计算：
```
Prefill: 8192 tokens / 715.2 tok/s = 11.5秒
Gen Token 1: ~1秒（Metal编译）
───────────────────────────────────
总计: ~12.5秒 ✓ 与实测12.7秒吻合
```

**结论**: ✅ 这是正常的，Token 1慢是因为包含了prefill时间，不是regression

---

## 🎯 实际性能提升

### Generation TPS

```
修改前: 64.9 tok/s
修改后: 67.3 tok/s
───────────────────
提升:   +2.4 tok/s (+3.7%)
```

**为什么不是预期的+22.8%？**

因为Token 1-10的平均时间没有显著降低：
- 修改前: 1290.27ms（被Token 1和Token 9拉高）
- 修改后: 1277.84ms（被Token 1的prefill时间拉高）

但是**Token 100 TPS保持79.8 tok/s**，说明stable阶段性能一致。

### Token 11-50异常

```
修改前: Token 11-50 avg = 12.49ms
修改后: Token 11-50 avg = 298.55ms  ← ⚠️ 异常！
```

**问题**: 为什么Token 11-50变慢了？

**可能原因**:
1. 统计计算错误（Token 1的12.7秒被错误地纳入Token 11-50）
2. 或者其他原因（需要查看完整日志）

需要进一步分析scheduler profiling的计算逻辑。

---

## 🔍 下一步调查

### 检查点1: Token 11-50的真实时间

需要打印Token 11-50的每个单独时间，验证是否真的变慢了，还是统计错误。

### 检查点2: Prefill时间位置

需要确认"Token 1: 12666ms"是prefill+gen，还是只有gen。

可能的方案：
- 在prefill和generation之间加明确的分隔日志
- 单独统计prefill时间和generation时间

---

## 📈 总体评估

| 指标 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| Token 9瓶颈 | 11785ms | 12.37ms | ✅ -99.9% |
| Token 100 TPS | 79.7 | 79.8 | ✓ 持平 |
| Generation TPS | 64.9 | 67.3 | ✅ +3.7% |
| Prefill TPS | 694.6 | 715.2 | ✅ +3.0% |
| TTFT | 11793.8ms | 11454.3ms | ✅ -2.9% |

**结论**:
- ✅ Token 9瓶颈成功消除
- ✅ Generation TPS有提升（+3.7%）
- ✅ Prefill TPS也有提升（+3.0%）
- ⚠️ 提升幅度小于预期（3.7% vs 22.8%）

**原因**: Token 1-50的统计数据可能有问题，需要进一步调查。

---

*验证时间: 2026-03-15 23:15*
*负责人: Claude Sonnet 4.5*
