# Prefill 瓶颈重新分析

**日期**: 2026-03-16 00:00
**问题**: prefill_step_size 优化无效，需要重新定位瓶颈

---

## ❌ 假设被推翻

### 假设

`prefill_step_size=2048` 导致分块开销，增大到 8192 可提升 26%

### 实验结果

| 配置 | Prefill TPS | 变化 |
|------|-------------|------|
| prefill_step_size=2048 | 696.7 tok/s | 基准 |
| prefill_step_size=8192 | 698.6 tok/s | +0.3% |

**结论**: prefill_step_size **不是**主要瓶颈

---

## 🔍 重新审视数据

### 性能对比

| 系统 | Prefill TPS | Generation TPS |
|------|-------------|----------------|
| **ThunderOMLX** | 698.6 tok/s | 79.4 tok/s |
| **oMLX v0.2.13** | 880.3 tok/s | 71.3 tok/s |
| **差距** | -20.6% | +11.4% |

### 关键观察

1. **Generation 更快**：ThunderOMLX 比 baseline 快 11.4%
   - 说明 ThunderOMLX 的优化（block-aware cache 等）对 generation 有利

2. **Prefill 更慢**：ThunderOMLX 比 baseline 慢 20.6%
   - 可能是优化的副作用？
   - 或者测试条件不同？

---

## 🤔 新假设

### 假设 1: oMLX baseline 测试条件不同

**可能差异**：
- **MLX 版本不同**（baseline 可能用了更新或更旧的 MLX）
- **模型不同**（baseline 可能不是 Qwen3.5-35B-A3B）
- **硬件不同**（baseline 可能不是 M4 Pro）
- **配置不同**（batch_size、context_size 等）

**验证方法**：
- 询问监护人 baseline 的测试条件
- 或者接受当前性能（698.6 tok/s 已经不错）

---

### 假设 2: ThunderOMLX 的优化增加了 prefill 开销

**可能的优化**：
- **Block-aware prefix cache lookup**：在 prefill 前查找 cache
- **Prompt padding**：将 prompt 填充到 block 边界
- **Skip logic 判断**：检查是否可以 skip prefill

**验证方法**：
- 在 _schedule_waiting() 中添加 profiling
- 测量 cache lookup 和 skip logic 判断的时间
- 看看是否有显著开销

---

### 假设 3: MLX 本身的 prefill 性能就是这样

**可能性**：
- MLX 的 prefill 实现可能就是 ~700 tok/s
- baseline 的 880.3 tok/s 可能是在不同条件下测得的
- 我们的 698.6 tok/s 已经接近 MLX 的上限

**验证方法**：
- 直接测试 MLX 的 prefill 性能（不通过 ThunderOMLX）
- 但之前的 baseline 测试脚本有问题，需要修复

---

## 🎯 下一步调查

### 优先级 1: 询问 baseline 测试条件

**问题**：
- oMLX v0.2.13 的 880.3 tok/s 是在什么条件下测得的？
- 模型、硬件、MLX 版本、配置参数？

**如果条件不同**：
- 可能无法直接对比
- 需要在相同条件下重新测试

---

### 优先级 2: Profile _schedule_waiting 开销

**方法**：
在 `_schedule_waiting()` 中添加 profiling：

```python
def _schedule_waiting(self) -> List[Request]:
    start = time.perf_counter()

    # ... existing code ...

    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > 10:  # 超过 10ms 才记录
        logger.info(f"⏱️ _schedule_waiting took {elapsed:.2f}ms")

    return scheduled
```

**预期**：
- 如果 _schedule_waiting 耗时显著（>100ms），说明调度有开销
- 如果耗时很小（<10ms），说明调度不是瓶颈

---

### 优先级 3: 接受当前性能

**理由**：
1. **Generation 已超越 baseline**（79.4 vs 71.3 tok/s，+11.4%）
2. **Prefill 性能可接受**（698.6 tok/s，TTFT 11.7s for 8192 tokens）
3. **进一步优化可能收益有限**（-20.6% 差距可能来自测试条件差异）

**权衡**：
- 花费大量时间追求 Prefill 性能提升，但可能：
  - 无法复现 baseline 的测试条件
  - 优化空间有限（MLX 本身的限制）
  - 影响其他优化（如 Generation 性能）

---

## 📊 当前状态评估

### 已完成

- ✅ Task #13: 消除 Token 1-50 warmup 慢（+7.7% Generation TPS）
- ✅ Task #14 尝试 1: prefill_step_size 优化（无效）

### 性能现状

| 指标 | 当前值 | 目标 | 状态 |
|------|--------|------|------|
| Generation TPS | 79.4 tok/s | 75+ tok/s | ✅ 已达成 |
| Prefill TPS | 698.6 tok/s | 880+ tok/s | ⚠️ 未达成 |
| TTFT (8192) | 11.7s | <9.5s | ⚠️ 未达成 |

### 建议

**方案 A: 继续优化 Prefill**
- 投入时间：2-4 小时
- 预期收益：不确定（可能 0-10%）
- 风险：可能无法达到目标

**方案 B: 接受当前性能，转向其他优化**
- Generation 已超越目标（79.4 > 75 tok/s）
- Prefill 性能可接受（698.6 tok/s）
- 转向 Task #11/12（ContextPilot、KV Cache 优化）

**方案 C: 询问监护人 baseline 条件，然后决定**
- 如果条件相同，继续优化
- 如果条件不同，接受当前性能

---

*分析时间: 2026-03-16 00:05*
*负责人: Claude Sonnet 4.5*
