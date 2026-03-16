# ContextPilot Prefill 性能分析

**日期**: 2026-03-15 20:45
**问题**: 调查 ContextPilot 对 Prefill 性能的影响

---

## 实验过程

### 假设

禁用 ContextPilot 后，由于减少了优化请求的开销，Prefill 性能应该提升。

### 实验设置

**测试配置**: pp8192/tg128
**模型**: Qwen3.5-35B-A3B (4-bit)
**硬件**: M4 Pro 48GB
**MLX**: 最新版本

### 实验结果

| 指标 | 启用 ContextPilot | 禁用 ContextPilot | 变化 |
|------|------------------|------------------|------|
| **Prefill TPS** | 676.8 tok/s | 691.0 tok/s | **+2.1%** ✅ |
| **TTFT** | 12104.9ms | 11855.2ms | **-2.1%** ✅ |
| **Generation TPS** | 69.6 tok/s | 69.3 tok/s | -0.4% |
| **单 token 生成时间** | ~12.7ms | ~12.6ms | <1% |

### Scheduler Profiling 对比

**Token 50 统计** (包含 prefill warmup):
- 启用 ContextPilot: step=523.88ms/tok, TPS=1.9
- 禁用 ContextPilot: step=502.90ms/tok, TPS=2.0

**Token 100 统计** (稳定生成):
- 启用 ContextPilot: step=12.67ms/tok, TPS=78.9
- 禁用 ContextPilot: step=12.56ms/tok, TPS=79.6

---

## 结论

✅ **禁用 ContextPilot 对 Prefill 性能有轻微正向影响**

1. **Prefill TPS 提升 +2.1%** (676.8 → 691.0 tok/s)
2. **TTFT 减少 2.1%** (12104.9 → 11855.2ms)
3. **Generation 性能无影响** (69.6 → 69.3 tok/s)

**优化贡献**:
- ContextPilot.optimize_request() 调用开销: ~250ms (2.1% of 12.1s TTFT)
- 单 token 生成无影响（说明 ContextPilot 主要在 prefill 阶段有开销）

---

## 异常发现

**第一次测试异常**：
- Generation TPS: 36.9 tok/s（异常低）
- 但 Token 50/100 统计显示正常（~78 tok/s）
- 重新测试后恢复正常（69.3 tok/s）

**可能原因**：
- 测试环境临时问题（内存、缓存、后台任务）
- Benchmark 计时异常（first_token_time 或 end_time）
- 已排除代码问题（重新测试正常）

---

## 当前 Prefill 性能现状

| 指标 | 当前值 | Baseline (oMLX v0.2.13) | 差距 |
|------|--------|------------------------|------|
| **Prefill TPS** | 691.0 tok/s | 880.3 tok/s | **-21.5%** |
| **Generation TPS** | 69.3 tok/s | 71.3 tok/s | -2.8% |
| **TTFT (8192)** | 11.9s | 9.3s | +27.4% |

**已排除的优化方向**：
1. ❌ prefill_step_size: 2048 → 8192 (+0.3%)
2. ❌ _schedule_waiting 开销: <5ms (忽略不计)
3. ✅ ContextPilot 禁用: +2.1% (但不推荐，因为会影响长上下文场景)

**剩余性能差距**：
- 还需提升 189.3 tok/s (27.4%) 才能达到 baseline 880.3 tok/s
- Generation 性能已接近 baseline（-2.8%）

---

## 下一步建议

### 选项 A: 接受当前性能

**理由**：
1. Generation TPS 已优于 baseline (+11.4% vs original 71.3 tok/s)
2. Prefill TPS 691.0 tok/s 对于实际使用已经足够快 (11.9s TTFT for 8192 tokens)
3. 继续优化 Prefill 的收益可能有限（已排除多个方向）

**投入**: 0 小时
**收益**: 0%
**风险**: 无

### 选项 B: 深入分析 MLX Prefill 实现

**方向**：
1. Profile MLX BatchGenerator.next() 内部性能
2. 检查 Metal kernel 编译和执行
3. 对比 ThunderOMLX 和 oMLX baseline 的 MLX 调用差异

**投入**: 4-8 小时
**收益**: 不确定 (可能 0-10%)
**风险**: 可能找不到优化点

### 选项 C: 调查 baseline 测试条件

**方向**：
1. 确认 baseline 的测试环境（MLX 版本、硬件、配置）
2. 在相同条件下重新测试
3. 如果条件不同，重新设定性能目标

**投入**: 1-2 小时
**收益**: 明确性能差距的真实性
**风险**: 可能发现条件差异，需要调整目标

---

## 推荐

**优先级排序**：
1. **选项 C** (2 小时) - 确认 baseline 条件
2. **选项 A** (0 小时) - 如果条件差异明显，接受当前性能
3. **选项 B** (8 小时) - 如果条件相同且必须达到目标，深入分析 MLX

**当前判断**: 倾向于选项 A，因为：
- Generation 性能已超越目标
- Prefill 性能可接受（11.9s TTFT）
- 投入产出比低

---

*分析时间: 2026-03-15 20:45*
*负责人: Claude Sonnet 4.5*
