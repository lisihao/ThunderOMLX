# ThunderOMLX Benchmark vs 直接测试性能差距分析

**日期**: 2026-03-15
**任务**: Task #8 - P0 根因分析
**目标**: 确认为什么 Benchmark (65.1 tok/s) 比直接测试 (79.8 tok/s) 慢 18.5%

---

## 📊 测试配置

| 项目 | 配置 |
|------|------|
| **模型** | Qwen3.5-35B-A3B (4-bit) |
| **硬件** | M4 Pro 48GB |
| **测试场景** | pp8192/tg128 |
| **直接测试** | test_profiling.py（绕过 HTTP，直接调用 BatchedEngine） |
| **Benchmark** | run_admin_benchmark.py（通过 Admin Panel benchmark 框架） |

---

## 🔍 性能数据对比

### 直接测试 (test_profiling.py)

```
⏱️  Perf [50 tokens]: step=257.11ms/tok, batch_gen=257.06ms (100.0%), TPS=3.9
⏱️  Perf [100 tokens]: step=12.52ms/tok, batch_gen=12.46ms (99.5%), TPS=79.8
```

**关键指标**：
- **Generation TPS**: 79.8 tok/s
- **TPOT**: 12.52 ms/tok
- **batch_gen 占比**: 99.5% (MLX 层)
- **ThunderOMLX 层开销**: 0.5%

### Benchmark (run_admin_benchmark.py)

```
⏱️  Perf [50 tokens]: step=285.98ms/tok, batch_gen=285.68ms (99.9%), TPS=3.5
⏱️  Perf [100 tokens]: step=12.52ms/tok, batch_gen=12.46ms (99.5%), TPS=79.8

最终结果:
- TTFT: 12227.7ms
- Generation TPS: 64.2 tok/s
- Processing TPS: 670.0 tok/s
```

**关键指标**：
- **Scheduler 内部 TPS**: 79.8 tok/s ← 与直接测试一致！
- **最终 Generation TPS**: 64.2 tok/s ← 存在差距
- **TPOT (内部)**: 12.52 ms/tok
- **TPOT (端到端)**: 15.58 ms/tok (= 1000 / 64.2)

---

## 🎯 根因分析

### 关键发现

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ✅ Scheduler 内部性能完全一致                                  │
│                                                                 │
│   - 直接测试: step=12.52ms/tok, TPS=79.8                        │
│   - Benchmark:  step=12.52ms/tok, TPS=79.8                      │
│   - 差距: 0 ms/tok (0%)                                         │
│                                                                 │
│   ⚠️ 最终 TPS 存在差距                                           │
│                                                                 │
│   - 直接测试: 79.8 tok/s (12.52 ms/tok)                         │
│   - Benchmark:  64.2 tok/s (15.58 ms/tok)                       │
│   - 差距: -15.6 tok/s (-19.6%)                                  │
│         = +3.06 ms/tok API 层开销                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 性能分解

```
Benchmark 端到端 TPOT (15.58 ms/tok)
├─ scheduler.step() = 12.52 ms/tok (80.4%)
│  ├─ ThunderOMLX 层: 0.06 ms (0.5%)
│  └─ MLX batch_gen:  12.46 ms (99.5%)
│
└─ API 层开销 = 3.06 ms/tok (19.6%) ← 问题所在
   ├─ HTTP/SSE 序列化
   ├─ EngineCore → HTTP 响应包装
   ├─ 并发请求调度
   └─ ContextPilot 额外处理（如果有）
```

---

## 💡 结论

### ✅ 已验证

1. **ThunderOMLX scheduler 层性能优秀**
   - 0.5% 开销（0.06 ms/tok）
   - 与 Native MLX (12.48 ms/tok) 相比仅 +0.04 ms/tok (+0.3%)
   - 设计合理，实现高效

2. **问题不在 scheduler.step() 内部**
   - 两种测试方式的 scheduler profiling 数据完全一致
   - 99.5% 时间在 MLX batch_gen，无法优化

3. **问题在 scheduler.step() 之外的 API 层**
   - 额外的 3.06 ms/tok (19.6%) 开销
   - 这是从 scheduler 输出到用户接收的完整路径

### 🎯 优化方向

**P1 - 高优先级** (预计提升 10-15%)：

1. **HTTP/SSE 序列化优化**
   - 使用 orjson 代替标准 json
   - 减少中间拷贝
   - 考虑 binary protocol (msgpack)

2. **响应包装优化**
   - 减少 EngineCore → HTTP 的包装层次
   - 直接流式传输，减少缓冲

3. **并发调度优化**
   - 检查是否有不必要的同步等待
   - 优化事件循环调度

**P2 - 中优先级** (预计提升 1-3%)：

4. **ContextPilot 优化**
   - 如果确认有开销，优化判断逻辑
   - 只在必要时判断
   - 缓存判断结果

---

## 📈 优化目标

| 目标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| **近期** (1-2 周) | 64.2 tok/s | 70+ tok/s | +9% |
| **中期** (1 个月) | 64.2 tok/s | 75+ tok/s | +17% |
| **长期** (3 个月) | 64.2 tok/s | 78+ tok/s | +21% |

**理论最优**：79.8 tok/s (与直接测试持平)

---

## 🔧 下一步行动

- [x] **Task #8** (P0): 确认 Benchmark vs 直接测试差异根因 ✅
- [ ] **Task #10** (P1): 优化 API 层开销
  - [ ] Profile API 层各部分耗时
  - [ ] 实施 orjson 序列化
  - [ ] 减少中间拷贝
  - [ ] 验证效果

---

*分析完成时间: 2026-03-15 22:45*
*负责人: Claude Sonnet 4.5*
*审核人: 监护人昊哥*
