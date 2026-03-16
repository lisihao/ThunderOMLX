# ThunderOMLX Benchmark Results (oMLX Admin Panel - 社区标准)

**Date**: 2026-03-15
**Method**: oMLX Admin Panel Built-in Benchmark
**Hardware**: M4 Pro 48GB
**Model**: Qwen3.5-35B-A3B (4-bit)

---

## Test Results

| Test | TTFT (ms) | TPOT (ms/tok) | pp TPS | tg TPS | E2E Latency | Throughput | Peak Mem |
|------|-----------|---------------|--------|--------|-------------|------------|----------|
| pp1024/tg128 | 1415.6 | 11.76 | 723.3 | **85.7** | 2.910s | 395.9 | 18.62 GB |
| pp4096/tg128 | 5554.3 | 13.13 | 737.5 | **76.7** | 7.222s | 584.9 | 18.73 GB |
| pp8192/tg128 | 11451.7 | 15.23 | 715.3 | **66.2** | 13.386s | 621.6 | 18.89 GB |

---

## vs Community Baseline

### pp8192/tg128 (社区标准对比)

| Metric | ThunderOMLX | Community | Difference |
|--------|-------------|-----------|------------|
| **Generation TPS** | **66.2 tok/s** | 71.3 tok/s | **-7.2%** ❌ |
| TTFT | 11.45s | ~3s (est.) | +282% slower |
| Peak Memory | 18.89 GB | ~19 GB | Similar |

**结论**: ThunderOMLX 在 8k 上下文测试中**比社区基准慢 7%**

---

## Performance Degradation with Context Length

| Context | tg TPS | vs pp1024 |
|---------|--------|-----------|
| 1024 | 85.7 tok/s | baseline |
| 4096 | 76.7 tok/s | -10.5% |
| 8192 | 66.2 tok/s | **-22.8%** |

**问题**: 长上下文性能显著下降

---

## 关键发现

### ✅ 短上下文性能良好
- pp1024: **85.7 tok/s** (与之前测试的 86.5 tok/s 一致)
- 与 Native MLX 基本相同

### ❌ 长上下文性能问题
- pp8192: **66.2 tok/s** (比社区慢 7%)
- TTFT 过高: **11.45s** (正常应该 < 5s)
- 性能随上下文长度线性下降

### 🔍 可能的原因

1. **Prefill 阶段瓶颈**
   - TTFT 11.45s 太高
   - pp TPS ~715 tok/s (正常应该 > 900 tok/s)

2. **KV Cache 配置**
   - 长上下文下 attention 计算开销
   - Paged attention 可能未正确启用

3. **ThunderOMLX 优化开销**
   - ContextPilot 判断开销？
   - Skip Logic 在长上下文下的开销？

4. **社区数据疑问**
   - 需要确认社区数据的测试条件
   - 可能使用了不同的优化配置

---

## 下一步

1. ✅ 完成社区标准 benchmark 测试
2. ❌ 性能不如预期，需要诊断
3. 🔍 检查环境变量配置
4. 🔍 对比 Native MLX 在相同条件下的性能
5. 🔍 Profile long-context prefill 阶段

---

*Generated: 2026-03-15*
*Test Method: oMLX Admin Panel Benchmark*
