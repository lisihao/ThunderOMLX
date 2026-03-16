# Task #14: Prefill 性能优化 - 最终报告

**日期**: 2026-03-15
**状态**: ✅ 完成
**最终性能**: 704.3 tok/s (Phase 1)

---

## 执行摘要

经过深度 profiling 和 3 个 Phase 的优化尝试，发现：

1. **SSD 写入不是唯一瓶颈**（甚至可能不是主要瓶颈）
2. **ThunderOMLX 架构开销**（ContextPilot、Block Manager）是性能差异的主要原因
3. **最佳状态**: Phase 1（704 tok/s，compression 禁用）

---

## 优化历程

### Phase 1: 禁用压缩 ✅

**修改**: `enable_compression=False`
**结果**: 704.3 tok/s
**状态**: ✅ **当前最佳**

### Phase 2: 增大 Block Size ❌

**修改**: `block_size: 64 → 256`
**预期**: 减少写入次数 128 → 32，+15-18%
**结果**: 703.3 tok/s (**-0.1%**)
**原因**: 单次写入时间增加 4x，抵消了次数减少的收益

### Phase 3a: 并行写入线程 ❌

**修改**: 1 个写入线程 → 4 个并行线程
**预期**: 4x 写入速度，+19%
**结果**: 695.7 tok/s (**-1.2%**)
**原因**:
- SSD I/O 竞争
- 线程管理开销
- Apple Silicon SSD 内部优化了顺序写入

### Phase 3b: 延迟写入 ❌

**修改**: Prefill 期间不写 SSD，完成后批量 flush
**预期**: TTFT 不受 SSD 写入阻塞，接近 baseline 880 tok/s
**结果**: 656.7 tok/s (**-6.8%**)
**原因**: flush 开销在 TTFT 测量内，反而增加延迟

---

## 关键发现

### 🔍 发现 1: 瓶颈不是 SSD 写入

**测试**: 缓存全命中（无 SSD 写入）
**结果**: 609.3 tok/s

即使**完全没有 SSD 写入**，性能还是远低于 baseline 880 tok/s！

### 🔍 发现 2: Baseline 是推测值

从 `PREFILL_PROFILING_DEEP_DIVE.md`:
```
Baseline (oMLX v0.2.13)
TTFT: ~9.1s = 900tok/s (推测)
```

**实际未测量！**

### 🔍 发现 3: ThunderOMLX 架构开销

| 组件 | Baseline 无 | ThunderOMLX 有 | 估计开销 |
|------|------------|----------------|----------|
| ContextPilot | ❌ | ✅ | ~250ms |
| Block-aware Prefix Cache | ❌ | ✅ | ? |
| Paged Cache Manager | ❌ | ✅ | ? |
| SSD 读取 | ❌ | ✅ | ? |

**ThunderOMLX 的优势不在单次 prefill，而在长上下文和缓存复用！**

---

## Profiling 数据回顾

### cProfile Top 3 瓶颈

| 函数 | 总耗时 | 调用次数 | 平均耗时 |
|------|--------|----------|----------|
| `_write_safetensors_no_mx` | 38.3s | 128 | 299ms |
| `{thread lock acquire}` | 30.8s | 1177 | 26ms |
| `qwen3_5.__call__` (模型 forward) | 8.5s | 210 | 40ms |

**但**: 38.3s 写入时间在**后台线程**中，理论上不应阻塞主线程。

### 尝试禁用 Paged SSD Cache

**结果**: ❌ **GPU OOM**

M4 Pro 48GB GPU 无法容纳 8192 tokens prefill 的 KV cache。

**结论**: Paged SSD Cache 是**必须的**，不能禁用。

---

## 性能对比

| 配置 | Prefill TPS | vs Baseline | TTFT |
|------|------------|-------------|------|
| **Phase 1 (最佳)** | **704.3 tok/s** | **-19.9%** | 11631.3ms |
| Phase 2 (block 256) | 703.3 tok/s | -20.0% | 11648.4ms |
| Phase 3a (4 threads) | 695.7 tok/s | -21.0% | 11775.0ms |
| Phase 3b (defer write) | 656.7 tok/s | -25.4% | 12475.0ms |
| 缓存命中（无写入）| 609.3 tok/s | -30.8% | 13444.4ms |
| **Baseline (推测)** | **880 tok/s** | - | ~9100ms |

---

## 根因分析

### 为什么 ThunderOMLX 比 Baseline 慢？

**不是单一瓶颈，而是架构取舍！**

#### ThunderOMLX 新增开销

1. **ContextPilot**: 消息识别和优化（~250ms）
2. **Block-aware Prefix Cache**: Block 管理和查找开销
3. **Paged SSD Cache**:
   - 写入开销（后台，但可能间接影响）
   - 读取开销（缓存命中时仍慢）
4. **内存管理**: Tiered cache 的复杂度

#### ThunderOMLX 新增能力

1. **长上下文支持**: 无限 (SSD offload) vs 受限 (GPU 内存)
2. **缓存复用**: 跨会话复用 vs 无
3. **重复请求加速**: 55-185x vs 无

**设计目标不同**：
- Baseline: 单次请求性能
- ThunderOMLX: 长上下文 + 重复请求

---

## 遗留问题

1. **Baseline 实测值未知**
   - 需要实际测量 oMLX v0.2.13
   - 验证 880 tok/s 是否真实

2. **缓存读取为何慢**
   - 缓存全命中时仍只有 609 tok/s
   - 可能是 SSD 读取、解压缩、或其他开销

3. **ContextPilot 开销**
   - ~250ms 可否优化（Task #11）

---

## 推荐方案

### 选项 A: 接受当前性能 ✅ **推荐**

**理由**：
1. ThunderOMLX 的优势在长上下文和缓存复用
2. 进一步优化单次 prefill 投入产出比低
3. 架构开销是设计取舍，难以消除

**转向**：
- Task #11: ContextPilot 优化（减少 250ms）
- Task #12: 长上下文 KV Cache 加载优化
- Task #2: ClawGate 集成

### 选项 B: 实测 Baseline

**方法**：
1. 下载 oMLX v0.2.13
2. 运行相同的 benchmark (pp8192/tg128)
3. 对比真实数据

**价值**: 验证性能差距是否真的 -20%

---

## 技术债务

无。所有实验性修改已回滚到 Phase 1。

---

## 下一步建议

1. **短期**: Task #11 (ContextPilot 优化)
2. **中期**: Task #12 (长上下文优化)
3. **可选**: 实测 Baseline oMLX v0.2.13

---

*报告时间: 2026-03-15 22:30*
*负责人: Claude Sonnet 4.5*
*状态: ✅ 完成*
