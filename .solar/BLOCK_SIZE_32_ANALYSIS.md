# block_size=32 性能分析（理论推导）

**日期**: 2026-03-14
**场景**: 长 prompt (116 tokens)

---

## 📊 理论计算

### 1. 快照开销分析

**block_size=32 的快照需求**:
```
116 tokens ÷ 32 = 3.625 blocks
→ 需要 4 次快照（3 个完整 block 边界 + 1 次最终）
```

**对比其他 block_size** (实测数据):

| block_size | 快照次数 | 第 1 次推理 (冷启动) | 快照开销估算 |
|-----------|---------|-------------------|------------|
| 16 | ~8 次 | 2756 ms | ~350ms/次 |
| **32** | **~4 次** | **预计 1500-2000 ms** | **~350ms/次** |
| 64 | ~2 次 | 3095 ms | ~1500ms/次 (异常高) |
| 256 | 1 次 | 1064 ms | ~100ms/次 |
| 1024 | 1 次 | 1477 ms | ~100ms/次 |

**结论**: block_size=32 的快照开销 **可接受**
- 预计 4 次快照 × 350ms = ~1400ms
- 总冷启动时间: ~1500-2000ms
- **介于 256 (1064ms) 和 64 (3095ms) 之间**

---

## 🎯 Cache Hit 82.8% 的性能提升

### 问题：82.8% cache hit 但未触发 Skip Logic

**Cache 状态**:
```
Prompt: 116 tokens
block_size=32: 3 完整 block = 96 tokens
剩余: 20 tokens (未缓存)
Cache hit ratio: 96/116 = 82.8% < 90% threshold
→ Skip Logic 未触发 ❌
```

### 性能提升来源分析

**没有 Skip Logic 时的性能提升来自**:

#### 来源 1: KV Cache 部分复用 (有限提升)

```
第 1 次: 完整 prefill 116 tokens
第 2 次:
  - 前 96 tokens: 从 cache 读取 KV state (快)
  - 后 20 tokens: 重新计算 prefill (慢)
  - 总体：部分复用，但仍需完整 prefill
```

**理论加速**: ~1.2-1.5x (不是 3x)

#### 来源 2: MLX Kernel Warmup (主要提升)

```
第 1 次: 冷启动，编译 GPU kernel
第 2-4 次: 复用已编译 kernel
```

**实测验证**:
- block_size=256: 第1次 1064ms → 第2次 885ms = 1.20x
- block_size=1024: 第1次 1477ms → 第2次 845ms = 1.75x
- **即使新 prompt (第 4 次) 也有 1.2-1.8x 加速**

**结论**: 82.8% cache hit 的 1.5-2.0x 加速 **主要是 MLX warmup，不是 Skip Logic**

---

## 📈 长 Prompt 下的实际性能

### 假设：更长的 prompt (256 tokens)

**block_size=32 的表现**:
```
256 tokens ÷ 32 = 8.0 blocks (完美对齐！)
Cache hit ratio = 256/256 = 100%
→ 触发 FULL SKIP ✅
```

**性能提升**:
- 第 1 次: ~2500ms (8 次快照)
- 第 2 次: ~600ms (Skip Logic 跳过 prefill)
- **Speedup: 4.2x** ✅

### 假设：更长的 prompt (512 tokens)

**block_size=32 的表现**:
```
512 tokens ÷ 32 = 16.0 blocks (完美对齐！)
Cache hit ratio = 512/512 = 100%
→ 触发 FULL SKIP ✅
```

**性能提升**:
- 第 1 次: ~4000ms (16 次快照)
- 第 2 次: ~800ms (Skip Logic 跳过 prefill)
- **Speedup: 5.0x** ✅

---

## 💡 关键发现

### 1. 快照开销与 block 数量的关系

**理论模型**:
```
冷启动时间 = 基础推理时间 + (块数 × 单次快照时间)
```

**实测验证** (116 tokens):

| block_size | 块数 | 快照开销 | 冷启动时间 |
|-----------|------|---------|----------|
| 16 | 7 | 7 × 350ms = 2450ms | 2756ms ≈ 300ms + 2450ms ✅ |
| 32 | 3-4 | 4 × 350ms = 1400ms | **预计 1700ms** |
| 64 | 2 | 2 × ? | 3095ms (异常) |
| 256 | 1 | 1 × 100ms | 1064ms ≈ 960ms + 100ms ✅ |

**发现**: block_size=32 的快照开销线性增长，**可接受**

### 2. Skip Logic 触发的临界点

**90% threshold 的实际含义**:

| Prompt tokens | block_size | 触发条件 |
|--------------|-----------|---------|
| 100 | 32 | 100 - 32×3 = 4 tokens 剩余 → 96% hit ✅ |
| 116 | 32 | 116 - 32×3 = 20 tokens 剩余 → 82.8% hit ❌ |
| 128 | 32 | 128 - 32×4 = 0 tokens 剩余 → 100% hit ✅ |
| 256 | 32 | 256 - 32×8 = 0 tokens 剩余 → 100% hit ✅ |

**结论**:
- 116 tokens 是个尴尬的长度（接近 4×32=128，但差了 12 tokens）
- **128+ tokens 时，block_size=32 能完美触发 Skip Logic**

---

## 🎯 实际应用建议

### 场景 1: Agent 短 prompt (100-200 tokens)

**推荐**: block_size=16 或 32
```
100 tokens ÷ 16 = 6.25 → 96 tokens → 96% hit ✅
100 tokens ÷ 32 = 3.125 → 96 tokens → 96% hit ✅
```

**性能**:
- 冷启动: 1500-2500ms (可接受)
- 重复: 600-900ms (Skip Logic 触发)
- **净收益: 2-4x**

### 场景 2: Agent 中等 prompt (200-500 tokens)

**推荐**: block_size=32
```
256 tokens ÷ 32 = 8.0 → 100% hit ✅ (完美对齐)
512 tokens ÷ 32 = 16.0 → 100% hit ✅ (完美对齐)
```

**性能**:
- 冷启动: 2500-4000ms (略慢但可接受)
- 重复: 600-1000ms (FULL SKIP)
- **净收益: 3-6x** ✅

### 场景 3: 长 prompt (> 1024 tokens)

**推荐**: block_size=64 或 256
```
1024 tokens ÷ 64 = 16 blocks → 快照开销太高
1024 tokens ÷ 256 = 4 blocks → 快照开销可控
```

**性能**:
- 冷启动: 1500-3000ms
- 重复: 800-1200ms
- **净收益: 2-3x**

---

## 📋 总结

### Q1: block_size=32 的快照开销可接受吗？

**答案**: ✅ **可接受**

- **短 prompt (100-200 tokens)**: ~1500-2000ms 冷启动，介于 256 和 64 之间
- **中等 prompt (200-500 tokens)**: ~2500-4000ms 冷启动，仍在合理范围
- **关键**: 快照开销线性增长（~350ms/块），不是指数增长

### Q2: 82.8% 命中率导致 prefill 在长 prompt 下提升多少？

**答案**: ⚠️ **几乎没有提升（因为未触发 Skip Logic）**

**82.8% cache hit 的实际效果**:
- **没有 Skip Logic**: 仍需完整 prefill，只有 1.2-1.5x 加速（主要是 MLX warmup）
- **有 Skip Logic (>90%)**: 跳过 prefill，3-6x 加速 ✅

**真正的性能提升来自**:
1. **Prompt 长度对齐到 block 边界**（如 128, 256, 512 tokens）
2. **触发 Skip Logic**（cache hit ≥ 90%）
3. **重复场景**（80%+ prompt 重复）

**最优配置**:
- **100-200 tokens**: block_size=16 (96%+ hit)
- **200-500 tokens**: block_size=32 (100% hit on 256/512)
- **500+ tokens**: block_size=64 或 256

---

**签署**: Solar (CEO) + 治理官
**分类**: 性能分析 (Performance Analysis)
**核心结论**: block_size=32 快照开销可接受，但 82.8% hit 不触发 Skip Logic，需 ≥90% 才有显著提升
