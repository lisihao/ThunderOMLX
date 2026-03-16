# Prefill Step Size 对性能的影响分析

**日期**: 2026-03-15 23:45
**任务**: Task #14 - P1 优化 Prefill 性能
**发现**: prefill_step_size=2048 可能导致 prefill 慢 20.8%

---

## 🔍 关键发现

### 当前配置

```python
# src/omlx/scheduler.py:947
prefill_step_size: int = 2048
```

### 性能数据

| 系统 | Prefill TPS | Generation TPS |
|------|-------------|----------------|
| **ThunderOMLX** | 696.7 tok/s | 79.5 tok/s |
| **oMLX v0.2.13** | 880.3 tok/s | 71.3 tok/s |
| **差距** | -20.8% | +11.5% |

---

## 🤔 问题分析

### prefill_step_size 的作用

`prefill_step_size` 控制每次 prefill 处理的 token 数量（分块大小）。

**当前行为**（prefill_step_size=2048）：
```
8192 tokens prefill = 4 chunks
├─ Chunk 1: tokens [0:2048]     → forward pass 1
├─ Chunk 2: tokens [2048:4096]  → forward pass 2
├─ Chunk 3: tokens [4096:6144]  → forward pass 3
└─ Chunk 4: tokens [6144:8192]  → forward pass 4
     ↓
   4 次 forward pass + 3 次 KV cache merge
```

**如果 prefill_step_size=8192**：
```
8192 tokens prefill = 1 chunk
└─ Chunk 1: tokens [0:8192]  → forward pass 1
     ↓
   1 次 forward pass，无 merge
```

---

## 📊 性能影响

### Forward Pass 开销

每次 forward pass 除了计算本身，还有：
1. **Metal 同步点**：CPU → GPU 数据传输
2. **KV Cache 管理**：分配、拷贝、释放
3. **MLX eval() 调用**：触发 Metal kernel 执行

**4 次 forward pass vs 1 次**：
- 额外的 3 次同步开销
- 额外的 3 次 KV cache merge
- 可能降低 Metal kernel 批处理效率

### KV Cache Merge 开销

MLX 的 `_merge_caches()` 函数需要：
1. 拷贝现有 KV cache
2. Concatenate 新的 KV cache
3. 更新 cache 指针

**3 次 merge 开销**：
- 每次 merge 处理 ~40 layers × 2 (K, V)
- 需要 CPU-GPU 同步
- 内存分配/释放

---

## 🎯 假设验证

### 假设

**prefill_step_size=2048 导致 prefill 慢 20.8%**

### 验证方案

**实验 1**: 修改 prefill_step_size 为 8192（或更大）
```python
# src/omlx/scheduler.py:947
prefill_step_size: int = 8192  # 从 2048 改为 8192
```

**预期结果**：
- Prefill TPS: 696.7 → 880+ tok/s (+26.3%)
- 无副作用（单次 forward pass 内存足够）

**实验 2**: 对比不同 prefill_step_size
- 2048（当前）
- 4096
- 8192
- 16384

**测量**：
- Prefill TPS
- 内存峰值
- TTFT

---

## 🔧 优化方案

### 方案 1: 增大 prefill_step_size（推荐）⭐⭐⭐⭐⭐

**修改**:
```python
# src/omlx/scheduler.py:947
prefill_step_size: int = 8192  # 从 2048 改为 8192（或更大）
```

**优点**：
- ✅ 简单（修改 1 行代码）
- ✅ 立即生效
- ✅ 预期提升 26.3%
- ✅ 无副作用（M4 Pro 48GB 内存足够）

**缺点**：
- ⚠️ 对于超长 prompt（>8192 tokens）需要更大的值
- ⚠️ 可能增加内存峰值（对于大模型）

**时间**：5 分钟

---

### 方案 2: 动态 prefill_step_size（最优）⭐⭐⭐⭐

**设计**：根据 prompt 长度动态选择 prefill_step_size
```python
def _get_optimal_prefill_step_size(prompt_length: int) -> int:
    """根据 prompt 长度返回最优的 prefill_step_size"""
    if prompt_length <= 2048:
        return 2048
    elif prompt_length <= 4096:
        return 4096
    elif prompt_length <= 8192:
        return 8192
    elif prompt_length <= 16384:
        return 16384
    else:
        return 32768  # 超长 prompt
```

**优点**：
- ✅ 自适应（短 prompt 不浪费，长 prompt 不慢）
- ✅ 最优性能
- ✅ 可配置

**缺点**：
- ⚠️ 需要修改更多代码
- ⚠️ 需要测试验证

**时间**：1-2 小时

---

### 方案 3: 禁用分块（激进）⭐⭐

**设计**：将 prefill_step_size 设置为极大值（如 65536）
```python
prefill_step_size: int = 65536  # 实际上禁用分块
```

**优点**：
- ✅ 最大化 prefill 性能
- ✅ 最简单

**缺点**：
- ❌ 超长 prompt 可能 OOM
- ❌ 不灵活

**不推荐**：可能导致内存问题

---

## 📈 预期效果

### 方案 1（prefill_step_size=8192）

**当前**:
- Prefill: 696.7 tok/s (8192 tokens / 11.76s)
- 4 chunks × 2.94s/chunk = 11.76s

**优化后**:
- Prefill: 880+ tok/s (8192 tokens / 9.3s)
- 1 chunk × 9.3s = 9.3s
- **提升**: +26.3% ✅

**TTFT 改进**:
- 当前: 11454.3ms
- 优化后: ~9300ms
- **提升**: -18.8% ✅

---

## 🎯 立即执行

**推荐**: 方案 1（修改 prefill_step_size=8192）

**步骤**：
1. 修改 `src/omlx/scheduler.py:947`
2. 重新运行 benchmark
3. 验证 Prefill TPS ≥ 880 tok/s
4. 检查内存峰值（应该正常）

**预计时间**: 10 分钟（修改 + 测试）

---

*分析时间: 2026-03-15 23:50*
*负责人: Claude Sonnet 4.5*
