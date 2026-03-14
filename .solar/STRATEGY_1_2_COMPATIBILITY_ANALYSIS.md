# 策略 1 vs 策略 2 兼容性分析

**日期**: 2026-03-14
**分析者**: Solar (战略家 + 治理官)

---

## 📊 策略对比

| 维度 | 策略 1: Prompt Padding | 策略 2: 智能 block_size 选择 |
|------|----------------------|---------------------------|
| **核心思路** | 固定 block_size，调整 prompt | 固定 prompt，调整 block_size |
| **实现位置** | scheduler.py:2315-2346 | 待实现（需新增方法） |
| **输入** | `paged_cache_block_size`（固定） | `adaptive_block_size_candidates`（多个） |
| **输出** | 填充后的 prompt tokens | 选择的 block_size |
| **修改对象** | `request.prompt_token_ids` | `self.config.paged_cache_block_size`（动态） |

---

## ⚠️ 冲突点分析

### 冲突 1: 执行顺序问题

**问题**: 两个策略都需要在 tokenize 之后执行，但执行顺序会影响结果。

```python
# 当前代码（策略1已实施）
if request.prompt_token_ids is None:
    request.prompt_token_ids = self.tokenizer.encode(request.prompt)  # Step 1: Tokenize

# ⚡ 策略 1: Padding（基于固定 block_size）
block_size = self.config.paged_cache_block_size  # 读取固定值
padding_needed = block_size - (len(tokens) % block_size)

# ⚡ 策略 2: 智能选择（需要在策略1之前执行）
block_size = select_optimal_block_size(len(tokens))  # 动态选择
```

**冲突**:
- 如果先执行策略1（padding），然后执行策略2（选择 block_size），则策略2会基于已 padding 的长度选择 block_size，可能不是最优
- 如果先执行策略2（选择 block_size），然后执行策略1（padding），则策略1会基于动态的 block_size 计算 padding，这是合理的

**结论**: **必须先执行策略2，再执行策略1**

---

### 冲突 2: 缓存隔离问题

**问题**: 不同 prompt 使用不同 block_size，会导致缓存无法跨请求复用。

**示例**:
```
Request 1: 116 tokens → 选择 block_size=16 → 缓存 key 基于 block_size=16
Request 2: 200 tokens → 选择 block_size=32 → 缓存 key 基于 block_size=32
Request 3: 116 tokens (重复) → 选择 block_size=16 → 缓存命中 ✅
Request 4: 116 tokens (重复) → 如果配置变化，可能选择 block_size=32 → 缓存未命中 ❌
```

**根因**: PagedSSDCacheManager 的缓存 key 包含 `block_size`，不同 block_size 的缓存无法共享。

```python
# paged_ssd_cache.py 中的 key 生成
def _get_block_file_path(self, tokens_hash: str, block_idx: int) -> Path:
    # 缓存 key 依赖 block_size
    key = f"{tokens_hash}_{self.block_size}_{block_idx}"
    return self.cache_dir / f"{key}.npy"
```

**影响**:
- ❌ 跨请求缓存复用率下降
- ❌ 同一个 prompt 模板在不同场景下可能使用不同 block_size，无法复用缓存
- ❌ Agent 高重复场景（system prompt）收益降低

**结论**: **策略2会破坏跨请求缓存复用**

---

### 冲突 3: 配置管理问题

**问题**: `paged_cache_block_size` 是全局配置，策略2需要动态修改。

**当前架构**:
```python
# scheduler.py
class SchedulerConfig:
    paged_cache_block_size: int = 32  # 全局固定值

# 策略1依赖这个固定值
block_size = self.config.paged_cache_block_size
```

**策略2需要**:
```python
# 动态修改（per-request）
def _select_block_size_for_prompt(self, prompt_length: int) -> int:
    # 返回最优 block_size
    return optimal_block_size

# 问题：如何应用到当前请求？
# 选项1: 修改全局配置（影响后续所有请求）❌
# 选项2: 传递局部 block_size（需要修改所有相关方法）⚠️
```

**结论**: **策略2需要重构 block_size 传递机制**

---

## ✅ 兼容性方案

### 方案 A: 顺序执行（推荐 ⭐⭐⭐⭐）

**原理**: 先执行策略2选择 block_size，再执行策略1 padding

**实现**:
```python
# scheduler.py:2315 之前插入策略2
if request.prompt_token_ids is None:
    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
    request.num_prompt_tokens = len(request.prompt_token_ids)

# ⚡ 策略 2: 智能选择 block_size（新增）
if self.config.enable_adaptive_block_size and self.block_aware_cache is not None:
    optimal_block_size = self._select_block_size_for_prompt(
        len(request.prompt_token_ids)
    )

    # 临时覆盖 block_size（仅用于当前请求）
    original_block_size = self.config.paged_cache_block_size
    self.config.paged_cache_block_size = optimal_block_size

    logger.info(
        f"⚡ Adaptive Block Size: selected {optimal_block_size} "
        f"for {len(request.prompt_token_ids)} tokens"
    )

# ⚡ 策略 1: Prompt Padding（现有代码，基于动态 block_size）
if self.config.enable_prompt_padding and self.block_aware_cache is not None:
    block_size = self.config.paged_cache_block_size  # 使用策略2选择的值
    # ... padding 逻辑 ...
```

**优点**:
- ✅ 最小化代码修改（复用策略1的 padding 逻辑）
- ✅ 策略1自动基于策略2选择的 block_size 计算 padding

**缺点**:
- ⚠️ 仍然存在缓存隔离问题（不同请求可能用不同 block_size）
- ⚠️ 需要在请求结束后恢复 `original_block_size`（避免污染后续请求）

---

### 方案 B: 局部 block_size（复杂 ⭐⭐⭐）

**原理**: 每个请求携带自己的 block_size，不修改全局配置

**实现**:
```python
# 在 Request 类中新增字段
class Request:
    # ...
    effective_block_size: Optional[int] = None  # 当前请求使用的 block_size

# 策略2选择 block_size
if self.config.enable_adaptive_block_size:
    request.effective_block_size = self._select_block_size_for_prompt(
        len(request.prompt_token_ids)
    )

# 策略1使用局部 block_size
if self.config.enable_prompt_padding:
    block_size = request.effective_block_size or self.config.paged_cache_block_size
    # ... padding 逻辑 ...
```

**优点**:
- ✅ 不污染全局配置
- ✅ 更清晰的架构（每个请求独立）

**缺点**:
- ❌ 需要修改 Request 类
- ❌ 需要修改所有读取 `paged_cache_block_size` 的地方（传递 `request.effective_block_size`）
- ❌ 工作量大，风险高

---

### 方案 C: 只用策略1或策略2（互斥 ⭐⭐⭐⭐⭐）

**原理**: 两个策略不同时启用，用户二选一

**配置**:
```python
class SchedulerConfig:
    # 策略1配置
    enable_prompt_padding: bool = False
    max_padding_tokens: int = 64

    # 策略2配置
    enable_adaptive_block_size: bool = False
    adaptive_block_size_candidates: List[int] = [8, 16, 32, 64, 128, 256]

# 互斥检查
if enable_prompt_padding and enable_adaptive_block_size:
    raise ValueError("Cannot enable both prompt_padding and adaptive_block_size")
```

**实现**:
```python
# 策略1: 固定 block_size + Padding
if self.config.enable_prompt_padding and not self.config.enable_adaptive_block_size:
    # ... padding 逻辑 ...

# 策略2: 动态 block_size（无 padding）
if self.config.enable_adaptive_block_size and not self.config.enable_prompt_padding:
    # ... 选择 block_size 逻辑 ...
```

**优点**:
- ✅ 简单直接，避免冲突
- ✅ 用户明确知道使用哪个策略
- ✅ 无缓存隔离问题（策略1固定 block_size，策略2虽然动态但只在该策略下）

**缺点**:
- ❌ 无法同时享受两个策略的优势
- ❌ 需要用户手动选择

---

### 方案 D: 策略4混合（推荐 ⭐⭐⭐⭐⭐）

**原理**: 结合策略2的智能选择 + 策略1的 padding，但**固定 block_size 池**

**关键改进**: 限制 block_size 候选池，减少缓存隔离

**实现**:
```python
# 配置：只允许少数几个 block_size
class SchedulerConfig:
    enable_mixed_strategy: bool = False
    mixed_strategy_block_sizes: List[int] = [16, 32, 64]  # 限制为3个
    max_padding_tokens: int = 32

# 混合策略逻辑
if self.config.enable_mixed_strategy:
    prompt_length = len(request.prompt_token_ids)

    # Step 1: 智能选择 block_size（从有限池中选）
    best_block_size = 32  # 默认
    best_hit_ratio = 0.0

    for block_size in self.config.mixed_strategy_block_sizes:
        hit_ratio = (prompt_length // block_size) * block_size / prompt_length
        if hit_ratio >= 0.90 and hit_ratio > best_hit_ratio:
            best_block_size = block_size
            best_hit_ratio = hit_ratio

    # Step 2: 如果仍然 < 90%，考虑 padding
    if best_hit_ratio < 0.90:
        remainder = prompt_length % best_block_size
        padding_needed = best_block_size - remainder

        if padding_needed <= self.config.max_padding_tokens:
            # Padding 到边界
            pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            request.prompt_token_ids = list(request.prompt_token_ids) + [pad_token] * padding_needed
            best_hit_ratio = 1.0

    # Step 3: 应用选择的 block_size
    self.config.paged_cache_block_size = best_block_size

    logger.info(
        f"⚡ Mixed Strategy: block_size={best_block_size}, "
        f"hit_ratio={best_hit_ratio:.1%}, "
        f"prompt_length={len(request.prompt_token_ids)}"
    )
```

**优点**:
- ✅ 结合两个策略的优势
- ✅ 限制 block_size 池（如3个）减少缓存隔离影响
- ✅ 自动平衡（优先选择 block_size，必要时 padding）

**缺点**:
- ⚠️ 仍然存在一定的缓存隔离（但影响降低到 1/3）
- ⚠️ 需要权衡池的大小（越小缓存复用越好，越大优化空间越大）

---

## 📊 方案对比

| 方案 | 兼容性 | 性能 | 缓存复用 | 实现难度 | 推荐度 |
|------|--------|------|----------|---------|--------|
| **方案 A: 顺序执行** | ⚠️ 部分兼容 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 方案 B: 局部 block_size | ✅ 完全兼容 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **方案 C: 互斥** | ✅ 无冲突 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **方案 D: 混合策略** | ✅ 完全兼容 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 💡 推荐方案

### 短期（立即可用）: **方案 C - 互斥**

**配置**:
```python
# 用户选择策略1（推荐，Agent 高重复场景）
scheduler_config = SchedulerConfig(
    paged_cache_block_size=32,
    enable_prompt_padding=True,  # ⚡ 启用策略1
    max_padding_tokens=32,
    enable_adaptive_block_size=False,  # 禁用策略2
)

# 用户选择策略2（多样化 prompt 场景）
scheduler_config = SchedulerConfig(
    paged_cache_block_size=32,  # 默认值（被动态覆盖）
    enable_prompt_padding=False,  # 禁用策略1
    enable_adaptive_block_size=True,  # ⚡ 启用策略2
    adaptive_block_size_candidates=[8, 16, 32, 64, 128, 256],
)
```

**理由**:
- ✅ 简单直接，零冲突
- ✅ 缓存复用率高（策略1固定 block_size）
- ✅ 实现成本低（只需添加互斥检查）
- ✅ 用户可根据场景自主选择

---

### 中期（优化版）: **方案 D - 混合策略**

**配置**:
```python
# 混合策略：有限池 + 智能选择 + Padding
scheduler_config = SchedulerConfig(
    enable_mixed_strategy=True,
    mixed_strategy_block_sizes=[16, 32, 64],  # 限制为3个
    max_padding_tokens=32,
)
```

**理由**:
- ✅ 结合两个策略的优势（智能选择 + Padding）
- ✅ 缓存复用率较高（有限池降低隔离）
- ✅ 自适应优化（自动选择最优策略）
- ⚠️ 实现成本中等（需要新增混合逻辑）

---

## 🎯 实施建议

### 阶段 1: 互斥模式（1 小时）

1. ✅ 添加互斥检查（enable_prompt_padding XOR enable_adaptive_block_size）
2. ✅ 实现策略2（智能 block_size 选择）
3. ✅ 测试两个策略独立运行

### 阶段 2: 混合策略（2 小时）

1. ✅ 实现混合策略逻辑（有限池 + 智能选择 + Padding）
2. ✅ 测试不同 prompt 长度下的表现
3. ✅ 性能基准测试（对比策略1、策略2、混合策略）

---

## 📋 决策矩阵

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| **Agent 高重复场景** | 策略1（Padding） | 固定 block_size，缓存复用率高 |
| **多样化 prompt** | 策略2（智能选择） | 动态优化，每个 prompt 都最优 |
| **混合场景** | 混合策略 | 自适应，兼顾复用和优化 |

---

## ⚠️ 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **缓存隔离** | 策略2跨请求缓存复用下降 | 使用有限池（方案D）或固定 block_size（方案C） |
| **配置冲突** | 用户同时启用两个策略导致未定义行为 | 添加互斥检查，抛出错误 |
| **性能回退** | 动态选择 block_size 增加计算开销 | 计算量极小（< 1ms），可忽略 |

---

## 📊 总结

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   策略1 vs 策略2 兼容性                                     │
│                                                             │
│   冲突点:                                                   │
│   ❌ 执行顺序问题（必须先策略2后策略1）                     │
│   ❌ 缓存隔离问题（不同 block_size 无法复用缓存）           │
│   ❌ 配置管理问题（全局 vs 局部 block_size）                │
│                                                             │
│   推荐方案:                                                 │
│   ✅ 短期: 方案C（互斥）- 简单直接                          │
│   ✅ 中期: 方案D（混合策略）- 最优性能                      │
│                                                             │
│   核心原则:                                                 │
│   • 策略1: 固定 block_size，适合高重复场景                 │
│   • 策略2: 动态 block_size，适合多样化场景                 │
│   • 不建议同时启用（除非使用混合策略）                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**签署**: Solar (战略家 + 治理官双签)
**日期**: 2026-03-14
**下一步**: 决定采用方案C（互斥）还是方案D（混合策略）
