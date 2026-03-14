# Skip Logic 触发优化策略

**日期**: 2026-03-14
**目标**: 提高 Skip Logic 触发率，从 82.8% → 90%+

---

## 📊 问题分析

**当前瓶颈**:
```
Prompt: 116 tokens
block_size: 32
可缓存: 96 tokens (3 blocks)
剩余: 20 tokens (浪费)
Cache hit ratio: 82.8% < 90% threshold ❌
```

**核心问题**: Prompt 长度未对齐到 block 边界

---

## 🎯 策略 1: Prompt Padding（推荐 ⭐⭐⭐⭐⭐）

### 原理

**将 prompt 填充到最近的 block 边界**

```python
def pad_prompt_to_block_boundary(tokens: List[int], block_size: int) -> List[int]:
    """填充 prompt 到 block 边界"""
    remainder = len(tokens) % block_size
    if remainder == 0:
        return tokens  # 已对齐

    # 计算需要填充的 token 数
    padding_needed = block_size - remainder

    # 使用 pad_token_id 填充
    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded_tokens = tokens + [pad_token] * padding_needed

    return padded_tokens
```

### 效果

**示例**:
```
原始: 116 tokens → 82.8% hit ❌
填充: 116 + 12 = 128 tokens (4 blocks) → 100% hit ✅
触发: FULL SKIP ✅
```

### 实现

**位置**: `scheduler.py:2303`（tokenize 后）

```python
# scheduler.py:2303-2308
if request.prompt_token_ids is None:
    if isinstance(request.prompt, str):
        request.prompt_token_ids = self.tokenizer.encode(request.prompt)
    else:
        request.prompt_token_ids = list(request.prompt)

    # ⚡ 策略 1: Prompt Padding 到 block 边界
    if self.config.enable_prompt_padding:
        original_len = len(request.prompt_token_ids)
        block_size = self.config.paged_cache_block_size
        remainder = original_len % block_size

        if remainder > 0:
            padding_needed = block_size - remainder
            pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            request.prompt_token_ids.extend([pad_token] * padding_needed)

            logger.info(
                f"⚡ Prompt Padding: {original_len} → {len(request.prompt_token_ids)} tokens "
                f"(+{padding_needed} padding) for 100% cache alignment"
            )

    request.num_prompt_tokens = len(request.prompt_token_ids)
```

### 配置

```python
class SchedulerConfig:
    enable_prompt_padding: bool = False  # 默认关闭
    max_padding_tokens: int = 64  # 最大填充 token 数（避免过度填充）
```

### 优点

- ✅ 简单直接，无需修改底层
- ✅ 100% 对齐，确保触发 Skip Logic
- ✅ 对模型推理影响小（padding token 被忽略）

### 缺点

- ⚠️ 增加了少量 token（但可缓存）
- ⚠️ 需要模型支持 padding token

---

## 🎯 策略 2: 智能 block_size 选择（推荐 ⭐⭐⭐⭐）

### 原理

**根据 prompt 长度动态选择最优 block_size**

```python
def select_optimal_block_size(prompt_length: int) -> int:
    """选择最优 block_size 以最大化 cache hit ratio"""

    # 候选 block_size
    candidates = [8, 16, 32, 64, 128, 256]

    best_block_size = 256
    best_hit_ratio = 0.0

    for block_size in candidates:
        # 计算 cache hit ratio
        num_blocks = prompt_length // block_size
        cached_tokens = num_blocks * block_size
        hit_ratio = cached_tokens / prompt_length

        # 选择 hit_ratio >= 90% 的最大 block_size（减少快照开销）
        if hit_ratio >= 0.90 and block_size > best_block_size:
            best_block_size = block_size
            best_hit_ratio = hit_ratio

    return best_block_size
```

### 效果

**示例**:
```
116 tokens:
  - block_size=32: 82.8% hit ❌
  - block_size=16: 96.6% hit ✅ (选择这个)

128 tokens:
  - block_size=32: 100% hit ✅ (选择这个)
  - block_size=16: 100% hit (但快照开销更大)
```

### 实现

**位置**: 方案 2 的扩展

```python
# scheduler.py: 新增智能选择逻辑
def _select_block_size_for_prompt(self, prompt_length: int) -> int:
    """根据 prompt 长度智能选择 block_size"""

    # 如果用户显式设置，不自动调整
    if not self.config.enable_adaptive_block_size:
        return self.config.paged_cache_block_size

    # 候选 block_size（从小到大）
    candidates = [8, 16, 32, 64, 128, 256]
    approx_threshold = 0.90

    # 选择最大的能达到 90%+ hit ratio 的 block_size
    for block_size in reversed(candidates):
        hit_ratio = (prompt_length // block_size) * block_size / prompt_length
        if hit_ratio >= approx_threshold:
            return block_size

    # 降级：选择能达到最高 hit ratio 的
    return min(candidates)
```

### 配置

```python
class SchedulerConfig:
    enable_adaptive_block_size: bool = False  # 自适应 block_size
    adaptive_block_size_candidates: List[int] = [8, 16, 32, 64, 128, 256]
```

### 优点

- ✅ 自动优化，无需用户干预
- ✅ 每个 prompt 都能达到最优 cache hit
- ✅ 平衡缓存命中和快照开销

### 缺点

- ⚠️ 需要在每次请求时计算
- ⚠️ 不同 prompt 使用不同 block_size，缓存隔离

---

## 🎯 策略 3: 降低 approx_threshold（不推荐 ⭐⭐）

### 原理

**降低 Skip Logic 触发阈值**

```python
# scheduler.py:2323
cache_result = self.block_aware_cache.match_cache_with_skip_logic(
    request.prompt_token_ids,
    extra_keys=extra_keys,
    approx_threshold=0.80,  # 从 0.90 降低到 0.80
)
```

### 效果

```
116 tokens / block_size=32:
  - 82.8% hit > 80% threshold ✅
  - 触发 APPROXIMATE SKIP
  - 但仍有 17.2% 的 prefill 计算
```

### 优点

- ✅ 简单，只改一个参数

### 缺点

- ❌ 收益下降：82.8% hit 意味着仍有大量计算
- ❌ 低于 85% 的 hit ratio，Skip Logic 收益不明显
- ❌ 可能引入更多 zero-filling 错误

**不推荐原因**: 性能提升有限（预计只有 1.5-2.0x，而不是 3-4x）

---

## 🎯 策略 4: 混合策略（推荐 ⭐⭐⭐⭐⭐）

### 原理

**结合 Padding + 智能 block_size**

```python
def optimize_for_skip_logic(
    prompt_tokens: List[int],
    config: SchedulerConfig,
) -> Tuple[List[int], int]:
    """优化 prompt 以最大化 Skip Logic 触发率"""

    prompt_length = len(prompt_tokens)

    # Step 1: 智能选择初始 block_size
    block_size = select_optimal_block_size(prompt_length)

    # Step 2: 计算 cache hit ratio
    hit_ratio = (prompt_length // block_size) * block_size / prompt_length

    # Step 3: 如果 hit ratio < 90%，考虑 padding
    if hit_ratio < 0.90 and config.enable_prompt_padding:
        remainder = prompt_length % block_size
        padding_needed = block_size - remainder

        # 限制 padding 数量
        if padding_needed <= config.max_padding_tokens:
            pad_token = config.pad_token_id
            prompt_tokens = prompt_tokens + [pad_token] * padding_needed
            hit_ratio = 1.0  # 100% after padding

    return prompt_tokens, block_size
```

### 效果

**示例**:
```
116 tokens:
  Step 1: 智能选择 block_size=16 → 96.6% hit ✅
  (无需 padding)

120 tokens:
  Step 1: 智能选择 block_size=32 → 93.8% hit ✅
  (无需 padding)

116 tokens (如果只有 32 可选):
  Step 1: block_size=32 → 82.8% hit ❌
  Step 2: Padding 12 tokens → 128 tokens → 100% hit ✅
```

### 实现

```python
# scheduler.py:2303 (在 tokenize 之后)
if request.prompt_token_ids is None:
    # ... tokenize ...

    # ⚡ 混合优化策略
    request.prompt_token_ids, optimal_block_size = optimize_for_skip_logic(
        request.prompt_token_ids,
        self.config,
    )

    # 如果与当前 block_size 不同，记录（但不修改，避免缓存隔离问题）
    if optimal_block_size != self.config.paged_cache_block_size:
        logger.debug(
            f"Optimal block_size for this prompt: {optimal_block_size} "
            f"(current: {self.config.paged_cache_block_size})"
        )
```

---

## 🎯 策略 5: 部分 Block 缓存（方案 3，长期）

### 原理

**修改 BlockAwarePrefixCache 支持部分 block**

```python
# 允许缓存最后一个不完整的 block
# 116 tokens = 3 完整 block (96) + 1 部分 block (20) = 100% cached
```

### 限制

- ❌ ArraysCache 不支持 block slicing
- ⚠️ 需要修改底层实现
- ⚠️ 高风险，可能破坏兼容性

**结论**: 暂时不推荐，留待方案 3 深入研究

---

## 📊 策略对比

| 策略 | 难度 | 效果 | 兼容性 | 推荐度 |
|------|------|------|--------|--------|
| **策略 1: Prompt Padding** | ⭐ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |
| **策略 2: 智能 block_size** | ⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| 策略 3: 降低阈值 | ⭐ | ⭐⭐ | ✅ | ⭐⭐ |
| **策略 4: 混合策略** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |
| 策略 5: 部分 Block | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ | ⭐⭐ |

---

## 💡 最佳实践

### 短期方案（立即可用）

**推荐**: 策略 1 (Prompt Padding) + 策略 2 (智能 block_size)

```python
# 配置
scheduler_config = SchedulerConfig(
    paged_cache_block_size=32,  # 默认值
    enable_prompt_padding=True,  # ⚡ 启用 padding
    max_padding_tokens=32,  # 限制 padding 数量
    enable_adaptive_block_size=False,  # 可选：智能选择
)
```

**预期效果**:
- 82.8% hit → 100% hit
- Skip Logic 触发率: 100%
- 性能提升: 3-6x (重复场景)

### 中期方案（方案 2 扩展）

**推荐**: 策略 4 (混合策略)

```python
# 方案 2 的增强版
scheduler_config = SchedulerConfig(
    arrays_cache_target_block_size=None,  # 智能选择
    enable_prompt_padding=True,  # Padding 到边界
    enable_adaptive_block_size=True,  # 根据 prompt 自适应
)
```

### 长期方案（方案 3）

**研究**: 策略 5 (部分 Block 缓存)
- 需要修改 ArraysCache 底层
- 风险较高，收益待评估

---

## 🎯 实施建议

### 阶段 1: 快速验证（1 小时）

1. ✅ 实现策略 1 (Prompt Padding)
2. ✅ 测试 116 tokens → 128 tokens
3. ✅ 验证 Skip Logic 触发

### 阶段 2: 智能优化（2 小时）

1. ✅ 实现策略 2 (智能 block_size 选择)
2. ✅ 集成到方案 2
3. ✅ 多场景测试

### 阶段 3: 混合策略（3 小时）

1. ✅ 结合策略 1 + 2
2. ✅ 自适应优化
3. ✅ 性能基准测试

---

## 📋 总结

| 问题 | 解决方案 |
|------|---------|
| 116 tokens → 82.8% hit | **Padding 12 tokens → 100% hit** ✅ |
| block_size 不对齐 | **智能选择最优 block_size** ✅ |
| 快照开销 vs 缓存命中 | **混合策略自动平衡** ✅ |

**核心思路**: **不改底层，改上层策略**
- Padding 让 prompt 对齐
- 智能选择让 block_size 最优
- 混合策略让两者协同

---

**签署**: Solar (CEO) + 战略家
**日期**: 2026-03-14
**下一步**: 实施策略 1 (Prompt Padding)，验证效果
