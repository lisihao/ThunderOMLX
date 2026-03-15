# ThunderOMLX 缓存优化整合方案

> **目标**: 整合 ThunderLLAMA 的优化策略 + 已有的 padding/智能策略，提升缓存命中率到 90%+

---

## 📊 当前状态分析

### ✅ 已实现（但未完全生效）

| 优化 | 状态 | 效果 |
|------|------|------|
| 方案 1: 禁用 block_size 自动提升 | ✅ 启用 | `disable_block_size_enlargement=True` |
| 方案 2: 动态 block_size 选择 | ✅ 实现 | `block_size=64` |
| 策略 1: Prompt Padding 到 block 边界 | ✅ 实现 | **未启用** |
| 策略 2: 智能 block_size 选择 | ✅ 实现 | **未启用** |
| 策略 4: 混合策略 | ✅ 实现 | **未启用** |

### ⚠️ 当前测试结果

```
请求 1: 13.45s
请求 2: 0.61s (快了 22 倍)

缓存匹配:
- ✅ Cache HIT at block 0: cached_tokens=64
- cache_hit_ratio=55.2% (64/116 tokens)
- can_skip=False (需要 90%+ 才能触发 Skip Logic)
```

**问题**:
- Prompt 有 116 tokens，只匹配了 64 tokens（1 个 block）
- 剩余 52 tokens 无法匹配（不足 1 个完整 block）
- Skip Logic 未生效

---

## 🎯 ThunderLLAMA 的优化策略

### 1. **Prefix Matching（前缀匹配）**

**实现**（`thunder-lmcache-storage.cpp:1460-1640`）:

```cpp
thunder_prefix_match find_prefix_match(
    const llama_token *tokens,
    size_t n_tokens,
    ...
) {
    // 1. ContextPilot chunk hash matching (fast path)
    for (size_t chunk_idx = 0; chunk_idx < contextpilot_chunk_hashes->size(); chunk_idx++) {
        // Check if this chunk exists for ALL layers
        if (chunk_found) {
            matched_chunks++;
        } else {
            break;  // Stop at first missing chunk
        }
    }

    result.matched_tokens = matched_chunks * THUNDER_CHUNK_SIZE;

    // 2. Token-based matching (fallback)
    // Try longest prefix first, then shorter prefixes
    for (size_t len = n_tokens; len >= THUNDER_CHUNK_SIZE; len -= THUNDER_CHUNK_SIZE) {
        if (all_chunks_match_for_all_layers) {
            result.matched_tokens = len;
            break;
        }
    }
}
```

**关键特性**:
- ✅ 逐个 chunk 匹配，直到遇到第一个不匹配的 chunk
- ✅ 返回匹配的 token 数量（即使不是完整的请求）
- ✅ **支持部分匹配**（这是 ThunderOMLX 缺失的）

### 2. **Approximate Matching（近似匹配）**

**环境变量** (`thunderllama.conf`):
```bash
THUNDER_PREFIX_MATCHING=1  # 启用 prefix 匹配
```

**效果**:
- 如果前 90% 的 tokens 匹配，可以复用缓存
- 剩余 10% 的 tokens 重新计算

---

## 🔧 ThunderOMLX 已有策略（未启用）

### 策略 1: Prompt Padding 到 block 边界

**位置**: `src/omlx/cache/prefix_cache.py:180-204`

```python
def _pad_to_block_boundary(
    self,
    tokens: List[int],
    block_size: int
) -> Tuple[List[int], int]:
    """
    Pad tokens to next block boundary.

    Example:
        - Input: 116 tokens, block_size=64
        - Output: 128 tokens (116 + 12 padding)
        - Result: 2 complete blocks (vs 1.8 blocks)
    """
    remainder = len(tokens) % block_size
    if remainder == 0:
        return tokens, 0

    padding_needed = block_size - remainder
    padded_tokens = tokens + [self.pad_token_id] * padding_needed

    return padded_tokens, padding_needed
```

**效果**:
- 116 tokens → 128 tokens (padding 12)
- 缓存匹配: 64 tokens → 128 tokens（2 个完整 blocks）
- 命中率: 55.2% → **100%**（如果 padding tokens 被忽略）

### 策略 2: 智能 block_size 选择

**位置**: `src/omlx/cache/prefix_cache.py:206-240`

```python
def _choose_optimal_block_size(
    self,
    prompt_length: int,
    available_sizes: List[int] = [32, 64, 128, 256]
) -> int:
    """
    Choose block_size that minimizes waste.

    Example:
        - Input: 116 tokens
        - Options: [32, 64, 128, 256]
        - Best: 64 (waste=52) vs 32 (waste=20) vs 128 (waste=12)
        - Choose: 128 (waste 率 10.3%)
    """
    min_waste = float('inf')
    best_size = available_sizes[0]

    for size in available_sizes:
        num_blocks = (prompt_length + size - 1) // size
        total_capacity = num_blocks * size
        waste = total_capacity - prompt_length
        waste_ratio = waste / prompt_length

        if waste_ratio < min_waste and waste_ratio < 0.2:  # 最多浪费 20%
            min_waste = waste_ratio
            best_size = size

    return best_size
```

**效果**:
- 116 tokens → block_size=128 → waste=12 (10.3%)
- 缓存匹配: 1 个 block (128 tokens)
- 命中率: 55.2% → **100%**

---

## 🚀 整合方案

### 阶段 1: 启用已有策略（10 分钟）

**修改**: `src/omlx/scheduler.py`

```python
# Line 2450-2470
if self.block_aware_cache is not None:
    # 1. 启用 Prompt Padding（策略 1）
    padded_tokens, padding_count = self.block_aware_cache._pad_to_block_boundary(
        request.prompt_token_ids,
        self.config.paged_cache_block_size
    )

    # 或者 2. 启用智能 block_size 选择（策略 2）
    optimal_block_size = self.block_aware_cache._choose_optimal_block_size(
        len(request.prompt_token_ids)
    )
    # 动态调整 block_size
    self.block_aware_cache.paged_cache.block_size = optimal_block_size

    # 使用 padded tokens 进行缓存匹配
    cache_result = self.block_aware_cache.match_cache_with_skip_logic(
        padded_tokens,  # ← 使用 padded tokens
        extra_keys=extra_keys,
        approx_threshold=0.90,
    )
```

**预期效果**:
- 命中率: 55.2% → **100%**
- Skip Logic: 生效 (Full Skip)
- 端到端延迟: 0.61s → **0.2s**（3x 加速）

---

### 阶段 2: 实现部分 block 匹配（30 分钟）

**问题**: 当前逻辑只匹配完整 blocks

```python
# paged_cache.py:1003
num_full_blocks = len(token_ids) // self.block_size  # ← 116 // 64 = 1
```

**解决方案**: 实现类似 ThunderLLAMA 的 prefix matching

```python
# paged_cache.py:1003-1047 (修改)
def get_computed_blocks(self, token_ids, extra_keys=None):
    """
    Find cached prefix blocks, supporting partial block matching.
    """
    cached_blocks = []
    parent_hash = None
    num_cached_tokens = 0

    num_full_blocks = len(token_ids) // self.block_size

    for i in range(num_full_blocks):
        # ... 现有逻辑 ...
        if cached_block is None:
            break  # Cache miss, stop here

        cached_blocks.append(cached_block)
        parent_hash = block_hash
        num_cached_tokens += self.block_size

    # ⭐ 新增: 尝试匹配部分 block（最后不足 block_size 的 tokens）
    remaining_tokens = len(token_ids) - num_cached_tokens
    if remaining_tokens > 0 and remaining_tokens < self.block_size:
        partial_tokens = token_ids[num_cached_tokens:]

        # 计算部分 block 的 hash
        partial_hash = compute_block_hash(
            parent_hash, partial_tokens,
            extra_keys=extra_keys, model_name=self.model_name,
        )

        # 查找部分 block
        partial_block = self.cached_block_hash_to_block.get_block(partial_hash)

        if partial_block:
            cached_blocks.append(partial_block)
            num_cached_tokens += remaining_tokens
            logger.info(f"✅ Partial block HIT: {remaining_tokens} tokens")

    return cached_blocks, num_cached_tokens
```

**预期效果**:
- 116 tokens → 匹配 64 (block 0) + 52 (partial block 1)
- 命中率: 55.2% → **100%**

---

### 阶段 3: 实现 Approximate Skip（已有，需调优）

**当前逻辑** (`prefix_cache.py:322-345`):

```python
# Full Skip: 100% cache hit
can_full_skip = (cache_hit_ratio == 1.0) and (len(remaining) == 0)

# Approximate Skip: 90%+ cache hit
can_approx_skip = (cache_hit_ratio >= approx_threshold) and not can_full_skip
```

**问题**:
- 55.2% < 90% → 不触发 Approximate Skip

**解决**:
- 启用阶段 1 或阶段 2 → 命中率提升到 90%+ → 自动触发

---

## 📈 预期效果对比

| 场景 | 当前 | 阶段 1 | 阶段 2 | 阶段 1+2 |
|------|------|--------|--------|----------|
| 匹配 tokens | 64/116 | 128/128 | 116/116 | 128/128 |
| 命中率 | 55.2% | 100% | 100% | 100% |
| Skip Logic | ❌ | ✅ Full | ✅ Full | ✅ Full |
| 请求 2 延迟 | 0.61s | 0.2s | 0.2s | 0.2s |
| 加速比 | 22x | **67x** | **67x** | **67x** |

---

## 🎯 实施顺序

1. **立即**: 启用策略 1（Prompt Padding）→ 10 分钟
2. **短期**: 实现阶段 2（部分 block 匹配）→ 30 分钟
3. **中期**: 整合 ThunderLLAMA 的 ContextPilot chunk hashing → 2 小时

---

## ✅ 验证方法

```bash
# 1. 运行测试
python3 scripts/test_cache_matching_debug.py

# 2. 检查日志
grep "Cache match result" omlx_server_debug.log
# 应该看到: cache_hit_ratio=100.0%, can_skip=True

# 3. 验证性能
# 请求 1: ~13s
# 请求 2: ~0.2s (65x 加速)
```

---

*Created: 2026-03-14*
*Priority: P0 - 立即执行*
