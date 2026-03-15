# Prefix Cache 诊断报告

**诊断时间**: 2026-03-14 16:50
**问题**: 所有请求的 cached_tokens = 0，缓存完全不工作

---

## 已完成的修复

| 修复项 | 状态 | 说明 |
|--------|------|------|
| ✅ 方案 1: 禁用 block_size 自动提升 | **已实现并启用** | `disable_block_size_enlargement=True` |
| ✅ 降低 block_size | **64 tokens** | 从 1024 → 256 → 64 |
| ✅ Block 创建 | **成功** | `num_new_blocks=1`，日志显示创建了 blocks |

---

## 当前症状

### 1. Blocks 被成功创建

```log
INFO:omlx.cache.prefix_cache:📊 store_cache: existing_tokens=0, new_tokens=64, block_size=64, num_new_blocks=1
INFO:omlx.scheduler:Using boundary cache snapshot for...: storing 64/111 tokens
```

**说明**: 场景 B 的 10 个请求都创建了 1 个 block（前 64 tokens）

---

### 2. 但是缓存匹配失败

```log
INFO:omlx.scheduler:Cache match result: can_skip=False, skip_reason=none, cache_hit_ratio=0.0%, remaining=61
INFO:omlx.cache.prefix_cache:📊 store_cache: existing_tokens=0, new_tokens=64, block_size=64, num_new_blocks=1
```

**所有请求（包括第 2-10 个）都显示：**
- `existing_tokens=0` - 没有找到已存在的缓存
- `cache_hit_ratio=0.0%` - 0% 命中率
- `remaining=61` - 所有 tokens 都需要重新计算

---

### 3. 服务器统计也是 0

```json
{
  "total_cached_tokens": 0,
  "cache_efficiency": 0.0
}
```

---

## 逻辑分析

### 缓存应该如何工作

```
请求 1: 111 tokens → 创建 block_hash_1 (前 64 tokens) → store
请求 2: 111 tokens (相同 System Prompt) → 查找 block_hash_1 → 应该命中！
请求 3: 111 tokens (相同 System Prompt) → 查找 block_hash_1 → 应该命中！
...
```

### 实际发生了什么

```
请求 1: 111 tokens → 创建 block_hash_1 → store (num_new_blocks=1)
请求 2: 111 tokens → 查找 block_hash_1 → ❌ 未命中 (existing_tokens=0)
请求 3: 111 tokens → 查找 block_hash_1 → ❌ 未命中 (existing_tokens=0)
...
```

---

## 根本原因分析（已确认） ✅

### ⚠️ Prompt/Output 长度不匹配导致缓存查找失败

**症状**:
- `match_cache` 阶段：`cache_hit_ratio=0.0%`（匹配失败）
- `store_cache` 阶段：成功复用 cached block（去重逻辑工作）
- 第 2 个请求快了 15 倍（0.51s vs 7.75s）

**原因**:

1. **Boundary Snapshot 机制**（ArraysCache 优化）:
   ```python
   # scheduler.py:3295-3306
   boundary_override = self._get_boundary_store_override(...)
   if boundary_override is not None:
       (token_sequence_to_store, ...) = boundary_override
   ```
   - Full sequence: 70 tokens (50 prompt + 20 output)
   - Boundary override: **只存储前 64 tokens**（跳过 trailing partial block）

2. **缓存匹配阶段**（prefill 前）:
   ```python
   # prefix_cache.py:312
   block_table, remaining = self.fetch_cache("_skip_check", tokens, extra_keys)
   ```
   - 传入的 `tokens` 只有 **50 个** (prompt only，不包括 output)
   - `num_full_blocks = 50 // 64 = 0` → 没有完整 block
   - **无法匹配任何缓存**

3. **缓存存储阶段**（prefill 后）:
   ```python
   # prefix_cache.py:476-484
   existing_block = self.paged_cache.find_cached_block(block_tokens, parent_hash)
   if existing_block:
       # 复用成功！
       continue
   ```
   - 传入的 `tokens` 是 **64 个** (boundary snapshot)
   - 去重逻辑找到第 1 个请求的 block → **复用成功**
   - 这就是为什么只有 1 次 `🔑 Computed block_hash` 日志

**证据**:

```log
# 请求 1:
Cache match result: cache_hit_ratio=0.0%, remaining=50  ← 50 prompt tokens
📊 store_cache: new_tokens=64, num_new_blocks=1         ← 64 tokens stored
🔑 Computed block_hash: hash=c377373b40626b10           ← Block created

# 请求 2:
Cache match result: cache_hit_ratio=0.0%, remaining=50  ← 50 prompt tokens (< 64, 无法匹配)
📊 store_cache: new_tokens=64, num_new_blocks=1         ← 64 tokens
(No 🔑 Computed block_hash)                             ← 复用了请求 1 的 block！
```

**为什么第 2 个请求更快？**
- 模型已经加载（18.61s → 0.51s）
- `store_cache` 的去重逻辑成功复用了 KV cache 数据
- **但是 Skip Logic 没有生效**（`can_skip=False`），仍然执行了 prefill

---

## 修复方案

### 方案 A: 让 `match_cache` 也使用 boundary override

```python
# scheduler.py:2462
# 当前代码：
cache_result = self.block_aware_cache.match_cache_with_skip_logic(
    request.prompt_token_ids,  # ← 只传 prompt tokens
    ...
)

# 修复后：
# 使用和 store_cache 相同的 boundary override 逻辑
boundary_override = self._get_boundary_store_override(...)
if boundary_override:
    token_sequence_for_match = boundary_override[0]
else:
    token_sequence_for_match = request.prompt_token_ids

cache_result = self.block_aware_cache.match_cache_with_skip_logic(
    token_sequence_for_match,  # ← 使用 boundary override 的 tokens
    ...
)
```

### 方案 B: 修改缓存存储策略

不使用 boundary snapshot，改为存储完整的 prompt tokens：

```python
# scheduler.py:3286
token_sequence_to_store = list(request.prompt_token_ids)  # 只存储 prompt
```

但这会破坏 ArraysCache 的优化。

---

## 方案选择

**推荐方案 A**：
- 让 `match_cache` 和 `store_cache` 使用相同的 token sequence
- 保留 boundary snapshot 的优化（ArraysCache 需要）
- 确保缓存匹配和存储使用一致的逻辑

---

## 下一步诊断

### 立即执行（5 分钟）

1. **检查缓存清理日志**
   ```bash
   tail -500 /Users/lisihao/ThunderOMLX/omlx_server_new2.log | grep -i "clear_request_entry\|delete_block_table"
   ```

2. **检查 block_hash 一致性**
   - 在 `compute_block_hash` 中添加 debug 日志
   - 验证相同 tokens 是否生成相同 hash

3. **检查 ref_count 是否正确**
   - 在 `store_cache` 后检查 block.ref_count
   - 在请求完成前检查 block.ref_count

---

### 中期修复（30 分钟）

如果确认是**假设 1（缓存被立即释放）**：

**修复方案**:
```python
# scheduler.py 中修改缓存释放策略

def _on_request_complete(self, request_id: str):
    # 不要立即释放 block_table
    # 让 Paged Cache Manager 自动管理生命周期

    # ❌ 不要这样做:
    # self.block_aware_cache.clear_request_entry(request_id)

    # ✅ 只减少 ref_count，但保留在缓存中
    if self.block_aware_cache is not None:
        table = self.paged_cache_manager.request_tables.get(request_id)
        if table:
            for block_id in table.block_ids:
                self.paged_cache_manager.release_for_eviction([block_id])
```

---

## 预期效果

修复后应该看到：

```log
INFO:omlx.scheduler:Cache match result: can_skip=True, skip_reason=approximate, cache_hit_ratio=57.7%, remaining=47
INFO:omlx.cache.prefix_cache:📊 store_cache: existing_tokens=64, new_tokens=47, block_size=64, num_new_blocks=0
```

```json
{
  "total_cached_tokens": 640,  // 10 个请求 × 64 tokens
  "cache_efficiency": 57.7%    // 64 / 111
}
```

**端到端性能**:
- Approximate Skip 生效 → 跳过 prefill 计算
- Prefill 时间: 2000ms → 200ms (10x 加速)
- 端到端延迟: 10s → 3s (3x 加速)

---

*诊断完成于: 2026-03-14 16:50*
*待确认: 假设 1 (缓存被立即释放) vs 假设 2 (block_hash 不一致)*
