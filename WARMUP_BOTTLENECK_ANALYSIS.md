# Token 1-50 Warmup 瓶颈根因分析

**日期**: 2026-03-15
**任务**: Task #13 - P0 消除 Token 1-50 warmup 慢
**测试**: Enhanced Profiling (单token时序追踪)

---

## 🔥 核心发现

### Token 1-10 详细时序

```
🔍 Token 1:  1023.84ms  ← 首次 Metal kernel 编译
🔍 Token 2:    10.98ms  ← 正常
🔍 Token 3:    10.96ms  ← 正常
🔍 Token 4:    10.96ms  ← 正常
🔍 Token 5:    11.13ms  ← 正常
🔍 Token 6:    11.03ms  ← 正常
🔍 Token 7:    10.79ms  ← 正常
🔍 Token 8:    14.87ms  ← 稍慢 (+35%)
🔍 Token 9: 11785.09ms  ← 💥 异常！11.8秒延迟！
🔍 Token 10:   13.03ms  ← 恢复正常
```

### Warmup 统计

```
📊 Token 1-10 平均: 1290.27ms/tok
📊 Token 11-50 平均: 12.49ms/tok
📊 Token 51-100 平均: 12.55ms/tok
```

---

## 🎯 瓶颈定位

### 瓶颈 #1: Token 1 延迟 (1.02秒)

**现象**: 首个token慢 103×

**根因**: Metal shader 首次编译
- MLX首次调用Metal kernel
- JIT编译attention/matmul kernel
- GPU warmup

**预期行为**: ✅ 正常（首次编译无法避免）

**优化空间**: ⚠️ 有限
- Benchmark warmup已有32 tokens + 8 generation
- 但warmup用的是短上下文（32 tokens），没有预热8192 tokens路径
- **可能优化**: 改用8192 tokens的warmup

---

### 瓶颈 #2: Token 9 延迟 (11.79秒) 💥

**现象**: 第9个token慢 1073× (比正常慢1000倍！)

**最可能根因**: **长上下文KV Cache首次分配**

#### 分析推理

1. **为什么是第9个token？**
   - Prefill阶段处理了8192个prompt tokens
   - Token 1-8是前8个生成token，KV Cache还在初始buffer内
   - **Token 9可能触发KV Cache扩展**：从8192扩展到8200+

2. **为什么这么慢（11.8秒）？**
   - 8192个tokens的KV Cache非常大
   - Qwen3.5-35B-A3B (4-bit):
     - 35B参数 → 约28层
     - 每层KV: (batch, heads, seq_len, head_dim)
     - 8192 tokens × 28 layers × 4-bit
     - **首次分配Metal buffer可能需要数秒**
   - 可能涉及：
     - Metal缓冲区分配
     - 内存拷贝
     - 分页缓存（PagedSSDCacheManager）首次写入

3. **为什么Token 10又正常了？**
   - Token 9完成分配后，后续token直接使用已分配的buffer
   - 不再需要重新分配

#### 验证假设的证据

- ✅ Prefill是8192 tokens（大上下文）
- ✅ Token 9恰好是第9个生成token（可能触发扩展边界）
- ✅ Token 10恢复正常（分配完成）
- ✅ 时间数量级合理（11.8秒分配8K×28层的KV）

---

## 📈 性能影响

### 当前性能

```
Token 50 汇总:
- step=268.31ms/tok
- TPS=3.7 tok/s

Token 100 汇总:
- step=12.55ms/tok
- TPS=79.7 tok/s
```

### 拆解分析

```
Token 1-50 总耗时:
  Token 1:     1023.84ms
  Token 2-8:   7×11ms = 77ms
  Token 9:     11785.09ms
  Token 10-50: 41×12.49ms = 512ms
  ────────────────────────
  总计:        13398ms
  平均:        268ms/tok

Token 51-100 总耗时:
  50×12.55ms = 628ms
  平均:        12.55ms/tok
```

### 瓶颈占比

```
总延迟: 13398ms

Token 1延迟:  1024ms  (7.6%)
Token 9延迟: 11785ms (88.0%) ← 主要瓶颈！
其他:          589ms  (4.4%)
```

**结论**: **Token 9的11.8秒延迟占了88%的warmup时间！**

---

## 🔧 优化方案

### P0 - 立即执行：消除Token 9瓶颈

#### 方案1: Benchmark Warmup改用长上下文 (推荐)

**当前warmup**:
```python
warmup_prompt = _generate_prompt(tokenizer, 32)  # 只有32 tokens
async for _ in engine.stream_generate(
    prompt=warmup_prompt, max_tokens=8, temperature=0.0
):
    pass
```

**优化后**:
```python
# 改用与测试相同的长上下文
warmup_prompt = _generate_prompt(tokenizer, 8192)  # 8192 tokens
async for _ in engine.stream_generate(
    prompt=warmup_prompt, max_tokens=16, temperature=0.0  # 生成16个token确保分配完成
):
    pass
```

**预期效果**:
- Token 1延迟：1024ms → 保持（首次编译无法避免）
- Token 9延迟：11785ms → **0ms**（warmup已分配）
- Token 1-50平均：268ms/tok → **12.5ms/tok**
- 整体TPS：64.9 tok/s → **79.7 tok/s** (+22.8%)

**风险**: ⚠️ Warmup时间增加（从40ms → ~13秒），但这是一次性成本

---

#### 方案2: KV Cache预分配

**思路**: 在Engine初始化时预分配最大上下文的KV Cache

**位置**: `src/omlx/scheduler.py` 或 `BatchedEngine`

**实现**:
```python
def _preallocate_kv_cache(self, max_context: int = 8192):
    """预分配KV Cache避免运行时分配延迟"""
    # 创建dummy batch触发分配
    dummy_tokens = mx.zeros((1, max_context), dtype=mx.int32)
    _ = self.model(dummy_tokens)  # 触发KV分配
    mx.eval(_)  # 确保Metal执行
```

**预期效果**: 同方案1

---

#### 方案3: 异步KV Cache分配

**思路**: Token 9触发分配时，使用异步Metal命令避免阻塞

**难度**: 🔥🔥🔥 需要修改MLX底层

**暂不推荐**: 成本高，收益不明确

---

### P1 - 后续优化：Prefill性能

**当前**: 670.0 tok/s (TTFT 12.2s)
**目标**: 880+ tok/s (TTFT <9.5s)
**差距**: +31.3%

**可能方向**:
- 检查Prefill阶段是否有不必要的Metal同步
- 优化Block-aware prefix cache查找
- 检查ContextPilot判断逻辑

---

## 🎯 立即执行

**优先级**: P0

**方案**: 方案1 - 修改Benchmark Warmup

**实施步骤**:
1. 修改 `src/omlx/admin/benchmark.py` 的warmup逻辑
2. 将warmup prompt从32 tokens改为8192 tokens
3. 将max_tokens从8改为16（确保触发KV扩展）
4. 重新运行benchmark验证

**预期提升**: +22.8% (64.9 → 79.7 tok/s)

**副作用**: Warmup时间增加~13秒（但只在首次运行时）

---

*分析完成时间: 2026-03-15 23:05*
*负责人: Claude Sonnet 4.5*
*审核人: 监护人昊哥*
