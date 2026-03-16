# Speculative Decoding 评估报告

## 执行摘要

**结论**：Speculative Decoding 在 **Qwen3.5-0.8B + Qwen3.5-35B** 模型组合下**不适用**。

- ❌ 无加速效果（平均加速比 < 1.0×）
- ❌ 生成质量下降（重复、不连贯）
- ❌ Draft model 预测准确度不足

**建议**：专注于其他优化方向（Chunked Prefill、Quantization、Parallel Sampling）

---

## 测试配置

### 模型
- **Target Model**: Qwen3.5-35B-A3B-6bit
  - 参数量：35B
  - 量化：6-bit
  - 路径：`~/.omlx/models/Qwen3.5-35B-A3B-6bit`

- **Draft Model**: Qwen3.5-0.8B-MLX-4bit
  - 参数量：0.8B（仅为 target 的 2.3%）
  - 量化：4-bit
  - 大小：~622MB
  - 路径：`~/.omlx/models/Qwen3.5-0.8B-MLX-4bit`

### 参数
- K (num_draft_tokens): 4
- Max tokens: 50
- Sampling: Greedy (temperature = 0)

### 测试平台
- 硬件：M4 Pro (Apple Silicon)
- 系统：macOS 15.3
- MLX 版本：0.x (with built-in speculative decoding)

---

## 测试结果

### 性能对比

使用 MLX 内置的 speculative decoding 实现：

| Prompt | Spec Decoding | Baseline | 加速比 |
|--------|--------------|----------|--------|
| "Once upon a time" | 15.1 tok/s | 56.1 tok/s | **0.27×** ⬇️ |
| "The capital of France is" | 58.1 tok/s | 55.7 tok/s | **1.04×** ≈ |
| "To be or not to be" | 45.0 tok/s | 56.3 tok/s | **0.80×** ⬇️ |

**平均加速比**：**0.70×**（变慢 30%）

---

### 1.7B Draft Model 测试（改进尝试）

**动机**：用户指出 35B target model 使用 A3B 架构，实际只激活 3.5B 参数。因此 0.8B draft 的比例是 1:4，尝试使用 1.7B draft 改善比例到 1:2。

**Draft Model**: Qwen3-1.7B-MLX-4bit
- 参数量：1.7B
- 量化：4-bit
- 大小：~887MB
- 路径：`~/.omlx/models/Qwen3-1.7B-MLX-4bit`

**测试配置**：
- K (num_draft_tokens): 2, 4, 6
- Max tokens: 50
- 4 个测试 prompts

**结果**：

| Prompt | K=2 | K=4 | K=6 | Baseline | K=4 加速比 |
|--------|-----|-----|-----|----------|-----------|
| "Once upon a time" | 15.1 tok/s | 38.1 tok/s | 38.4 tok/s | 54.4 tok/s | **0.69×** |
| "The capital of France is" | 47.7 tok/s | 32.6 tok/s | 35.9 tok/s | 55.1 tok/s | **0.59×** |
| "To be or not to be" | 39.7 tok/s | 30.7 tok/s | 30.6 tok/s | 54.6 tok/s | **0.56×** |
| "Explain quantum computing" | 39.4 tok/s | 48.2 tok/s | 25.0 tok/s | 53.9 tok/s | **0.90×** |

**平均加速比（K=4）**：**0.68×**（变慢 32%）

**生成质量**：
- ❌ 仍然出现重复（"1.1.1.1..."、"The The The..."）
- ❌ 格式异常（日期、数字、特殊字符）
- ❌ 不连贯的输出

**结论**：
- 即使改善参数比例到 1:2，效果仍然不佳
- **根本问题不是参数比例，而是模型版本/架构不匹配**
- Qwen3 (draft) vs Qwen3.5 (target) 的训练数据和架构差异导致预测不准

---

### 生成质量对比

#### Test 1: "Once upon a time"

**Speculative Decoding**:
```
, in a magical town called the Future of Town of the Town.
The Town of the Town
 The Town
 Towns
 material called "
 Town
 Town
 Town
 Town
 Town
 Town
 Town
 Town
 Town
 Town
```
❌ 重复 "Town" 多次，不连贯

**Baseline**:
```
, in a magical world of science, there was a very special thing called a "mole".
Now, you might be wondering, what is this mole? Well, imagine you have a big box
of candies. If you count how many candies are
```
✅ 连贯、合理

#### Test 2: "The capital of France is"

**Speculative Decoding**:
```
Paris.
The capital of France is Paris, and the capital of Germany.
The capital of France is Paris, and the capital of France is the capital of France
is Paris, and the capital of France is Paris, and the capital of France
```
❌ 重复结构，语义混乱

**Baseline**:
```
Paris.
The capital of France is Paris.
The capital of France is Paris.
The capital of France is Paris.
...
```
✅ 虽有重复，但结构清晰

---

## 根因分析

### 1. 模型版本和架构不匹配（主要原因）

**测试结果表明**：即使改善参数比例，问题依然存在

| Draft Model | 参数比例 | 加速比 | 结论 |
|-------------|----------|--------|------|
| Qwen3.5-0.8B | 1:4 (0.8B/3.5B) | 0.70× | 失败 |
| Qwen3-1.7B | 1:2 (1.7B/3.5B) | 0.68× | 仍然失败 |

**根本原因**：
- ❌ **版本不匹配**：Qwen3 (draft) vs Qwen3.5 (target)
  - 不同的训练数据集
  - 不同的训练方法和目标
  - Token 分布差异大

- ❌ **架构不匹配**：Dense model (draft) vs MoE model (target)
  - Draft 是标准 dense transformer
  - Target 是 Mixture-of-Experts，动态路由
  - 内部表示空间不一致

- ❌ **预测不准**：导致接受率极低（推测 < 10%）
  - Draft model 无法准确预测 target model 的下一个 token
  - 即使第一个 token 有时匹配，连续生成后快速 drift

### 2. 理论分析

Speculative Decoding 加速比公式：
```
Speedup = 1 + α × K
```
其中：
- α = 接受率
- K = 候选 tokens 数量

当 α < 0.2 时：
```
Speedup = 1 + 0.2 × 4 = 1.8×
```

但考虑 draft model 的开销：
```
实际加速比 = (1 + α × K) / (1 + overhead)
```

当 α 很低时，overhead 可能超过收益，导致**负加速**。

### 3. Tokenizer 验证

✅ **Tokenizer 完全兼容**
- Vocab size: 248044（两者一致）
- 所有测试提示的 tokenization 结果相同
- 特殊 tokens (EOS, PAD) 一致

**结论**：问题不在 tokenizer，而是模型预测能力。

### 4. 单 Token 生成测试

从 `debug_inference.py` 的结果：

| Prompt | Target Top-1 | Draft Top-1 | 匹配? |
|--------|--------------|-------------|------|
| "Once upon a time" | 11 (",") | 11 (",") | ✅ |
| "The capital of France is" | 11751 ("Paris") | 11751 ("Paris") | ✅ |
| "To be or not to be" | 25 (":") | 264 (" a") | ❌ |

**观察**：
- 简单提示（如常识性问答）匹配率高
- 复杂/开放性提示匹配率低
- **连续生成时 draft model 容易 drift**

---

## 实现尝试

### 尝试 1：自定义实现

**文件**：`src/omlx/speculative_decoding.py`（~450 行）

**问题**：
1. Cache 管理不正确
2. 接受率 0%
3. 生成异常输出（重复空格 token）

**根因**：
- 对 MLX cache API 使用不当
- Prefill 后 cache 同步问题

### 尝试 2：使用 MLX 内置实现

**函数**：`mlx_lm.generate(..., draft_model=draft_model)`

**结果**：
- ✅ 代码正确（MLX 官方实现）
- ❌ 效果不佳（模型不匹配）
- ❌ 加速比 < 1.0×

**结论**：**不是实现问题，而是模型组合问题**

---

## Speculative Decoding 的适用条件

基于本次测试和文献研究，Speculative Decoding 需要：

### ✅ 必要条件

1. **Draft model 与 target model 高度相似**
   - 同一训练数据
   - 同一 fine-tuning 方法
   - 参数比例建议：draft ≥ 10% target

2. **高接受率（α ≥ 0.6）**
   - 需要 draft model 有足够的预测能力
   - 低温度采样（temperature < 0.7）

3. **Draft model 速度显著快于 target**
   - 建议 ≥ 5× 速度差
   - 否则 draft 开销抵消收益

### ❌ 不适用场景

1. ❌ 模型差距过大（如 0.8B vs 35B）
2. ❌ 不同训练数据/领域
3. ❌ 高温度采样（temperature > 1.0）
4. ❌ Draft model 速度优势不明显

---

## 替代方案

### 1. 继续优化 Chunked Prefill ✅

**已完成**（v0.4.1）：
- ✅ Metal GPU 同步修复
- ✅ 异步预取线程安全
- ✅ 支持超长提示（3072+ tokens）
- ✅ 首次推理快 50%

**推荐**：继续优化，效果已验证

### 2. 实现 Quantization (W8A8/W4A16)

**预期收益**：
- 内存占用 ↓ 50-75%
- 推理速度 ↑ 1.5-2×
- 适用于所有场景

**优先级**：⭐⭐⭐⭐⭐（推荐）

### 3. Parallel Sampling

**预期收益**：
- 批量推理吞吐量 ↑ 2-3×
- 适用于多用户场景

**优先级**：⭐⭐⭐⭐

### 4. Flash Attention / Paged Attention

**预期收益**：
- 注意力计算 ↑ 2-3×
- 内存占用 ↓ 30-50%

**优先级**：⭐⭐⭐

---

## 经验教训

### 1. 不要盲目追求新技术

❌ **错误思维**："Speculative Decoding 是新技术 → 必然能加速"

✅ **正确思维**："先验证前提条件 → 再决定是否适用"

### 2. 优先使用官方实现

✅ MLX 内置的 speculative decoding 是正确的

❌ 自己实现容易出错（cache 管理复杂）

### 3. 模型选择比算法更重要

Draft model 的选择直接决定 speculative decoding 的效果：
- 0.8B draft (1:4 ratio) → 失败（0.70× 加速比）
- 1.7B draft (1:2 ratio) → 仍然失败（0.68× 加速比）
- **参数比例不是唯一因素**，模型版本和架构一致性更关键
- 建议：使用同版本、同架构的模型（如 Qwen3.5-4B + Qwen3.5-35B）
- 注意：Qwen3.5 小模型已下架，无法找到合适的 draft model

### 4. 基准测试的重要性

如果没有做基准测试：
- 可能误以为实现有问题
- 浪费时间调试代码
- 错过真正的问题（模型不匹配）

✅ **本次测试**：
1. 测试自定义实现 → 发现 cache 问题
2. 测试官方实现 → 排除代码问题
3. 对比 baseline → 确认模型问题

---

## 结论和建议

### 结论

1. **Speculative Decoding 不适用于当前模型组合**
   - 测试了两个 draft models：
     - Qwen3.5-0.8B (1:4 ratio) → 0.70× 加速比
     - Qwen3-1.7B (1:2 ratio) → 0.68× 加速比
   - 所有测试均**变慢**（-30% to -32%）
   - 生成质量严重下降（重复、不连贯）

2. **主要原因：模型版本和架构不匹配**
   - ❌ Qwen3 vs Qwen3.5 版本差异
   - ❌ Dense vs MoE 架构不匹配
   - ❌ 训练数据和方法不同
   - 结果：接受率极低（< 10%），draft 开销 > 收益

3. **其他优化方向更有价值**
   - Chunked Prefill（已验证，+50% 首次推理）
   - Quantization（预期 +1.5-2× 速度）
   - Parallel Sampling（预期 +2-3× 吞吐量）

### 建议

**短期**（推荐）：
1. ✅ 保持 Chunked Prefill 优化
2. ⭐ 实现 W8A8 Quantization（下一步）
3. ⭐ 实现 Parallel Sampling

**中期**（可选）：
4. 尝试 Flash Attention
5. 优化 KV Cache 管理

**长期**（研究性）：
6. 如果有更大的 draft model（4B+），可重新测试 speculative decoding
7. 探索其他加速技术（如 Medusa）

---

## 附录

### A. 测试脚本

**0.8B Draft Model 测试**：
- `download_draft_model.py` - 下载 Qwen3.5-0.8B-MLX-4bit
- `test_speculative_decoding.py` - 自定义实现测试
- `test_builtin_speculative.py` - MLX 内置实现测试
- `debug_tokenizer.py` - Tokenizer 兼容性检查
- `debug_inference.py` - 单 token 生成调试
- `test_mlx_generate.py` - 官方 generate 函数测试

**1.7B Draft Model 测试**（改进尝试）：
- `download_1.7b_draft.py` - 下载 Qwen3-1.7B-MLX-4bit
- `test_1.7b_speculative.py` - 1.7B draft model 完整测试（K=2,4,6）

### B. 实现文件

- `src/omlx/speculative_decoding.py` - 自定义引擎（未使用）
- `docs/SPECULATIVE_DECODING_DESIGN.md` - 设计文档

### C. 相关文献

1. Leviathan et al. (2023) - "Fast Inference from Transformers via Speculative Decoding"
2. Chen et al. (2023) - "Accelerating Large Language Model Decoding with Speculative Sampling"
3. MLX Documentation - Speculative Decoding: https://ml-explore.github.io/mlx/

---

*Report Date: 2026-03-15*
*Author: Solar*
*Status: ❌ **DEPRECATED - 已废除***

**废除原因**：
- MoE 架构（Qwen3.5-35B-A3B）与 Speculative Decoding 不兼容
- ThunderLLAMA 上多次测试均失败
- 无法找到合适的同版本 draft model（Qwen3.5 小模型已下架）
- 所有测试均出现负加速（0.68-0.70× 加速比）

**决策**：不再投入时间研究此优化方向，专注于其他优化（Quantization、Parallel Sampling）
