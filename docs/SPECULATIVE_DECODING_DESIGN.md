# Speculative Decoding 设计文档

## 概述

Speculative Decoding 是一种推理加速技术，通过小模型（draft model）快速生成候选 tokens，然后由大模型（target model）并行验证，从而减少推理延迟。

## 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                  Speculative Decoding Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Draft Phase (小模型快速生成 K 个 tokens)                │
│     ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│     │ token 1 │ -> │ token 2 │ -> │ token K │              │
│     └─────────┘    └─────────┘    └─────────┘              │
│          ▼              ▼              ▼                      │
│                                                              │
│  2. Verification Phase (大模型并行验证)                      │
│     ┌───────────────────────────────┐                       │
│     │  Target Model Forward Pass    │                       │
│     │  (parallel verification)      │                       │
│     └───────────────────────────────┘                       │
│          │                                                   │
│          ▼                                                   │
│  3. Accept/Reject Decision                                  │
│     ✅ ✅ ✅ ❌ (accept 3, reject from 4th)                  │
│                                                              │
│  4. Continue from last accepted token                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 理论加速比

- K = 候选 tokens 数量
- 平均接受率 α ≈ 0.6-0.8（draft 和 target 模型相似度越高，α 越大）
- 理论加速比 ≈ 1 + α × K

示例：
- K=4, α=0.7 → 1 + 0.7 × 4 = 3.8× 加速
- K=6, α=0.6 → 1 + 0.6 × 6 = 4.6× 加速

## 模型选择

### Draft Model
- **Qwen3.5-0.8B-MLX-4bit**
- 参数量：0.8B（仅为 35B 的 2.3%）
- 量化：4-bit（内存占用 ~500MB）
- 推理速度：~100-150 tok/s（M4 Pro）
- 同属 Qwen 系列，架构相同，兼容性好

### Target Model
- **Qwen3.5-35B-A3B-6bit**
- 参数量：35B
- 量化：6-bit
- 推理速度：~13-23 tok/s（baseline）

## 实现架构

### 1. SpeculativeDecodingEngine

```python
class SpeculativeDecodingEngine:
    """
    Speculative Decoding 引擎

    管理 draft model 和 target model，协调生成流程。
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_tokenizer: PreTrainedTokenizer,
        draft_model: nn.Module,
        draft_tokenizer: PreTrainedTokenizer,
        num_speculative_tokens: int = 4,  # K
        config: Optional[SpeculativeConfig] = None
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.K = num_speculative_tokens

        # KV Cache for both models
        self.target_cache = None
        self.draft_cache = None

    def generate_speculative(
        self,
        prompt_tokens: mx.array,
        max_tokens: int,
        sampling_params: SamplingParams
    ) -> Iterator[int]:
        """
        使用 Speculative Decoding 生成 tokens

        Yields:
            Accepted tokens
        """
        # 1. Prefill phase (both models)
        self._prefill(prompt_tokens)

        # 2. Decode loop
        while num_generated < max_tokens:
            # 2.1 Draft phase: generate K candidates
            draft_tokens = self._draft_generate(K=self.K)

            # 2.2 Verify phase: parallel verification
            accepted_count = self._verify_and_accept(draft_tokens)

            # 2.3 Yield accepted tokens
            for i in range(accepted_count):
                yield draft_tokens[i]
                num_generated += 1

            # 2.4 If not all accepted, generate 1 token from target
            if accepted_count < self.K:
                bonus_token = self._target_generate_one()
                yield bonus_token
                num_generated += 1
```

### 2. 关键方法

#### Prefill Phase
```python
def _prefill(self, prompt_tokens: mx.array):
    """
    两个模型同时 prefill
    """
    # Target model prefill (可能使用 Chunked Prefill)
    if self.chunked_prefill_engine:
        logits, cache = self.chunked_prefill_engine.prefill(
            self.target_model, prompt_tokens
        )
    else:
        logits, cache = self.target_model(prompt_tokens)

    self.target_cache = cache

    # Draft model prefill (直接 forward)
    draft_logits, draft_cache = self.draft_model(prompt_tokens)
    self.draft_cache = draft_cache
```

#### Draft Generate
```python
def _draft_generate(self, K: int) -> List[int]:
    """
    Draft model 快速生成 K 个候选 tokens

    使用贪婪采样（temperature=0）
    """
    draft_tokens = []

    for _ in range(K):
        # Forward pass (single token)
        logits, self.draft_cache = self.draft_model(
            mx.array([current_token]), cache=self.draft_cache
        )

        # Greedy sampling
        next_token = mx.argmax(logits[0, -1, :]).item()
        draft_tokens.append(next_token)
        current_token = next_token

    return draft_tokens
```

#### Verify and Accept
```python
def _verify_and_accept(self, draft_tokens: List[int]) -> int:
    """
    Target model 并行验证候选 tokens

    Returns:
        Number of accepted tokens
    """
    # Parallel forward pass (K tokens at once)
    verify_input = mx.array([draft_tokens])
    logits, next_cache = self.target_model(
        verify_input, cache=self.target_cache
    )

    # Check acceptance
    accepted_count = 0
    for i, draft_token in enumerate(draft_tokens):
        target_probs = mx.softmax(logits[0, i, :], axis=-1)
        target_token = mx.argmax(target_probs).item()

        if target_token == draft_token:
            # Accept
            accepted_count += 1
            # Update target cache to include this token
            self.target_cache = self._advance_cache(
                self.target_cache, next_cache, step=i
            )
        else:
            # Reject: rollback draft cache
            self._rollback_draft_cache(accepted_count)
            break

    return accepted_count
```

### 3. 集成到 Scheduler

```python
# src/omlx/scheduler.py

class Scheduler:
    def __init__(self, model, tokenizer, ...):
        # ...existing code...

        # Speculative Decoding (optional)
        self.speculative_engine = None
        if config.get("enable_speculative_decoding"):
            draft_model_path = config.get("draft_model_path")
            if draft_model_path:
                self.speculative_engine = SpeculativeDecodingEngine(
                    target_model=model,
                    target_tokenizer=tokenizer,
                    draft_model=load_draft_model(draft_model_path),
                    draft_tokenizer=load_draft_tokenizer(draft_model_path),
                    num_speculative_tokens=config.get("num_speculative_tokens", 4)
                )

    def _generate_tokens(self, request: Request):
        if self.speculative_engine:
            # Use speculative decoding
            for token in self.speculative_engine.generate_speculative(
                prompt_tokens=request.prompt_tokens,
                max_tokens=request.sampling_params.max_tokens,
                sampling_params=request.sampling_params
            ):
                yield token
        else:
            # Normal generation
            for token in self._normal_generate(request):
                yield token
```

## 配置参数

### 环境变量

```bash
# 启用 Speculative Decoding
export OMLX_ENABLE_SPECULATIVE_DECODING=true

# Draft model 路径
export OMLX_DRAFT_MODEL_PATH=~/.omlx/models/Qwen3.5-0.8B-MLX-4bit

# 候选 tokens 数量（K）
export OMLX_NUM_SPECULATIVE_TOKENS=4

# Draft model 最大并发数（限制内存占用）
export OMLX_DRAFT_MAX_BATCH_SIZE=1
```

### SpeculativeConfig

```python
@dataclass
class SpeculativeConfig:
    enabled: bool = False
    draft_model_path: Optional[str] = None
    num_speculative_tokens: int = 4  # K
    draft_max_batch_size: int = 1
    acceptance_threshold: float = 0.0  # 0 = greedy

    @classmethod
    def from_env(cls) -> "SpeculativeConfig":
        return cls(
            enabled=os.getenv("OMLX_ENABLE_SPECULATIVE_DECODING", "false").lower() == "true",
            draft_model_path=os.getenv("OMLX_DRAFT_MODEL_PATH"),
            num_speculative_tokens=int(os.getenv("OMLX_NUM_SPECULATIVE_TOKENS", "4")),
            draft_max_batch_size=int(os.getenv("OMLX_DRAFT_MAX_BATCH_SIZE", "1"))
        )
```

## 实现步骤

### Phase 1: 核心引擎（MVP）
1. ✅ 下载 draft model（Qwen3.5-0.8B-MLX-4bit）
2. ⏳ 实现 `SpeculativeDecodingEngine` 基础类
3. ⏳ 实现 prefill、draft_generate、verify_and_accept
4. ⏳ 单请求验证测试

### Phase 2: Scheduler 集成
5. ⏳ 集成到 Scheduler
6. ⏳ 配置系统（环境变量 + SpeculativeConfig）
7. ⏳ 多请求并发测试

### Phase 3: 优化和监控
8. ⏳ KV Cache 复用优化
9. ⏳ 性能指标统计（接受率、加速比）
10. ⏳ 自适应 K 值（根据接受率动态调整）

## 性能预期

### 基准对比

| 场景 | Baseline | Speculative (K=4) | 加速比 |
|------|----------|-------------------|--------|
| 短文本生成（50 tokens） | ~2.5s | ~0.8s | **3.1×** |
| 中等长度（200 tokens） | ~10s | ~3.2s | **3.1×** |
| 长文本（500 tokens） | ~25s | ~8s | **3.1×** |

假设：
- Draft model: 100 tok/s
- Target model: 20 tok/s（baseline）
- 平均接受率 α = 0.7
- K = 4

## 风险和限制

### 限制
1. **内存占用增加**：需要同时加载两个模型
   - Target: ~20GB（35B 6-bit）
   - Draft: ~0.5GB（0.8B 4-bit）
   - 总计：~20.5GB（增加 2.5%）

2. **适用场景**：
   - ✅ 生成式任务（文本生成、对话）
   - ❌ 分类任务（只需要 logits）
   - ✅ 低温度采样（temperature < 0.7）
   - ❌ 高温度采样（temperature > 1.0，接受率低）

3. **Draft-Target 匹配**：
   - ✅ 同系列模型（Qwen 3.5）
   - ✅ 同训练数据/领域
   - ❌ 不同系列（接受率低）

### 风险缓解
- 配置开关：默认禁用，需要显式启用
- 内存监控：集成到现有 MemoryMonitor
- 降级策略：内存不足时自动禁用

## 参考文献

1. Leviathan et al. (2023) - "Fast Inference from Transformers via Speculative Decoding"
2. Chen et al. (2023) - "Accelerating Large Language Model Decoding with Speculative Sampling"
3. MLX Examples - Speculative Decoding: https://github.com/ml-explore/mlx-examples

---

*Author: Solar + DeepSeek-R1 (审判官)*
*Date: 2026-03-15*
*Version: v1.0*
