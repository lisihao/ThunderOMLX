# OpenClaw 多 Agent 场景策略分析

**日期**: 2026-03-14
**场景**: OpenClaw 多 Agent 系统（5-8个 agent）
**用途**: 开发、研究、技术分析、工作、部分生活

---

## 🎯 场景特征分析

### 多 Agent 特点

```
OpenClaw 系统
├── Agent 1: 开发专家 (system prompt: ~300 tokens)
├── Agent 2: 研究助手 (system prompt: ~450 tokens)
├── Agent 3: 技术分析师 (system prompt: ~200 tokens)
├── Agent 4: 代码审查员 (system prompt: ~350 tokens)
├── Agent 5: 文档生成器 (system prompt: ~150 tokens)
├── Agent 6: 项目管理 (system prompt: ~250 tokens)
├── Agent 7: 生活助手 (system prompt: ~100 tokens)
└── Agent 8: 通用助手 (system prompt: ~180 tokens)
```

### Prompt 结构

```
每个请求 = System Prompt（固定，agent 特定） + User Query（动态）

Agent 1 请求:
  [System: 300 tokens] + [User: 50 tokens] = 350 tokens

Agent 2 请求:
  [System: 450 tokens] + [User: 30 tokens] = 480 tokens
```

### 关键特性

| 特性 | 描述 |
|------|------|
| **System Prompt** | 固定（每个 agent 特定） |
| **System Prompt 长度** | 不同（100-450 tokens） |
| **User Query** | 动态变化 |
| **User Query 长度** | 较短（通常 < 100 tokens） |
| **重复率** | 高（同一 agent 的 system prompt 重复） |
| **跨 Agent 复用** | 不可能（system prompt 不同） |
| **同 Agent 复用** | 关键！（system prompt 固定） |

---

## 🔍 缓存复用分析

### 误区：跨 Agent 缓存复用

**错误假设**: 不同 agent 使用不同 block_size 会降低缓存复用

**真相**:
```
Agent 1 的 system prompt ≠ Agent 2 的 system prompt
→ 即使使用相同 block_size，缓存也无法跨 agent 复用
→ 跨 agent 缓存复用本来就不存在！
```

**结论**: **跨 agent 缓存隔离不是问题**（因为本来就不能复用）

---

### 关键：同 Agent 缓存复用

**重要场景**: 用户与同一个 agent 多次交互

```
用户连续与 Agent 1 交互:
  Request 1: [System: 300 tokens] + [Query 1: 50 tokens]
  Request 2: [System: 300 tokens] + [Query 2: 30 tokens]
  Request 3: [System: 300 tokens] + [Query 3: 70 tokens]

关键: System Prompt 的 300 tokens 重复 3 次
     → 需要缓存复用 ✅
```

**缓存复用条件**:
1. ✅ 相同的 system prompt tokens
2. ✅ 相同的 block_size
3. ✅ 相同的 cache key

**结论**: **同 agent 缓存复用是核心价值**

---

## 📊 策略重新评估

### 策略 1: Prompt Padding（固定 block_size）

**实现**:
```python
scheduler_config = SchedulerConfig(
    paged_cache_block_size=32,  # 全局固定
    enable_prompt_padding=True,
    max_padding_tokens=64,
)
```

**效果**:
```
Agent 1 (300 tokens system):
  - 300 % 32 = 12 (剩余)
  - padding 20 tokens → 320 tokens (10 blocks) → 100% hit ✅

Agent 2 (450 tokens system):
  - 450 % 32 = 2 (剩余)
  - padding 30 tokens → 480 tokens (15 blocks) → 100% hit ✅

Agent 7 (100 tokens system):
  - 100 % 32 = 4 (剩余)
  - padding 28 tokens → 128 tokens (4 blocks) → 100% hit ✅
```

**优点**:
- ✅ **所有 agent 都 100% hit**
- ✅ **固定 block_size = 完美的同 agent 缓存复用**
- ✅ **实现简单**，已验证

**缺点**:
- ⚠️ **所有 agent 都用 block_size=32 可能不是最优**
  - Agent 7 (100 tokens) 用 block_size=32 → 需要 padding 28 tokens（太多）
  - Agent 7 用 block_size=16 更合适 → 只需 padding 12 tokens
- ⚠️ **快照开销不平衡**
  - Agent 7: 128 tokens / 32 = 4 个快照（可能偏多）
  - Agent 7: 128 tokens / 16 = 8 个快照（更多，但每个更小）

**评分**: ⭐⭐⭐⭐ (good，但不是最优)

---

### 策略 2: 智能 block_size（每个 agent 独立优化）

**关键改进**: 为每个 agent 选择**一次**最优 block_size（而不是每个请求都动态选择）

**实现思路**:
```python
# 方案 A: 用户手动为每个 agent 配置 block_size
agent_configs = {
    "agent_1": {"block_size": 32},  # 300 tokens system
    "agent_2": {"block_size": 64},  # 450 tokens system
    "agent_7": {"block_size": 16},  # 100 tokens system
}

# 方案 B: 自动检测（首次调用时选择，然后固定）
# 每个 agent 首次调用时：
#   1. 分析 system prompt 长度
#   2. 从候选池选择最优 block_size
#   3. 缓存这个决策（固定下来）
#   4. 后续调用都使用这个 block_size
```

**效果**:
```
Agent 1 (300 tokens system):
  - 自动选择: block_size=32 (300/32=9.375 → 93.75% hit)
  - 或 padding: 300→320 (100% hit) ✅

Agent 2 (450 tokens system):
  - 自动选择: block_size=64 (450/64=7.03 → 98.2% hit)
  - 或 padding: 450→512 (100% hit) ✅

Agent 7 (100 tokens system):
  - 自动选择: block_size=16 (100/16=6.25 → 96% hit)
  - 或 padding: 100→112 (100% hit) ✅
```

**优点**:
- ✅ **每个 agent 都用最适合的 block_size**
- ✅ **同 agent 缓存复用完美**（block_size 固定）
- ✅ **跨 agent 缓存隔离无影响**（本来就不能复用）
- ✅ **快照开销优化**（大 agent 用大 block，小 agent 用小 block）

**缺点**:
- ⚠️ **需要管理每个 agent 的 block_size**
- ⚠️ **实现复杂度中等**

**评分**: ⭐⭐⭐⭐⭐ (optimal for multi-agent)

---

### 策略 3: 降低阈值（保底）

**作用**: 覆盖策略1或策略2优化不到的 corner case

**示例**:
```
Agent X (375 tokens system):
  - block_size=32: 375/32=11.72 → 88.5% hit < 90% → 不触发 Skip Logic
  - 策略3: 降低到 80% → 88.5% > 80% → 触发 APPROXIMATE SKIP ✅
```

**评分**: ⭐⭐⭐ (有一定补充价值)

---

## 💡 推荐方案

### 方案 A: 混合策略（推荐 ⭐⭐⭐⭐⭐）

**核心思路**: 有限 block_size 池 + 智能选择 + Padding

```python
scheduler_config = SchedulerConfig(
    enable_mixed_strategy=True,
    mixed_strategy_block_sizes=[16, 32, 64],  # 限制为 3 个
    max_padding_tokens=32,
    approx_threshold=0.85,  # 策略3保底
)
```

**逻辑**:
```
每个 agent 首次调用时:
  Step 1: 分析 system prompt 长度
  Step 2: 从 [16, 32, 64] 中选择最优 block_size
          - 优先选择能达到 90%+ hit 的最大 block_size
  Step 3: 如果 hit ratio < 90%，padding 到边界
  Step 4: 缓存这个决策（agent_id → block_size）

后续调用:
  - 直接使用缓存的 block_size
  - 同 agent 缓存完美复用 ✅
```

**效果预估**:
```
Agent 1 (300 tokens):
  - 选择 block_size=32 → 93.75% hit → padding 到 320 → 100% hit ✅

Agent 2 (450 tokens):
  - 选择 block_size=64 → 98.2% hit → padding 到 512 → 100% hit ✅

Agent 7 (100 tokens):
  - 选择 block_size=16 → 96% hit → padding 到 112 → 100% hit ✅

所有 agent: 100% hit → FULL SKIP → 55-78x 提升 ✅
同时: 每个 agent 用最优 block_size → 快照开销最优 ✅
```

**优点**:
- ✅ **结合策略1和策略2的优势**
- ✅ **每个 agent 独立优化**
- ✅ **同 agent 缓存完美复用**
- ✅ **跨 agent 缓存隔离无影响**（只有 3 种 block_size）
- ✅ **策略3保底覆盖 corner case**

**缺点**:
- ⚠️ 实现复杂度中等
- ⚠️ 需要管理 agent_id → block_size 的映射

**评分**: ⭐⭐⭐⭐⭐ (最推荐)

---

### 方案 B: 手动配置（实用 ⭐⭐⭐⭐）

**核心思路**: 用户为每个 agent 手动配置最优 block_size

```python
# openclaw/agents/agent_config.py
AGENT_CACHE_CONFIGS = {
    "dev_expert": {
        "block_size": 32,
        "max_padding": 32,
    },
    "research_assistant": {
        "block_size": 64,
        "max_padding": 64,
    },
    "tech_analyst": {
        "block_size": 32,
        "max_padding": 32,
    },
    "life_helper": {
        "block_size": 16,
        "max_padding": 16,
    },
    # ... 其他 agent
}
```

**OpenClaw 集成**:
```python
# 创建 agent 时加载配置
def create_agent(agent_id: str, system_prompt: str):
    config = AGENT_CACHE_CONFIGS.get(agent_id, {"block_size": 32})

    scheduler_config = SchedulerConfig(
        paged_cache_block_size=config["block_size"],
        enable_prompt_padding=True,
        max_padding_tokens=config["max_padding"],
    )

    return Agent(agent_id, system_prompt, scheduler_config)
```

**优点**:
- ✅ **简单直接**，无需复杂逻辑
- ✅ **用户可控**，明确每个 agent 的配置
- ✅ **同 agent 缓存完美复用**

**缺点**:
- ⚠️ 需要用户手动配置每个 agent
- ⚠️ 新增 agent 需要手动添加配置

**评分**: ⭐⭐⭐⭐ (实用，易理解)

---

### 方案 C: 保持策略1（保守 ⭐⭐⭐）

**核心思路**: 所有 agent 用统一的 block_size=32

```python
scheduler_config = SchedulerConfig(
    paged_cache_block_size=32,
    enable_prompt_padding=True,
    max_padding_tokens=64,
)
```

**优点**:
- ✅ **已验证**，生产可用
- ✅ **简单**，无需额外配置

**缺点**:
- ⚠️ 不是所有 agent 的最优选择
- ⚠️ 小 agent（如 100 tokens）padding 开销较大

**评分**: ⭐⭐⭐ (可用，但不是最优)

---

## 📊 方案对比

| 方案 | 复杂度 | 性能 | 灵活性 | 同 Agent 复用 | 跨 Agent 隔离 | 推荐度 |
|------|--------|------|--------|--------------|--------------|--------|
| **A: 混合策略** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 完美 | ✅ 无影响 | ⭐⭐⭐⭐⭐ |
| **B: 手动配置** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ 完美 | ✅ 无影响 | ⭐⭐⭐⭐ |
| **C: 统一策略1** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ✅ 完美 | ✅ 无影响 | ⭐⭐⭐ |

---

## 🎯 实施建议

### 短期（立即可用）: 方案 B - 手动配置

**步骤**:

1. **统计每个 agent 的 system prompt 长度**
   ```bash
   # 在 OpenClaw 中添加统计脚本
   python scripts/analyze_agent_prompts.py

   # 输出：
   # Agent 1: 300 tokens → 推荐 block_size=32
   # Agent 2: 450 tokens → 推荐 block_size=64
   # Agent 7: 100 tokens → 推荐 block_size=16
   ```

2. **创建配置文件**
   ```python
   # openclaw/config/agent_cache_config.py
   AGENT_CACHE_CONFIGS = {
       "dev_expert": {"block_size": 32, "max_padding": 32},
       "research_assistant": {"block_size": 64, "max_padding": 64},
       "tech_analyst": {"block_size": 32, "max_padding": 32},
       "code_reviewer": {"block_size": 32, "max_padding": 32},
       "doc_generator": {"block_size": 16, "max_padding": 16},
       "project_manager": {"block_size": 32, "max_padding": 32},
       "life_helper": {"block_size": 16, "max_padding": 16},
       "general_assistant": {"block_size": 16, "max_padding": 16},
   }
   ```

3. **集成到 OpenClaw**
   ```python
   # openclaw/agent_factory.py
   from config.agent_cache_config import AGENT_CACHE_CONFIGS

   def create_agent(agent_id: str, system_prompt: str):
       config = AGENT_CACHE_CONFIGS.get(
           agent_id,
           {"block_size": 32, "max_padding": 32}  # 默认值
       )

       scheduler_config = SchedulerConfig(
           paged_cache_block_size=config["block_size"],
           enable_prompt_padding=True,
           max_padding_tokens=config["max_padding"],
       )

       return ThunderLLAMAAgent(
           agent_id=agent_id,
           system_prompt=system_prompt,
           scheduler_config=scheduler_config,
       )
   ```

4. **测试验证**
   ```python
   # 测试每个 agent 的缓存命中率
   python tests/test_agent_cache_hit_ratio.py

   # 预期输出：
   # Agent 1: 100% cache hit ✅
   # Agent 2: 100% cache hit ✅
   # Agent 7: 100% cache hit ✅
   ```

**预期收益**:
- ✅ 所有 agent 都达到 100% cache hit
- ✅ 每个 agent 用最优 block_size（快照开销最优）
- ✅ 同 agent 缓存完美复用
- ✅ 实现简单，易维护

---

### 中期（进阶优化）: 方案 A - 混合策略

**步骤**:

1. **实现自动检测逻辑**
   ```python
   # scheduler.py
   def _select_optimal_block_size_for_agent(
       self,
       agent_id: str,
       system_prompt_tokens: List[int]
   ) -> int:
       """为 agent 选择最优 block_size（只在首次调用时执行）"""

       # 检查缓存
       if agent_id in self._agent_block_size_cache:
           return self._agent_block_size_cache[agent_id]

       # 分析 system prompt 长度
       prompt_length = len(system_prompt_tokens)
       candidates = self.config.mixed_strategy_block_sizes  # [16, 32, 64]

       # 选择最优 block_size
       best_block_size = 32
       best_hit_ratio = 0.0

       for block_size in reversed(candidates):
           hit_ratio = (prompt_length // block_size) * block_size / prompt_length
           if hit_ratio >= 0.90 and hit_ratio > best_hit_ratio:
               best_block_size = block_size
               best_hit_ratio = hit_ratio

       # 缓存决策
       self._agent_block_size_cache[agent_id] = best_block_size

       logger.info(
           f"⚡ Agent {agent_id}: selected block_size={best_block_size} "
           f"for {prompt_length} tokens system prompt (hit_ratio={best_hit_ratio:.1%})"
       )

       return best_block_size
   ```

2. **集成到 add_request**
   ```python
   def add_request(self, request: Request):
       # Tokenize
       if request.prompt_token_ids is None:
           request.prompt_token_ids = self.tokenizer.encode(request.prompt)

       # ⚡ 混合策略：为每个 agent 选择最优 block_size
       if self.config.enable_mixed_strategy:
           optimal_block_size = self._select_optimal_block_size_for_agent(
               request.agent_id,  # 需要在 Request 中添加 agent_id 字段
               request.prompt_token_ids
           )

           # 临时覆盖 block_size（仅用于当前 agent）
           self.config.paged_cache_block_size = optimal_block_size

       # ⚡ Prompt Padding（基于选择的 block_size）
       if self.config.enable_prompt_padding:
           # ... padding 逻辑 ...
   ```

3. **测试验证**
   ```python
   # 测试自动选择逻辑
   python tests/test_mixed_strategy.py

   # 预期：
   # Agent 1 (300 tokens): block_size=32 → padding → 100% hit ✅
   # Agent 2 (450 tokens): block_size=64 → padding → 100% hit ✅
   # Agent 7 (100 tokens): block_size=16 → padding → 100% hit ✅
   ```

---

## 📋 OpenClaw 集成清单

### 必要修改

1. **Request 类添加 agent_id 字段**
   ```python
   # omlx/request.py
   @dataclass
   class Request:
       # ... 现有字段 ...
       agent_id: Optional[str] = None  # 新增：标识 agent
   ```

2. **SchedulerConfig 添加混合策略配置**
   ```python
   # omlx/scheduler.py
   @dataclass
   class SchedulerConfig:
       # ... 现有配置 ...

       # 混合策略配置
       enable_mixed_strategy: bool = False
       mixed_strategy_block_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
   ```

3. **Scheduler 添加 agent block_size 缓存**
   ```python
   # omlx/scheduler.py
   class Scheduler:
       def __init__(self, ...):
           # ... 现有初始化 ...
           self._agent_block_size_cache: Dict[str, int] = {}  # 新增
   ```

4. **OpenClaw 传递 agent_id**
   ```python
   # openclaw/agent.py
   def generate(self, user_query: str):
       request = Request(
           prompt=self.system_prompt + user_query,
           agent_id=self.agent_id,  # 传递 agent_id
           sampling_params=self.sampling_params,
       )
       return self.engine.generate(request)
   ```

---

## 📊 预期收益

### 性能提升

| Agent | System Prompt | 方案 C (统一32) | 方案 B/A (优化) | 提升 |
|-------|--------------|----------------|----------------|------|
| Agent 1 | 300 tokens | 100% hit (padding 20) | 100% hit (padding 20) | 持平 |
| Agent 2 | 450 tokens | 100% hit (padding 30) | 100% hit (padding 62) | 持平 |
| Agent 7 | 100 tokens | 100% hit (padding 28) | 100% hit (padding 12) | **padding -57%** ✅ |

### 快照开销优化

| Agent | System Prompt | 方案 C (block=32) | 方案 B/A (优化) | 优化 |
|-------|--------------|------------------|----------------|------|
| Agent 1 | 300 → 320 | 10 blocks | 10 blocks | 持平 |
| Agent 2 | 450 → 480 | 15 blocks | 8 blocks (block=64) | **-47%** ✅ |
| Agent 7 | 100 → 128 | 4 blocks | 7 blocks (block=16) | 快照更细粒度 |

### 综合收益

- ✅ **所有 agent**: 100% cache hit → FULL SKIP → 55-78x
- ✅ **Padding 效率**: 小 agent 减少 57% padding 开销
- ✅ **快照优化**: 大 agent 减少 47% 快照数量
- ✅ **缓存复用**: 同 agent 完美复用，跨 agent 无影响

---

## 🎯 最终推荐

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   OpenClaw 多 Agent 场景最佳策略                            │
│                                                             │
│   短期推荐: 方案 B - 手动配置 ⭐⭐⭐⭐                        │
│   • 为每个 agent 配置最优 block_size                        │
│   • 简单、直接、易维护                                      │
│   • 预期收益: 100% hit + 快照优化                           │
│                                                             │
│   中期推荐: 方案 A - 混合策略 ⭐⭐⭐⭐⭐                      │
│   • 自动为每个 agent 选择最优 block_size                    │
│   • 灵活、自适应                                            │
│   • 预期收益: 100% hit + 自动优化                           │
│                                                             │
│   核心洞察:                                                 │
│   • 跨 agent 缓存隔离不是问题（本来就不能复用）             │
│   • 同 agent 缓存复用是关键（需要固定 block_size）          │
│   • 不同 agent 用不同 block_size 是优化，不是劣化           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**签署**: Solar (战略家 + 治理官双签)
**日期**: 2026-03-14
**下一步**: 等待监护人决策（方案 B 或 方案 A）
