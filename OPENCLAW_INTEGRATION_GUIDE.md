# OpenClaw 集成指南

**目标**: 将优化后的 Agent Cache 配置集成到 OpenClaw 项目
**预期收益**: 平均 padding 开销减少 65%，保持 100% cache hit

---

## 📋 集成步骤

### Step 1: 复制配置文件到 OpenClaw 项目

```bash
# 假设 OpenClaw 项目路径为 ~/openclaw
mkdir -p ~/openclaw/config
cp ~/ThunderOMLX/openclaw_agent_cache_config.py ~/openclaw/config/agent_cache_config.py
```

---

### Step 2: 修改 ThunderOMLX Engine 初始化

在创建 Agent 时，根据 agent_id 动态配置 scheduler：

```python
# ~/openclaw/agent_factory.py (或类似文件)
from pathlib import Path
import sys

# 导入配置
sys.path.insert(0, str(Path.home() / "openclaw" / "config"))
from agent_cache_config import get_agent_cache_config

from omlx.engine_core import EngineCore, EngineConfig
from omlx.scheduler import SchedulerConfig
from mlx_lm import load


def create_thunderllama_agent(agent_id: str, system_prompt: str, model_path: str):
    """创建 ThunderLLAMA Agent（带优化的 cache 配置）

    Args:
        agent_id: Agent 标识符（如 "chief-of-staff"）
        system_prompt: System prompt（从 SOUL.md 读取）
        model_path: 模型路径

    Returns:
        配置好的 EngineCore 实例
    """
    # 1. 获取 agent 专属 cache 配置
    cache_config = get_agent_cache_config(agent_id)

    print(f"🔧 Creating {agent_id} with cache config:")
    print(f"   block_size: {cache_config['block_size']}")
    print(f"   max_padding: {cache_config['max_padding']}")

    # 2. 加载模型
    model, tokenizer = load(model_path)

    # 3. 创建 scheduler 配置（使用 agent 专属配置）
    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        paged_cache_block_size=cache_config["block_size"],  # ⚡ Agent 专属
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / f"agent_{agent_id}"),
        model_name=model_path,
        # ⚡ 启用 Prompt Padding（策略 1）
        enable_prompt_padding=True,
        max_padding_tokens=cache_config["max_padding"],  # ⚡ Agent 专属
    )

    # 4. 创建 engine 配置
    engine_config = EngineConfig(
        model_name=model_path,
        scheduler_config=scheduler_config,
    )

    # 5. 创建 engine
    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

    return engine
```

---

### Step 3: 在 OpenClaw 中使用

```python
# ~/openclaw/main.py (示例)
import asyncio
from agent_factory import create_thunderllama_agent
from pathlib import Path


async def main():
    # 模型路径
    model_path = str(Path.home() / "models" / "qwen3.5-35b-mlx")

    # 创建各个 Agent
    agents = {}

    # Agent 1: chief-of-staff
    print("\n🤖 Creating chief-of-staff...")
    soul_path = Path.home() / ".openclaw" / "agents" / "chief-of-staff" / "agent" / "SOUL.md"
    system_prompt = soul_path.read_text(encoding='utf-8')
    agents["chief-of-staff"] = create_thunderllama_agent(
        agent_id="chief-of-staff",
        system_prompt=system_prompt,
        model_path=model_path,
    )
    await agents["chief-of-staff"].start()

    # Agent 2: ai-engineer
    print("\n🤖 Creating ai-engineer...")
    soul_path = Path.home() / ".openclaw" / "agents" / "ai-engineer" / "agent" / "SOUL.md"
    system_prompt = soul_path.read_text(encoding='utf-8')
    agents["ai-engineer"] = create_thunderllama_agent(
        agent_id="ai-engineer",
        system_prompt=system_prompt,
        model_path=model_path,
    )
    await agents["ai-engineer"].start()

    # ... 其他 agents

    # 使用示例
    from omlx.request import SamplingParams

    print("\n💬 Testing chief-of-staff...")
    response = await agents["chief-of-staff"].generate(
        prompt=system_prompt + "\n\nUser: 帮我分析一下项目进度",
        sampling_params=SamplingParams(max_tokens=100),
    )
    print(f"Response: {response.output_text[:100]}...")

    print("\n💬 Testing ai-engineer...")
    response = await agents["ai-engineer"].generate(
        prompt=system_prompt + "\n\nUser: 如何优化模型推理性能？",
        sampling_params=SamplingParams(max_tokens=100),
    )
    print(f"Response: {response.output_text[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔍 验证步骤

### 验证 1: 检查配置是否生效

运行时应该看到类似日志：

```
🔧 Creating chief-of-staff with cache config:
   block_size: 64
   max_padding: 64

⚡ Prompt Padding: 397 → 448 tokens (+51 padding) for 100% cache alignment to block_size=64
```

**验证点**:
- ✅ `block_size=64`（不是 128）
- ✅ Padding 数量约为 51 tokens（不是 115）
- ✅ 最终对齐到 `448 = 64 × 7`

---

### 验证 2: 检查 Cache Hit Ratio

在第二次相同 prompt 调用时，应该看到：

```
Cache match result: can_skip=True, skip_reason=full, cache_hit_ratio=100.0%, remaining=0
✨ FULL SKIP enabled for request: 100% cache hit (448 tokens), skipping prefill computation
```

**验证点**:
- ✅ `cache_hit_ratio=100.0%`
- ✅ `skip_reason=full`（FULL SKIP）
- ✅ `remaining=0`

---

### 验证 3: 性能测试

运行重复推理测试：

```python
import time

# 第一次推理（冷启动）
start = time.perf_counter()
response1 = await engine.generate(prompt=prompt, sampling_params=params)
time1 = (time.perf_counter() - start) * 1000

# 第二次推理（应该触发 FULL SKIP）
start = time.perf_counter()
response2 = await engine.generate(prompt=prompt, sampling_params=params)
time2 = (time.perf_counter() - start) * 1000

print(f"第 1 次: {time1:.2f} ms")
print(f"第 2 次: {time2:.2f} ms (应该显著更快)")
print(f"加速比: {time1 / time2:.1f}x")
```

**预期结果**:
- 第 1 次: ~10,000-30,000 ms（包含 prefill）
- 第 2 次: ~200-500 ms（FULL SKIP，只有 decode）
- 加速比: **50-100x**

---

## 📊 各 Agent 预期表现

| Agent ID | System Prompt | block_size | Padding | 预期表现 |
|----------|--------------|------------|---------|---------|
| chief-of-staff | 397 tokens | 64 | ~51 tokens (12.8%) | 100% hit, FULL SKIP ✅ |
| product-strategist | 631 tokens | 64 | ~9 tokens (1.4%) | 100% hit, FULL SKIP ✅ |
| ux-designer | 830 tokens | 64 | ~20 tokens (2.4%) | 100% hit, FULL SKIP ✅ |
| ai-engineer | 2262 tokens | 256 | ~42 tokens (1.9%) | 100% hit, FULL SKIP ✅ |

---

## 🚨 常见问题

### Q1: 如果新增 Agent，如何配置？

**方法 1: 手动添加到配置文件**

```python
# ~/openclaw/config/agent_cache_config.py
AGENT_CACHE_CONFIGS = {
    # ... 现有配置 ...
    "new-agent": {
        "block_size": 64,  # 根据 system prompt 长度选择
        "max_padding": 64,
    },
}
```

**方法 2: 使用默认配置**

如果不添加到配置文件，会自动使用 `DEFAULT_CACHE_CONFIG`（block_size=64）

**推荐**: 运行 `analyze_openclaw_usage.py` 重新分析，获取推荐配置

---

### Q2: 如何选择 block_size？

根据 System Prompt 长度：

| System Prompt 长度 | 推荐 block_size |
|-------------------|----------------|
| 50-200 tokens | 16 |
| 200-500 tokens | 64 |
| 500-1000 tokens | 64 或 128 |
| 1000-2000 tokens | 128 或 256 |
| 2000+ tokens | 256 |

**验证方法**:
```python
# 计算 padding 开销
tokens = len(tokenizer.encode(system_prompt))
block_size = 64
remainder = tokens % block_size
padding_needed = block_size - remainder if remainder > 0 else 0
padding_overhead = padding_needed / tokens * 100

print(f"Padding 开销: {padding_overhead:.1f}%")
# 如果 > 15%，考虑使用更小的 block_size
```

---

### Q3: 缓存目录在哪里？

每个 Agent 使用独立的缓存目录：

```
~/.cache/omlx/agent_chief-of-staff/
~/.cache/omlx/agent_ai-engineer/
~/.cache/omlx/agent_ux-designer/
~/.cache/omlx/agent_product-strategist/
```

**清理缓存**:
```bash
rm -rf ~/.cache/omlx/agent_*
```

---

### Q4: 如何确认配置生效？

检查日志中的 `block_size` 值：

```bash
# 应该看到不同的 block_size
grep "block_size" /tmp/llama-server.log

# chief-of-staff: block_size=64
# ai-engineer: block_size=256
```

---

## 📋 完整示例代码

完整的集成示例代码在：
- **配置文件**: `~/ThunderOMLX/openclaw_agent_cache_config.py`
- **集成指南**: 本文件
- **分析报告**: `~/ThunderOMLX/.solar/OPENCLAW_REAL_DATA_ANALYSIS.md`

---

## ✅ 完成检查清单

集成完成后，检查以下项目：

- [ ] 配置文件已复制到 OpenClaw 项目
- [ ] `create_thunderllama_agent()` 函数已实现
- [ ] 各 Agent 创建时使用了正确的配置
- [ ] 日志中显示正确的 block_size
- [ ] Prompt Padding 日志显示
- [ ] Cache Hit Ratio 达到 100%
- [ ] FULL SKIP 触发
- [ ] 第二次推理性能提升 50-100x

---

**签署**: Solar (CEO + 战略家 + 治理官)
**日期**: 2026-03-14
**状态**: ✅ 配置已优化，等待集成到 OpenClaw
