#!/usr/bin/env python3
"""分析 OpenClaw 使用模式，生成最优 cache 配置

扫描 ~/.openclaw/agents 目录，分析每个 agent 的 system prompt 长度，
生成推荐的 block_size 配置。
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys

# 尝试导入 mlx_lm tokenizer
try:
    from mlx_lm import load
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  Warning: mlx_lm not available, using character-based approximation")


@dataclass
class AgentProfile:
    """Agent 配置文件"""
    agent_id: str
    agent_name: str
    soul_path: str
    soul_length_chars: int
    soul_length_tokens: int
    recommended_block_size: int
    recommended_max_padding: int
    padding_overhead: float  # padding 开销百分比
    num_blocks: int  # 需要的 block 数量


@dataclass
class UsagePattern:
    """使用模式分析结果"""
    total_agents: int
    agents: List[AgentProfile]
    block_size_distribution: Dict[int, int]  # block_size -> agent_count
    avg_soul_length: float
    min_soul_length: int
    max_soul_length: int
    recommended_global_block_size: int


def load_tokenizer(model_path: str):
    """加载 tokenizer"""
    if not HAS_MLX:
        return None

    try:
        print(f"🔄 Loading tokenizer from {model_path}...")
        _, tokenizer = load(model_path)
        print(f"✅ Tokenizer loaded")
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None


def count_tokens(text: str, tokenizer) -> int:
    """计算 token 数量"""
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"⚠️  Token counting failed: {e}, using approximation")

    # Fallback: 字符数估算（中文 ~1.5 chars/token，英文 ~4 chars/token）
    # 混合文本使用 2.5 chars/token 作为估算
    return int(len(text) / 2.5)


def select_optimal_block_size(token_count: int) -> Tuple[int, int]:
    """选择最优 block_size

    Returns:
        (recommended_block_size, max_padding_tokens)
    """
    candidates = [8, 16, 32, 64, 128, 256]

    # 选择能达到 90%+ hit ratio 的最大 block_size
    best_block_size = 32  # 默认
    best_hit_ratio = 0.0

    for block_size in reversed(candidates):
        hit_ratio = (token_count // block_size) * block_size / token_count
        if hit_ratio >= 0.90:
            best_block_size = block_size
            best_hit_ratio = hit_ratio
            break

    # max_padding_tokens 设置为 block_size（确保能 padding 到边界）
    max_padding = best_block_size

    return best_block_size, max_padding


def analyze_agent(agent_dir: Path, tokenizer) -> AgentProfile:
    """分析单个 agent"""
    agent_id = agent_dir.name
    soul_path = agent_dir / "agent" / "SOUL.md"

    if not soul_path.exists():
        print(f"⚠️  {agent_id}: SOUL.md not found, skipping")
        return None

    # 读取 SOUL.md
    soul_content = soul_path.read_text(encoding='utf-8')
    soul_chars = len(soul_content)
    soul_tokens = count_tokens(soul_content, tokenizer)

    # 选择最优 block_size
    block_size, max_padding = select_optimal_block_size(soul_tokens)

    # 计算 padding 后的 token 数
    remainder = soul_tokens % block_size
    padding_needed = block_size - remainder if remainder > 0 else 0
    padded_tokens = soul_tokens + padding_needed
    num_blocks = padded_tokens // block_size

    # 计算 padding 开销
    padding_overhead = (padding_needed / soul_tokens * 100) if soul_tokens > 0 else 0

    # 提取 agent 名称（从 SOUL.md 第一行）
    first_line = soul_content.split('\n')[0]
    agent_name = first_line.replace('#', '').replace('SOUL.md', '').strip()

    return AgentProfile(
        agent_id=agent_id,
        agent_name=agent_name,
        soul_path=str(soul_path),
        soul_length_chars=soul_chars,
        soul_length_tokens=soul_tokens,
        recommended_block_size=block_size,
        recommended_max_padding=max_padding,
        padding_overhead=padding_overhead,
        num_blocks=num_blocks,
    )


def analyze_openclaw_agents(openclaw_dir: Path, tokenizer) -> UsagePattern:
    """分析所有 OpenClaw agents"""
    agents_dir = openclaw_dir / "agents"

    if not agents_dir.exists():
        print(f"❌ Agents directory not found: {agents_dir}")
        return None

    print(f"📂 Scanning agents in {agents_dir}...")

    agents = []
    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        print(f"  🔍 Analyzing {agent_dir.name}...")
        profile = analyze_agent(agent_dir, tokenizer)
        if profile:
            agents.append(profile)
            print(f"    ✅ {profile.agent_name}: {profile.soul_length_tokens} tokens, "
                  f"block_size={profile.recommended_block_size}, "
                  f"padding={profile.padding_overhead:.1f}%")

    if not agents:
        print("❌ No agents found")
        return None

    # 统计 block_size 分布
    block_size_dist = {}
    for agent in agents:
        bs = agent.recommended_block_size
        block_size_dist[bs] = block_size_dist.get(bs, 0) + 1

    # 计算统计数据
    token_counts = [a.soul_length_tokens for a in agents]
    avg_length = sum(token_counts) / len(token_counts)
    min_length = min(token_counts)
    max_length = max(token_counts)

    # 推荐全局 block_size（使用最常见的）
    recommended_global = max(block_size_dist, key=block_size_dist.get)

    return UsagePattern(
        total_agents=len(agents),
        agents=agents,
        block_size_distribution=block_size_dist,
        avg_soul_length=avg_length,
        min_soul_length=min_length,
        max_soul_length=max_length,
        recommended_global_block_size=recommended_global,
    )


def generate_config_file(pattern: UsagePattern, output_path: Path):
    """生成配置文件"""
    config = {
        "analysis_summary": {
            "total_agents": pattern.total_agents,
            "avg_soul_length_tokens": int(pattern.avg_soul_length),
            "min_soul_length_tokens": pattern.min_soul_length,
            "max_soul_length_tokens": pattern.max_soul_length,
            "recommended_global_block_size": pattern.recommended_global_block_size,
            "block_size_distribution": pattern.block_size_distribution,
        },
        "agent_configs": {
            agent.agent_id: {
                "name": agent.agent_name,
                "soul_length_tokens": agent.soul_length_tokens,
                "recommended_block_size": agent.recommended_block_size,
                "max_padding_tokens": agent.recommended_max_padding,
                "padding_overhead_percent": round(agent.padding_overhead, 2),
                "num_blocks": agent.num_blocks,
            }
            for agent in sorted(pattern.agents, key=lambda a: a.soul_length_tokens)
        }
    }

    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n✅ Configuration saved to: {output_path}")


def print_summary(pattern: UsagePattern):
    """打印分析摘要"""
    print("\n" + "=" * 70)
    print("📊 OpenClaw Usage Pattern Analysis")
    print("=" * 70)

    print(f"\n总计 {pattern.total_agents} 个 Agents")
    print(f"平均 System Prompt 长度: {pattern.avg_soul_length:.0f} tokens")
    print(f"最短: {pattern.min_soul_length} tokens")
    print(f"最长: {pattern.max_soul_length} tokens")

    print(f"\n📊 Block Size 分布:")
    for block_size in sorted(pattern.block_size_distribution.keys()):
        count = pattern.block_size_distribution[block_size]
        percentage = count / pattern.total_agents * 100
        bar = "█" * int(percentage / 5)
        print(f"  block_size={block_size:3d}: {count:2d} agents ({percentage:5.1f}%) {bar}")

    print(f"\n💡 推荐全局 block_size: {pattern.recommended_global_block_size}")

    print("\n📋 各 Agent 详细配置:")
    print(f"{'Agent ID':<25} {'Name':<30} {'Tokens':>8} {'Block':>6} {'Padding%':>9} {'Blocks':>7}")
    print("-" * 100)

    for agent in sorted(pattern.agents, key=lambda a: a.soul_length_tokens):
        print(f"{agent.agent_id:<25} {agent.agent_name:<30} "
              f"{agent.soul_length_tokens:>8} {agent.recommended_block_size:>6} "
              f"{agent.padding_overhead:>8.1f}% {agent.num_blocks:>7}")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    # 配置
    openclaw_dir = Path.home() / ".openclaw"
    output_file = Path.home() / "ThunderOMLX" / "openclaw-history-mode.json"
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print("🚀 OpenClaw Usage Pattern Analyzer")
    print("=" * 70)

    # 检查 OpenClaw 目录
    if not openclaw_dir.exists():
        print(f"❌ OpenClaw directory not found: {openclaw_dir}")
        sys.exit(1)

    # 加载 tokenizer
    tokenizer = None
    if model_path.exists():
        tokenizer = load_tokenizer(str(model_path))
    else:
        print(f"⚠️  Model not found at {model_path}, using approximation")

    # 分析 agents
    pattern = analyze_openclaw_agents(openclaw_dir, tokenizer)

    if not pattern:
        print("❌ Analysis failed")
        sys.exit(1)

    # 打印摘要
    print_summary(pattern)

    # 生成配置文件
    generate_config_file(pattern, output_file)

    # 生成 Python 配置代码
    print("\n" + "=" * 70)
    print("🐍 Python Configuration Code")
    print("=" * 70)
    print("\n# openclaw/config/agent_cache_config.py")
    print("AGENT_CACHE_CONFIGS = {")
    for agent in sorted(pattern.agents, key=lambda a: a.agent_id):
        print(f'    "{agent.agent_id}": {{')
        print(f'        "block_size": {agent.recommended_block_size},')
        print(f'        "max_padding": {agent.recommended_max_padding},')
        print(f'        # {agent.agent_name}: {agent.soul_length_tokens} tokens, '
              f'{agent.padding_overhead:.1f}% padding overhead')
        print('    },')
    print("}")

    print("\n✅ Analysis complete!")
    print(f"📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
