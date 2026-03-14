"""OpenClaw Agent Cache 配置（基于实际使用数据优化）

数据来源: ~/.openclaw/agents SOUL.md 分析
分析日期: 2026-03-14
优化目标: 减少 padding 开销，保持 100% cache hit

分析结果:
- chief-of-staff (397 tokens): block_size 128→64, padding 29%→12.8% (-56%)
- product-strategist (631 tokens): block_size 64 保持, padding 1.4%
- ux-designer (830 tokens): block_size 256→64, padding 23.4%→2.4% (-90%)
- ai-engineer (2262 tokens): block_size 256 保持, padding 1.9%

综合收益: 平均 padding 开销减少 65%
"""

AGENT_CACHE_CONFIGS = {
    "chief-of-staff": {
        "block_size": 64,
        "max_padding": 64,
        # 爱音玛利亚（Aine Maria）: 397 tokens
        # 优化: block_size 128→64, padding 29%→12.8% (减少 56%)
    },
    "product-strategist": {
        "block_size": 64,
        "max_padding": 64,
        # 产品策略师: 631 tokens
        # 保持原配置: padding 1.4% (非常好)
    },
    "ux-designer": {
        "block_size": 64,
        "max_padding": 64,
        # UX设计师: 830 tokens
        # 优化: block_size 256→64, padding 23.4%→2.4% (减少 90%)
    },
    "ai-engineer": {
        "block_size": 256,
        "max_padding": 256,
        # AI工程师: 2262 tokens
        # 保持原配置: padding 1.9% (非常好)
    },
}

# 默认配置（用于未明确配置的 agent）
DEFAULT_CACHE_CONFIG = {
    "block_size": 64,
    "max_padding": 64,
}


def get_agent_cache_config(agent_id: str) -> dict:
    """获取 agent 的缓存配置

    Args:
        agent_id: Agent 标识符

    Returns:
        包含 block_size 和 max_padding 的配置字典
    """
    return AGENT_CACHE_CONFIGS.get(agent_id, DEFAULT_CACHE_CONFIG)


def print_config_summary():
    """打印配置摘要"""
    print("=" * 70)
    print("OpenClaw Agent Cache Configuration")
    print("=" * 70)
    print(f"\n配置的 Agent 数量: {len(AGENT_CACHE_CONFIGS)}")
    print(f"默认 block_size: {DEFAULT_CACHE_CONFIG['block_size']}")
    print("\n各 Agent 配置:")
    print(f"{'Agent ID':<25} {'Block Size':>12} {'Max Padding':>12}")
    print("-" * 70)

    for agent_id, config in sorted(AGENT_CACHE_CONFIGS.items()):
        print(f"{agent_id:<25} {config['block_size']:>12} {config['max_padding']:>12}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_config_summary()
