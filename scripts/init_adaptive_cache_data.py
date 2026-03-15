"""
初始化 Adaptive Cache Optimizer 数据

基于 OpenClaw 真实使用数据分析结果,预填充数据库,
使系统可以立即运行优化策略而无需等待数据收集。

数据来源: .solar/OPENCLAW_REAL_DATA_ANALYSIS.md
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


# 基于真实分析的 Agent 配置
AGENT_PROFILES = {
    "chief-of-staff": {
        "name": "爱音玛利亚 (Chief of Staff)",
        "system_prompt_length": 397,
        "current_block_size": 128,
        "optimal_block_size": 64,
        "current_padding_overhead": 29.0,  # 使用 block_size=128 时
        "optimal_padding_overhead": 12.8,  # 使用 block_size=64 时
        "typical_queries": [
            ("帮我分析一下项目进度", 15),
            ("总结最近的工作", 12),
            ("检查任务完成情况", 14),
            ("准备周报", 10),
            ("分析团队绩效", 13),
        ],
    },
    "product-strategist": {
        "name": "产品策略师",
        "system_prompt_length": 631,
        "current_block_size": 64,
        "optimal_block_size": 64,
        "current_padding_overhead": 1.4,
        "optimal_padding_overhead": 1.4,  # 已经最优
        "typical_queries": [
            ("分析市场趋势", 11),
            ("制定产品路线图", 14),
            ("用户需求分析", 13),
            ("竞品对比分析", 12),
            ("制定增长策略", 12),
        ],
    },
    "ux-designer": {
        "name": "UX 设计师",
        "system_prompt_length": 830,
        "current_block_size": 256,
        "optimal_block_size": 64,
        "current_padding_overhead": 23.4,  # 使用 block_size=256 时
        "optimal_padding_overhead": 2.4,   # 使用 block_size=64 时
        "typical_queries": [
            ("设计登录页面", 12),
            ("优化用户流程", 13),
            ("设计移动端界面", 14),
            ("可用性测试报告", 15),
            ("设计系统规范", 13),
        ],
    },
    "ai-engineer": {
        "name": "AI 工程师",
        "system_prompt_length": 2262,
        "current_block_size": 256,
        "optimal_block_size": 256,
        "current_padding_overhead": 1.9,
        "optimal_padding_overhead": 1.9,  # 已经最优
        "typical_queries": [
            ("优化模型推理性能", 16),
            ("实现缓存策略", 13),
            ("分析性能瓶颈", 13),
            ("设计自适应算法", 15),
            ("实现 A/B 测试框架", 18),
        ],
    },
}


def calculate_padding_tokens(prompt_length: int, block_size: int, max_padding: int = 64) -> int:
    """计算 padding tokens 数量"""
    remainder = prompt_length % block_size
    if remainder == 0:
        return 0
    padding_needed = block_size - remainder
    return padding_needed if padding_needed <= max_padding else 0


def generate_inference_data(
    agent_id: str,
    profile: dict,
    num_requests: int = 50,
    days_back: int = 7,
) -> list:
    """
    为一个 agent 生成模拟推理数据

    Args:
        agent_id: Agent 标识
        profile: Agent 配置信息
        num_requests: 生成请求数量
        days_back: 数据时间范围(天)

    Returns:
        推理数据列表
    """
    data = []

    system_prompt_length = profile["system_prompt_length"]
    current_block_size = profile["current_block_size"]
    typical_queries = profile["typical_queries"]

    # 生成过去 N 天的数据
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    for i in range(num_requests):
        # 随机选择一个典型查询
        query_text, query_length = random.choice(typical_queries)

        # 添加随机变化(±20%)
        query_length = int(query_length * random.uniform(0.8, 1.2))

        total_prompt_length = system_prompt_length + query_length

        # 计算 padding
        padding_tokens = calculate_padding_tokens(
            total_prompt_length, current_block_size
        )

        # Cache hit ratio (模拟)
        # System prompt 部分通常能命中缓存(90-100%)
        # User query 部分是新的(0% 命中)
        cache_hit_ratio = system_prompt_length / total_prompt_length

        # 根据 cache hit ratio 确定 skip logic type
        if cache_hit_ratio >= 1.0:
            skip_logic_type = "FULL"
        elif cache_hit_ratio >= 0.90:
            skip_logic_type = "APPROXIMATE"
        else:
            skip_logic_type = "NONE"

        # 模拟推理时间
        # Prefill 时间取决于是否 skip 和 prompt 长度
        if skip_logic_type == "FULL":
            prefill_time_ms = 10.0  # FULL SKIP 几乎无 prefill
        elif skip_logic_type == "APPROXIMATE":
            # Approximate skip: 只需 prefill 剩余 tokens
            remaining_tokens = int(total_prompt_length * (1 - cache_hit_ratio))
            prefill_time_ms = remaining_tokens * 0.5  # ~0.5ms per token
        else:
            # No skip: 全部 prefill
            prefill_time_ms = total_prompt_length * 0.5

        # Decode 时间(假设生成 50-100 tokens)
        output_tokens = random.randint(50, 100)
        decode_time_ms = output_tokens * 4.0  # ~4ms per token (MLX typical)

        # 随机时间戳(在时间范围内)
        timestamp = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )

        data.append({
            "agent_id": agent_id,
            "timestamp": timestamp,
            "system_prompt_length": system_prompt_length,
            "user_query_length": query_length,
            "cache_hit_ratio": cache_hit_ratio,
            "skip_logic_type": skip_logic_type,
            "block_size": current_block_size,
            "padding_tokens": padding_tokens,
            "prefill_time_ms": prefill_time_ms,
            "decode_time_ms": decode_time_ms,
        })

    return data


def init_database(db_path: str, requests_per_agent: int = 50):
    """
    初始化数据库并填充数据

    Args:
        db_path: 数据库路径
        requests_per_agent: 每个 agent 生成的请求数
    """
    print("=" * 70)
    print("Adaptive Cache Optimizer - 数据初始化")
    print("=" * 70)

    # 初始化 ACO
    print(f"\n📂 数据库路径: {db_path}")
    aco = AdaptiveCacheOptimizer(db_path)
    print("✅ 数据库初始化成功")

    # 为每个 agent 生成数据
    total_records = 0

    for agent_id, profile in AGENT_PROFILES.items():
        print(f"\n🤖 生成数据: {agent_id} ({profile['name']})")
        print(f"   System prompt: {profile['system_prompt_length']} tokens")
        print(f"   Current block_size: {profile['current_block_size']}")
        print(f"   Current padding: {profile['current_padding_overhead']:.1f}%")

        # 生成推理数据
        inference_data = generate_inference_data(
            agent_id, profile, num_requests=requests_per_agent
        )

        # 插入数据
        for record in inference_data:
            aco.log_inference(
                agent_id=record["agent_id"],
                system_prompt_length=record["system_prompt_length"],
                user_query_length=record["user_query_length"],
                cache_hit_ratio=record["cache_hit_ratio"],
                skip_logic_type=record["skip_logic_type"],
                block_size=record["block_size"],
                padding_tokens=record["padding_tokens"],
                prefill_time_ms=record["prefill_time_ms"],
                decode_time_ms=record["decode_time_ms"],
            )

        total_records += len(inference_data)
        print(f"   ✅ 插入 {len(inference_data)} 条记录")

    print(f"\n{'=' * 70}")
    print(f"✅ 初始化完成! 总计 {total_records} 条推理记录")
    print(f"{'=' * 70}")

    # 显示统计信息
    print("\n📊 数据统计:")
    for agent_id, profile in AGENT_PROFILES.items():
        stats = aco.get_stats(agent_id=agent_id)
        print(f"\n   {agent_id}:")
        print(f"      记录数: {stats['total_records']}")
        print(f"      当前配置: block_size={profile['current_block_size']}, "
              f"padding={profile['current_padding_overhead']:.1f}%")
        print(f"      优化潜力: 可减少 {profile['current_padding_overhead'] - profile['optimal_padding_overhead']:.1f}% padding "
              f"(切换到 block_size={profile['optimal_block_size']})")

    print(f"\n{'=' * 70}")
    print("🚀 下一步: 运行模式分析,生成优化建议")
    print("   示例:")
    print("   python scripts/analyze_and_optimize.py")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="初始化 Adaptive Cache Optimizer 数据")
    parser.add_argument(
        "--db-path",
        default="~/.cache/thunderomlx/adaptive_cache.db",
        help="数据库路径"
    )
    parser.add_argument(
        "--requests-per-agent",
        type=int,
        default=50,
        help="每个 agent 生成的请求数(默认 50)"
    )

    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()

    # 如果数据库已存在,询问是否覆盖
    if db_path.exists():
        response = input(f"\n⚠️  数据库已存在: {db_path}\n是否覆盖? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ 已取消")
            sys.exit(0)
        db_path.unlink()
        print(f"🗑️  已删除旧数据库")

    init_database(str(db_path), args.requests_per_agent)
