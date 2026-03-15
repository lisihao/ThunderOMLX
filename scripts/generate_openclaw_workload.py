"""
基于 OpenClaw 真实使用模式生成模拟生产环境负载数据

模拟 OpenClaw 的多 Agent 系统负载：
- 研究员 (Researcher): 长 prompt，高 cache hit
- 编码员 (Coder): 中等 prompt，中 cache hit
- 分析师 (Analyst): 短 prompt，低 cache hit
- PM: 短 prompt，高 cache hit
- 测试员 (Tester): 中等 prompt，中 cache hit
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


class OpenClawWorkloadGenerator:
    """OpenClaw 负载生成器"""

    def __init__(self, output_dir: str = "openclaw-workload"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # OpenClaw Agent 配置（基于真实使用模式）
        self.agent_profiles = {
            "researcher": {
                "system_prompt_range": (800, 1500),  # 长 system prompt（知识库、工具）
                "user_query_range": (50, 200),
                "cache_hit_range": (0.85, 0.95),  # 研究员经常查询相同主题
                "skip_logic_types": ["APPROXIMATE", "NONE"],
                "skip_logic_weights": [0.8, 0.2],
                "block_size": 128,  # 初始配置（次优）
                "usage_pattern": "working_hours",  # 工作时间为主
                "request_frequency": 0.3,  # 30% 请求来自研究员
            },
            "coder": {
                "system_prompt_range": (600, 1000),
                "user_query_range": (20, 100),
                "cache_hit_range": (0.75, 0.88),
                "skip_logic_types": ["APPROXIMATE", "NONE"],
                "skip_logic_weights": [0.7, 0.3],
                "block_size": 96,  # 初始配置（次优）
                "usage_pattern": "heavy_working_hours",
                "request_frequency": 0.35,  # 35% 请求来自编码员
            },
            "analyst": {
                "system_prompt_range": (400, 700),
                "user_query_range": (30, 150),
                "cache_hit_range": (0.65, 0.80),  # 分析任务多样性高
                "skip_logic_types": ["APPROXIMATE", "NONE"],
                "skip_logic_weights": [0.6, 0.4],
                "block_size": 64,  # 初始配置（相对合理）
                "usage_pattern": "all_day",
                "request_frequency": 0.15,  # 15% 请求来自分析师
            },
            "pm": {
                "system_prompt_range": (300, 500),
                "user_query_range": (10, 80),
                "cache_hit_range": (0.80, 0.92),  # PM 问题重复性高
                "skip_logic_types": ["APPROXIMATE", "NONE"],
                "skip_logic_weights": [0.5, 0.5],
                "block_size": 64,  # 初始配置（合理）
                "usage_pattern": "working_hours",
                "request_frequency": 0.10,  # 10% 请求来自 PM
            },
            "tester": {
                "system_prompt_range": (500, 800),
                "user_query_range": (15, 100),
                "cache_hit_range": (0.70, 0.85),
                "skip_logic_types": ["APPROXIMATE", "NONE"],
                "skip_logic_weights": [0.65, 0.35],
                "block_size": 80,  # 初始配置（次优）
                "usage_pattern": "working_hours",
                "request_frequency": 0.10,  # 10% 请求来自测试员
            },
        }

    def _get_hourly_traffic_multiplier(self, hour: int, pattern: str) -> float:
        """根据时间和使用模式返回流量系数"""
        if pattern == "heavy_working_hours":
            # 编码员：工作时间高峰（9-18）
            if 9 <= hour <= 18:
                return 1.5
            elif 19 <= hour <= 23:
                return 0.8
            else:
                return 0.2
        elif pattern == "working_hours":
            # 研究员/PM/测试员：正常工作时间
            if 9 <= hour <= 18:
                return 1.2
            elif 19 <= hour <= 23:
                return 0.5
            else:
                return 0.1
        else:  # all_day
            # 分析师：全天均匀
            return 1.0

    def _sample_agent_weighted(self) -> str:
        """根据请求频率加权采样 agent"""
        agents = list(self.agent_profiles.keys())
        weights = [self.agent_profiles[a]["request_frequency"] for a in agents]
        return random.choices(agents, weights=weights)[0]

    def _generate_inference_record(
        self, agent_type: str, timestamp: datetime
    ) -> Dict:
        """生成单条推理记录"""
        profile = self.agent_profiles[agent_type]

        # 采样 prompt 长度
        system_prompt_length = random.randint(*profile["system_prompt_range"])
        user_query_length = random.randint(*profile["user_query_range"])
        total_prompt_length = system_prompt_length + user_query_length

        # 采样 cache hit ratio
        cache_hit_ratio = random.uniform(*profile["cache_hit_range"])

        # 采样 skip logic
        skip_logic_type = random.choices(
            profile["skip_logic_types"], weights=profile["skip_logic_weights"]
        )[0]

        # 当前 block_size（初始配置）
        block_size = profile["block_size"]

        # 计算 padding（基于 block_size）
        padding_tokens = (block_size - (total_prompt_length % block_size)) % block_size
        padding_overhead = (
            (padding_tokens / total_prompt_length * 100) if total_prompt_length > 0 else 0
        )

        # 模拟推理时间（基于 prompt 长度和 padding）
        # Prefill: ~0.5ms/token + padding 惩罚
        base_prefill_ms = total_prompt_length * 0.5
        padding_penalty_ms = padding_tokens * 0.3
        prefill_time_ms = base_prefill_ms + padding_penalty_ms + random.uniform(-10, 10)

        # Decode: 固定 300ms（生成 ~60 tokens）
        decode_time_ms = 300.0 + random.uniform(-20, 20)

        total_time_ms = prefill_time_ms + decode_time_ms

        return {
            "agent_id": f"{agent_type}-agent",
            "timestamp": timestamp.isoformat(),
            "system_prompt_length": system_prompt_length,
            "user_query_length": user_query_length,
            "total_prompt_length": total_prompt_length,
            "cache_hit_ratio": round(cache_hit_ratio, 3),
            "skip_logic_type": skip_logic_type,
            "block_size": block_size,
            "padding_tokens": padding_tokens,
            "padding_overhead": round(padding_overhead, 2),
            "prefill_time_ms": round(prefill_time_ms, 2),
            "decode_time_ms": round(decode_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "config_version": "1.0.0-initial",
        }

    def generate_workload(
        self,
        num_days: int = 7,
        requests_per_day: int = 500,
        output_format: str = "jsonl",
    ) -> str:
        """
        生成 N 天的生产环境负载数据

        Args:
            num_days: 模拟天数
            requests_per_day: 每天平均请求数
            output_format: 输出格式（jsonl 或 csv）

        Returns:
            输出文件路径
        """
        print(f"生成 {num_days} 天的 OpenClaw 负载数据...")
        print(f"  每天平均请求数: {requests_per_day}")
        print(f"  Agent 类型: {len(self.agent_profiles)}")

        records = []
        start_time = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_day = start_time + timedelta(days=day)
            daily_requests = 0

            for hour in range(24):
                current_hour = current_day + timedelta(hours=hour)

                # 根据时间采样该小时的请求数
                # 每小时基础请求数：requests_per_day / 24
                base_hourly_requests = requests_per_day // 24

                for _ in range(base_hourly_requests):
                    # 加权采样 agent
                    agent_type = self._sample_agent_weighted()
                    profile = self.agent_profiles[agent_type]

                    # 根据 agent 的使用模式和当前时间决定是否生成请求
                    traffic_multiplier = self._get_hourly_traffic_multiplier(
                        hour, profile["usage_pattern"]
                    )

                    if random.random() < traffic_multiplier:
                        # 随机时间戳（在当前小时内）
                        minute = random.randint(0, 59)
                        second = random.randint(0, 59)
                        timestamp = current_hour + timedelta(minutes=minute, seconds=second)

                        record = self._generate_inference_record(agent_type, timestamp)
                        records.append(record)
                        daily_requests += 1

            print(f"  Day {day + 1}/{num_days}: {daily_requests} 请求")

        # 排序（按时间）
        records.sort(key=lambda x: x["timestamp"])

        # 写入文件
        if output_format == "jsonl":
            output_file = self.output_dir / f"openclaw-workload-{num_days}d.jsonl"
            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
        else:  # csv
            import csv

            output_file = self.output_dir / f"openclaw-workload-{num_days}d.csv"
            with open(output_file, "w", newline="") as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

        print(f"\n✅ 生成 {len(records)} 条记录")
        print(f"   输出文件: {output_file}")

        # 统计
        agent_counts = {}
        for record in records:
            agent_id = record["agent_id"]
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

        print(f"\n📊 Agent 分布:")
        for agent_id, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
            pct = count / len(records) * 100
            print(f"   {agent_id}: {count} ({pct:.1f}%)")

        return str(output_file)

    def generate_metadata(self, num_days: int, total_requests: int):
        """生成负载元数据"""
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "simulation_duration_days": num_days,
            "total_requests": total_requests,
            "agents": {
                agent_type: {
                    "system_prompt_range": profile["system_prompt_range"],
                    "user_query_range": profile["user_query_range"],
                    "cache_hit_range": profile["cache_hit_range"],
                    "initial_block_size": profile["block_size"],
                    "usage_pattern": profile["usage_pattern"],
                    "request_frequency": profile["request_frequency"],
                }
                for agent_type, profile in self.agent_profiles.items()
            },
            "optimization_opportunities": {
                "researcher": "过度 padding（block_size=128 vs optimal=64）",
                "coder": "轻度 padding（block_size=96 vs optimal=64）",
                "analyst": "配置合理（block_size=64）",
                "pm": "配置合理（block_size=64）",
                "tester": "轻度 padding（block_size=80 vs optimal=64）",
            },
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n   元数据: {metadata_file}")


def main():
    """主函数"""
    generator = OpenClawWorkloadGenerator(output_dir="openclaw-workload")

    # 生成 7 天负载（~3500 请求）
    workload_file = generator.generate_workload(
        num_days=7, requests_per_day=500, output_format="jsonl"
    )

    # 生成元数据
    with open(workload_file, "r") as f:
        total_requests = sum(1 for _ in f)
    generator.generate_metadata(num_days=7, total_requests=total_requests)

    print(f"\n🎯 负载数据已生成，可用于测试 Phase 3 高级策略！")


if __name__ == "__main__":
    main()
