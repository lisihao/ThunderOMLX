"""
测试 ThunderOMLX 的 Prefix Cache 效果

对比：
- 场景 A: 无 System Prompt (短 prompt，低命中率)
- 场景 B: 有 System Prompt (长共享 prefix，高命中率)
"""

import requests
import time
import json
from typing import Dict, List


class ThunderOMLXTester:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url

    def check_server(self) -> bool:
        """检查服务器是否运行"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        response = requests.get(f"{self.server_url}/api/status")
        response.raise_for_status()
        return response.json()

    def send_request(self, prompt: str, max_tokens: int = 50) -> Dict:
        """发送 completion 请求"""
        payload = {
            "model": "Qwen3.5-35B-A3B-6bit",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"   ⚠️  请求失败: {str(e)[:100]}")
            return {}

    def run_scenario_a(self, num_requests: int = 10) -> Dict:
        """场景 A: 无 System Prompt（短 prompt）"""
        print(f"\n📝 场景 A: 发送 {num_requests} 个短 prompt (无 System Prompt)...")

        prompts = [
            "What is AI?",
            "How to code?",
            "Explain Python",
            "What is love?",
            "Tell me a joke",
            "What is 1+1?",
            "How are you?",
            "What is water?",
            "Explain gravity",
            "What is time?",
        ]

        stats_before = self.get_stats()
        start_time = time.time()

        success_count = 0
        for i, prompt in enumerate(prompts[:num_requests]):
            print(f"   [{i+1}/{num_requests}] '{prompt}'", end=" ")
            result = self.send_request(prompt)
            if result:
                print("✅")
                success_count += 1
            else:
                print("❌")

        end_time = time.time()
        stats_after = self.get_stats()

        return {
            "success_count": success_count,
            "total_time": end_time - start_time,
            "stats_before": stats_before,
            "stats_after": stats_after
        }

    def run_scenario_b(self, num_requests: int = 10) -> Dict:
        """场景 B: 有 System Prompt（长共享 prefix）"""
        print(f"\n📝 场景 B: 发送 {num_requests} 个带 System Prompt 的请求...")

        system_prompt = """You are a helpful AI assistant specialized in technical explanations. Your responses should be:
- Concise and to the point
- Technically accurate
- Easy to understand
- Well-structured
- Educational

Always provide clear examples when explaining concepts."""

        user_queries = [
            "What is AI?",
            "How to code?",
            "Explain Python",
            "What is love?",
            "Tell me a joke",
            "What is 1+1?",
            "How are you?",
            "What is water?",
            "Explain gravity",
            "What is time?",
        ]

        print(f"   System Prompt: {len(system_prompt.split())} 词 (约 {len(system_prompt.split()) * 1.3:.0f} tokens)")

        stats_before = self.get_stats()
        start_time = time.time()

        success_count = 0
        for i, query in enumerate(user_queries[:num_requests]):
            full_prompt = f"{system_prompt}\n\nUser: {query}\n\nAssistant:"
            print(f"   [{i+1}/{num_requests}] 'System + {query}'", end=" ")
            result = self.send_request(full_prompt)
            if result:
                print("✅")
                success_count += 1
            else:
                print("❌")

        end_time = time.time()
        stats_after = self.get_stats()

        return {
            "success_count": success_count,
            "total_time": end_time - start_time,
            "stats_before": stats_before,
            "stats_after": stats_after
        }

    def print_comparison(self, scenario_a: Dict, scenario_b: Dict):
        """打印对比结果"""
        print("\n" + "="*70)
        print("📊 场景对比：无 System Prompt vs 有 System Prompt")
        print("="*70)

        # 提取统计信息
        a_before = scenario_a["stats_before"]
        a_after = scenario_a["stats_after"]
        b_before = scenario_b["stats_before"]
        b_after = scenario_b["stats_after"]

        # 计算差值
        a_cached = a_after["total_cached_tokens"] - a_before["total_cached_tokens"]
        b_cached = b_after["total_cached_tokens"] - b_before["total_cached_tokens"]

        a_prompt = a_after["total_prompt_tokens"] - a_before["total_prompt_tokens"]
        b_prompt = b_after["total_prompt_tokens"] - b_before["total_prompt_tokens"]

        # Cache 效率
        a_efficiency = a_after["cache_efficiency"]
        b_efficiency = b_after["cache_efficiency"]

        print(f"\n1️⃣ Prompt Tokens:")
        print(f"   场景 A (无 System Prompt): {a_prompt:,} tokens")
        print(f"   场景 B (有 System Prompt): {b_prompt:,} tokens")
        print(f"   差异: {b_prompt / a_prompt if a_prompt > 0 else 0:.1f}x")

        print(f"\n2️⃣ Cached Tokens:")
        print(f"   场景 A: {a_cached:,} tokens")
        print(f"   场景 B: {b_cached:,} tokens")
        print(f"   增加: +{b_cached - a_cached:,} tokens")

        print(f"\n3️⃣ Cache Efficiency:")
        print(f"   场景 A: {a_efficiency:.1%}")
        print(f"   场景 B: {b_efficiency:.1%}")
        print(f"   提升: +{(b_efficiency - a_efficiency)*100:.1f}%")

        print(f"\n4️⃣ 总耗时:")
        print(f"   场景 A: {scenario_a['total_time']:.2f}s")
        print(f"   场景 B: {scenario_b['total_time']:.2f}s")

        print("\n" + "="*70)


def main():
    print("="*70)
    print("ThunderOMLX Prefix Cache 效果测试")
    print("="*70)

    tester = ThunderOMLXTester()

    # 检查服务器
    print("\n🔍 检查 omlx.server 状态...")
    if not tester.check_server():
        print("❌ omlx.server 未运行 (http://localhost:8000)")
        print("请先启动: python3 -m omlx.server --port 8000")
        return
    print("✅ omlx.server 正在运行")

    # 场景 A
    print("\n" + "="*70)
    print("场景 A: 无 System Prompt（基线）")
    print("="*70)
    scenario_a = tester.run_scenario_a(num_requests=10)

    print("\n⏳ 等待 5 秒...")
    time.sleep(5)

    # 场景 B
    print("\n" + "="*70)
    print("场景 B: 有 System Prompt（优化）")
    print("="*70)
    scenario_b = tester.run_scenario_b(num_requests=10)

    # 对比
    tester.print_comparison(scenario_a, scenario_b)

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
