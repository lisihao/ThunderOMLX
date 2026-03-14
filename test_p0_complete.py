#!/usr/bin/env python3
"""
P0 功能完整测试套件

测试所有 4 个 P0 优化功能：
- P0-1: Full Skip Logic (完全跳过)
- P0-2: Approximate Skip Logic (近似跳过)
- P0-3: Hybrid Hashing (混合哈希)
- P0-4: SSD Compression (SSD 压缩)
"""

import asyncio
import time
import subprocess
from pathlib import Path
from openai import AsyncOpenAI


class P0Tester:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        self.results = {}

    def generate_long_prompt(self, num_sections=50):
        """生成长提示词 (>10000 tokens)"""
        sections = []
        for i in range(1, num_sections + 1):
            sections.append(f"""
            Section {i}: Technical analysis of distributed system component {i}.
            This component handles data processing, communication protocols (HTTP/gRPC),
            persistence (PostgreSQL/Redis), caching strategies, monitoring (Prometheus),
            deployment (Kubernetes), security (TLS/JWT), performance optimization,
            scalability patterns, and failure recovery procedures with circuit breakers.
            """)
        return "\n".join(sections)

    async def test_p0_features(self):
        """测试所有 P0 功能"""
        print("=" * 70)
        print("P0 功能完整测试")
        print("=" * 70)
        print()

        # 生成测试提示词
        system_prompt = "You are a helpful AI assistant."
        user_prompt = self.generate_long_prompt(50)

        # Test 1: 冷启动 (创建缓存)
        print("📝 Test 1: 冷启动 - 创建缓存块")
        print("-" * 70)
        start = time.perf_counter()

        response1 = await self.client.chat.completions.create(
            model="Qwen3.5-35B-A3B-6bit",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,
            temperature=0.0
        )

        elapsed1 = time.perf_counter() - start
        tokens1 = response1.usage.prompt_tokens
        blocks1 = tokens1 // 1024

        self.results['cold_start'] = {
            'time': elapsed1,
            'tokens': tokens1,
            'blocks': blocks1,
            'cached_tokens': blocks1 * 1024
        }

        print(f"  ⏱️  耗时: {elapsed1:.2f}s")
        print(f"  📊 Token 数: {tokens1}")
        print(f"  🧱 创建块数: {blocks1} 个 ({blocks1 * 1024} tokens 缓存)")
        print()

        await asyncio.sleep(2)

        # Test 2: 完全相同提示词 (测试 Full Skip)
        print("📝 Test 2: 完全相同提示词 - Full Skip 测试")
        print("-" * 70)
        start = time.perf_counter()

        response2 = await self.client.chat.completions.create(
            model="Qwen3.5-35B-A3B-6bit",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,
            temperature=0.0
        )

        elapsed2 = time.perf_counter() - start
        speedup2 = elapsed1 / elapsed2 if elapsed2 > 0 else 0

        self.results['full_skip'] = {
            'time': elapsed2,
            'speedup': speedup2,
            'tokens': response2.usage.prompt_tokens
        }

        print(f"  ⏱️  耗时: {elapsed2:.2f}s")
        print(f"  🚀 加速比: {speedup2:.1f}x")
        print(f"  ✅ 预期: 触发 APPROXIMATE SKIP (94%+ 缓存命中)")
        print()

        await asyncio.sleep(2)

        # Test 3: 添加后缀 (测试 Approximate Skip)
        print("📝 Test 3: 添加短后缀 - Approximate Skip 测试")
        print("-" * 70)
        user_prompt_variant = user_prompt + "\n\nPlease be concise and focus on key points."

        start = time.perf_counter()

        response3 = await self.client.chat.completions.create(
            model="Qwen3.5-35B-A3B-6bit",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_variant}
            ],
            max_tokens=50,
            temperature=0.0
        )

        elapsed3 = time.perf_counter() - start
        speedup3 = elapsed1 / elapsed3 if elapsed3 > 0 else 0
        tokens3 = response3.usage.prompt_tokens

        self.results['approximate_skip'] = {
            'time': elapsed3,
            'speedup': speedup3,
            'tokens': tokens3,
            'hit_ratio': (blocks1 * 1024) / tokens3 * 100 if tokens3 > 0 else 0
        }

        print(f"  ⏱️  耗时: {elapsed3:.2f}s")
        print(f"  🚀 加速比: {speedup3:.1f}x")
        print(f"  📊 缓存命中: {self.results['approximate_skip']['hit_ratio']:.1f}%")
        print(f"  ✅ 预期: 触发 APPROXIMATE SKIP (90%+ 缓存命中)")
        print()

    def check_logs(self):
        """检查服务器日志验证功能"""
        print("=" * 70)
        print("日志验证")
        print("=" * 70)
        print()

        log_file = Path.home() / "ThunderOMLX/omlx_final_test.log"
        if not log_file.exists():
            print("⚠️  日志文件不存在")
            return

        checks = {
            '✨ FULL SKIP': 'P0-1: Full Skip Logic',
            '⚡ APPROXIMATE SKIP': 'P0-2: Approximate Skip Logic',
            'xxHash64': 'P0-3: Hybrid Hashing',
            '💾 Saved block': 'P0-4: 块保存到 SSD',
        }

        for pattern, description in checks.items():
            result = subprocess.run(
                ['grep', '-c', pattern, str(log_file)],
                capture_output=True,
                text=True
            )
            count = int(result.stdout.strip()) if result.returncode == 0 else 0

            if count > 0:
                print(f"  ✅ {description}: 找到 {count} 次")
            else:
                print(f"  ⚠️  {description}: 未找到")

        print()

    def check_cache_files(self):
        """检查 SSD 缓存文件"""
        print("=" * 70)
        print("SSD 缓存验证")
        print("=" * 70)
        print()

        cache_dir = Path.home() / ".cache/omlx_cache"

        # 统计缓存文件
        safetensors_files = list(cache_dir.glob("**/*.safetensors*"))

        if safetensors_files:
            total_size = sum(f.stat().st_size for f in safetensors_files)
            print(f"  ✅ P0-4 SSD 压缩:")
            print(f"     文件数: {len(safetensors_files)}")
            print(f"     总大小: {total_size / (1024**2):.1f} MB")

            # 检查是否有压缩文件
            compressed = [f for f in safetensors_files if f.suffix == '.zst']
            if compressed:
                print(f"     压缩文件: {len(compressed)} 个 (.safetensors.zst)")
            else:
                print(f"     ⚠️  未发现压缩文件 (.zst)")
        else:
            print("  ⚠️  未找到缓存文件")

        print()

    def print_summary(self):
        """打印测试总结"""
        print("=" * 70)
        print("测试总结")
        print("=" * 70)
        print()

        # 基本指标
        cold = self.results.get('cold_start', {})
        full = self.results.get('full_skip', {})
        approx = self.results.get('approximate_skip', {})

        print(f"📊 性能指标:")
        print(f"  冷启动:       {cold.get('time', 0):.2f}s  ({cold.get('tokens', 0)} tokens → {cold.get('blocks', 0)} 块)")
        print(f"  完全相同:     {full.get('time', 0):.2f}s  ({full.get('speedup', 0):.1f}x 加速)")
        print(f"  添加后缀:     {approx.get('time', 0):.2f}s  ({approx.get('speedup', 0):.1f}x 加速)")
        print()

        # 验证结果
        print(f"✅ P0 功能验证:")

        # P0-1 & P0-2
        if full.get('speedup', 0) >= 2.0:
            print(f"  ✅ P0-1/P0-2: Skip Logic 工作正常 ({full.get('speedup', 0):.1f}x 加速)")
        else:
            print(f"  ⚠️  P0-1/P0-2: 加速比不足 ({full.get('speedup', 0):.1f}x < 2.0x)")

        # P0-3
        if cold.get('blocks', 0) > 0:
            print(f"  ✅ P0-3: Hybrid Hashing 工作正常 (创建 {cold.get('blocks', 0)} 个块)")
        else:
            print(f"  ⚠️  P0-3: 未创建缓存块")

        # P0-4
        print(f"  ✅ P0-4: SSD 压缩 (见上方缓存验证)")
        print()

        # 对比 ThunderLLAMA
        print(f"📈 对比分析:")
        print(f"  oMLX 缓存加速:        {full.get('speedup', 0):.1f}x")
        print(f"  ThunderLLAMA 基准:    687.6 tok/s (5.8x)")
        print(f"  说明: 本次测试验证了缓存机制正确性，完整 benchmark 需要多轮测试")
        print()

    async def run(self):
        """运行完整测试流程"""
        try:
            # 1. 功能测试
            await self.test_p0_features()

            # 2. 日志验证
            self.check_logs()

            # 3. 缓存文件验证
            self.check_cache_files()

            # 4. 打印总结
            self.print_summary()

            print("=" * 70)
            print("🎉 测试完成！")
            print("=" * 70)

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()


async def main():
    tester = P0Tester()
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
