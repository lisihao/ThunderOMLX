#!/usr/bin/env python3
"""
PP 测试 - 避免 cache hit（使用不同的 prompt）
"""
import asyncio
import sys
import time
import gc
import random
import string
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_unique_prompt(target_tokens: int) -> str:
    """生成唯一的 prompt（避免 cache hit）"""
    # 使用随机字符串 + 计数器确保每次都不同
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
        "algorithm", "datastructure", "framework", "library", "module"
    ]

    # 添加随机后缀避免完全重复
    random_suffix = ''.join(random.choices(string.ascii_lowercase, k=10))

    # 生成 prompt
    prompt_parts = []
    words_needed = target_tokens // 4  # 粗略估算：1 word ≈ 4 tokens

    for i in range(words_needed):
        word = random.choice(base_words)
        prompt_parts.append(f"{word}{i}")

    prompt = " ".join(prompt_parts) + f" {random_suffix}"
    return prompt


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 70)
    print("📊 PP 性能测试（无 cache hit）")
    print("=" * 70)
    print("社区基线 (8k): 637-693 tok/s")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print("初始化引擎...")
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()
    print("✅ 引擎启动\n")

    try:
        # 测试不同长度
        test_lengths = [1000, 2000, 4000, 6000, 8000]
        results = []

        for target_tokens in test_lengths:
            print(f"\n{'─' * 70}")
            print(f"测试 ~{target_tokens} tokens...")

            # 生成唯一 prompt（避免 cache hit）
            text = generate_unique_prompt(target_tokens)

            try:
                start = time.perf_counter()
                first_token = None

                async for output in engine.stream_generate(
                    prompt=text,
                    max_tokens=1,
                    temperature=0.0
                ):
                    if output.new_text and first_token is None:
                        first_token = time.perf_counter()
                        break

                if first_token:
                    prefill = first_token - start
                    actual_tokens = len(text.split())  # 粗略估算
                    pp_tps = actual_tokens / prefill

                    print(f"✅ 成功")
                    print(f"   实际 tokens: {actual_tokens} (估算)")
                    print(f"   Prefill 时间: {prefill:.3f}s")
                    print(f"   PP TPS: {pp_tps:.1f} tok/s")

                    results.append({
                        'target': target_tokens,
                        'actual': actual_tokens,
                        'pp_tps': pp_tps,
                        'prefill_time': prefill,
                        'success': True
                    })
                else:
                    print(f"❌ 未生成 token")
                    results.append({
                        'target': target_tokens,
                        'success': False,
                        'error': 'No token generated'
                    })

            except Exception as e:
                print(f"❌ 失败: {e}")
                results.append({
                    'target': target_tokens,
                    'success': False,
                    'error': str(e)
                })

                if 'Metal' in str(e) or 'assertion' in str(e).lower():
                    print("\n⚠️  Metal 错误，停止测试")
                    break

            # 清理
            gc.collect()
            await asyncio.sleep(2)

        # 总结
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)

        successful = [r for r in results if r['success']]

        if successful:
            print("\n成功的测试:")
            print(f"{'Tokens':<10} {'Prefill':<12} {'PP TPS':<15} {'vs 665'}")
            print("─" * 60)

            for r in successful:
                delta = r['pp_tps'] - 665
                status = "✅" if r['pp_tps'] >= 637 else "⚠️"
                print(f"{r['actual']:<10} {r['prefill_time']:<12.3f} "
                      f"{r['pp_tps']:<15.1f} {delta:+.1f} {status}")

            max_successful = max(successful, key=lambda x: x['actual'])
            print(f"\n最大成功测试: {max_successful['actual']} tokens")
            print(f"PP TPS: {max_successful['pp_tps']:.1f} tok/s")
            print(f"Prefill 时间: {max_successful['prefill_time']:.3f}s")
        else:
            print("\n❌ 所有测试均失败")

        return len(successful) > 0

    finally:
        print("\n关闭引擎...")
        await engine.stop()
        gc.collect()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
