#!/usr/bin/env python3
"""
PP 测试 - 临时禁用缓存，找性能瓶颈
"""
import asyncio
import sys
import time
import gc
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_unique_prompt(target_tokens: int, tokenizer) -> tuple[str, int]:
    """生成唯一的 prompt 并返回实际 token 数"""
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
        "algorithm", "datastructure", "framework", "library", "module",
        "performance", "scalability", "reliability", "security", "efficiency"
    ]

    # 生成足够长的文本
    prompt_parts = []
    for i in range(target_tokens * 2):
        word = random.choice(base_words)
        prompt_parts.append(f"{word}{i}")

    full_text = " ".join(prompt_parts)

    # 使用 tokenizer 精确计算
    tokens = tokenizer.encode(full_text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]

    final_text = tokenizer.decode(tokens)
    actual_tokens = len(tokens)

    return final_text, actual_tokens


async def main():
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    print("=" * 70)
    print("📊 PP 性能测试（缓存禁用）")
    print("=" * 70)
    print("目的：对比缓存开启 vs 关闭，找性能瓶颈")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        fix_mistral_regex=True  # 修复 Qwen tokenizer regex 问题
    )
    print("✅ Tokenizer 加载完成\n")

    print("初始化引擎（缓存禁用）...")
    # 关键：通过 SchedulerConfig 禁用缓存
    scheduler_config = SchedulerConfig(
        paged_ssd_cache_dir=None,  # 禁用 SSD 缓存
        hot_cache_max_size=0        # 禁用 hot cache
    )
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )
    await engine.start()
    print("✅ 引擎启动（缓存已禁用）\n")

    try:
        # 测试不同长度
        test_lengths = [1000, 2000, 4000, 6000, 8000]
        results = []

        for target_tokens in test_lengths:
            print(f"\n{'─' * 70}")
            print(f"测试 {target_tokens} tokens...")

            # 生成精确长度的 prompt
            text, actual_tokens = generate_unique_prompt(target_tokens, tokenizer)
            print(f"生成 prompt: {actual_tokens} tokens")

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
                    pp_tps = actual_tokens / prefill

                    print(f"✅ 成功")
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
        print("📊 测试总结（缓存禁用）")
        print("=" * 70)

        successful = [r for r in results if r['success']]

        if successful:
            print("\n成功的测试:")
            print(f"{'Tokens':<10} {'Prefill':<12} {'PP TPS':<15} {'vs 缓存开启':<15} {'状态'}")
            print("─" * 70)

            # 对比基线（缓存开启时的性能）
            baseline = {
                1000: 519.2,
                2000: 699.1,
                4000: 688.9,
                6000: 683.2,
                8000: 686.3
            }

            for r in successful:
                baseline_tps = baseline.get(r['target'], 655.4)
                delta = r['pp_tps'] - baseline_tps
                status = "🚀" if delta > 0 else "⚠️"
                print(f"{r['actual']:<10} {r['prefill_time']:<12.3f} "
                      f"{r['pp_tps']:<15.1f} {delta:+12.1f}   {status}")

            max_successful = max(successful, key=lambda x: x['actual'])
            print(f"\n最大成功测试: {max_successful['actual']} tokens")
            print(f"PP TPS: {max_successful['pp_tps']:.1f} tok/s")
            print(f"Prefill 时间: {max_successful['prefill_time']:.3f}s")

            # 性能对比
            avg_tps = sum(r['pp_tps'] for r in successful) / len(successful)
            baseline_avg = 655.4
            print(f"\n平均 PP TPS（缓存禁用）: {avg_tps:.1f} tok/s")
            print(f"平均 PP TPS（缓存开启）: {baseline_avg:.1f} tok/s")
            print(f"差异: {avg_tps - baseline_avg:.1f} tok/s ({(avg_tps/baseline_avg-1)*100:+.1f}%)")

            print("\n🔍 性能瓶颈分析:")
            if avg_tps > baseline_avg + 50:
                print(f"   ✅ 缓存写入是主要瓶颈（禁用后提升 {(avg_tps/baseline_avg-1)*100:.1f}%）")
            elif avg_tps > baseline_avg + 10:
                print(f"   ⚠️  缓存写入有影响，但不是主要瓶颈（提升 {(avg_tps/baseline_avg-1)*100:.1f}%）")
            else:
                print(f"   ❌ 缓存写入不是瓶颈（差异 {(avg_tps/baseline_avg-1)*100:.1f}%）")
                print("   💡 瓶颈在模型推理本身（mx.eval）")
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
