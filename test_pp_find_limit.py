#!/usr/bin/env python3
"""
找到最大稳定 PP 长度（避免 Metal 错误）
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_single_length(target_tokens: int):
    """测试单个长度的 PP 性能（新引擎）"""
    from omlx.engine.batched import BatchedEngine

    # 生成 prompt
    base = "word " * 100
    repeats = target_tokens // 100
    text = base * repeats

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)

    try:
        await engine.start()

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
            actual_tokens = len(text) // 4
            pp_tps = actual_tokens / prefill

            return {
                'success': True,
                'tokens': actual_tokens,
                'prefill_time': prefill,
                'pp_tps': pp_tps
            }
        else:
            return {'success': False, 'error': 'No token generated'}

    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        await engine.stop()


async def main():
    print("=" * 70)
    print("📊 寻找最大稳定 PP 长度")
    print("=" * 70)
    print("社区基线 (8k): 637-693 tok/s")
    print("")

    # 渐进测试不同长度
    test_lengths = [1000, 2000, 3000, 4000, 5000, 6000]
    results = []

    for target_tokens in test_lengths:
        print(f"\n{'─' * 70}")
        print(f"测试 ~{target_tokens} tokens...")

        result = await test_single_length(target_tokens)

        if result['success']:
            tokens = result['tokens']
            prefill = result['prefill_time']
            pp_tps = result['pp_tps']

            print(f"✅ 成功")
            print(f"   实际 tokens: {tokens}")
            print(f"   Prefill 时间: {prefill:.3f}s")
            print(f"   PP TPS: {pp_tps:.1f} tok/s")

            results.append({
                'target': target_tokens,
                'actual': tokens,
                'pp_tps': pp_tps,
                'success': True
            })
        else:
            print(f"❌ 失败: {result.get('error', 'Unknown error')}")
            results.append({
                'target': target_tokens,
                'success': False
            })
            # Metal 错误通常无法恢复，停止测试
            if 'Metal' in result.get('error', '') or 'assertion' in result.get('error', '').lower():
                print("\n⚠️  检测到 Metal 错误，停止测试")
                break

        # 等待清理
        await asyncio.sleep(3)

    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)

    successful = [r for r in results if r['success']]

    if successful:
        print("\n成功的测试:")
        print(f"{'Tokens':<10} {'PP TPS':<15} {'vs 社区平均 (665)'}")
        print("─" * 50)

        for r in successful:
            delta = r['pp_tps'] - 665
            status = "✅" if r['pp_tps'] >= 637 else "⚠️"
            print(f"{r['actual']:<10} {r['pp_tps']:<15.1f} {delta:+.1f} {status}")

        # 找最接近 8k 的成功测试
        max_successful = max(successful, key=lambda x: x['actual'])
        print(f"\n最大成功测试: {max_successful['actual']} tokens")
        print(f"PP TPS: {max_successful['pp_tps']:.1f} tok/s")

        if max_successful['actual'] >= 7000:
            print(f"✅ 已测试接近 8k context")
        elif max_successful['actual'] >= 5000:
            print(f"⚠️  最大测试 {max_successful['actual']} tokens，未达 8k")
        else:
            print(f"⚠️  最大测试 {max_successful['actual']} tokens，远低于 8k")
    else:
        print("\n❌ 所有测试均失败")

    return len(successful) > 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
