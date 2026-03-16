#!/usr/bin/env python3
"""
单引擎实例 PP 测试 - 避免重复创建引擎触发 Metal 错误
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 70)
    print("📊 单引擎 PP 性能测试（渐进式）")
    print("=" * 70)
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print("初始化引擎（只初始化一次）...")
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()
    print("✅ 引擎启动")

    try:
        # 测试不同长度
        test_lengths = [2500, 3000, 4000, 5000, 6000, 7000, 8000]
        results = []

        for target_tokens in test_lengths:
            print(f"\n{'─' * 70}")
            print(f"测试 ~{target_tokens} tokens...")

            # 生成 prompt
            base = "word " * 100
            repeats = target_tokens // 100
            text = base * repeats

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
                    actual_tokens = len(text) // 4
                    pp_tps = actual_tokens / prefill

                    print(f"✅ 成功")
                    print(f"   Tokens: {actual_tokens}")
                    print(f"   Prefill: {prefill:.3f}s")
                    print(f"   PP TPS: {pp_tps:.1f} tok/s")

                    results.append({
                        'tokens': actual_tokens,
                        'pp_tps': pp_tps,
                        'success': True
                    })
                else:
                    print("❌ 未生成 token")
                    results.append({'success': False})

            except Exception as e:
                print(f"❌ 错误: {e}")
                results.append({'success': False})
                # Metal 错误后停止
                break

            # 关键：等待足够长时间让清理完成
            print("   等待清理...")
            await asyncio.sleep(3)

        # 总结
        print("\n" + "=" * 70)
        print("📊 测试结果")
        print("=" * 70)

        successful = [r for r in results if r.get('success')]

        if successful:
            print(f"\n{'Tokens':<10} {'PP TPS':<15} {'vs 675'}")
            print("─" * 40)

            for r in successful:
                delta = r['pp_tps'] - 675
                status = "✅" if delta >= -40 else "⚠️"
                print(f"{r['tokens']:<10} {r['pp_tps']:<15.1f} {delta:+.1f} {status}")

            max_test = max(successful, key=lambda x: x['tokens'])
            print(f"\n最大成功: {max_test['tokens']} tokens → {max_test['pp_tps']:.1f} tok/s")

            if max_test['tokens'] >= 7500:
                print("✅ 已测试接近 8k！")
            elif max_test['tokens'] >= 5000:
                print("⚠️  测试到 5k+，但未达 8k")

        return len(successful) > 0

    except Exception as e:
        print(f"\n❌ 引擎错误: {e}")
        return False
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
