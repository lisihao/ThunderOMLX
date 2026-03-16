#!/usr/bin/env python3
"""
TG (Token Generation) 性能测试 - 独立版本

测试 generation 性能，避免 Metal 并发问题
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 70)
    print("📊 TG (Token Generation) 性能测试")
    print("=" * 70)
    print("Prompt: 短文本 (~50 tokens)")
    print("生成: 512 tokens")
    print("社区基线: 60-80+ tok/s")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    short_prompt = (
        "Explain the key differences between Python and JavaScript in detail, "
        "covering syntax, runtime environment, type systems, and common use cases."
    )

    print("初始化引擎...")
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()
    print("✅ 引擎启动")

    try:
        # 预热
        print("\n🔥 预热...")
        async for _ in engine.stream_generate(
            prompt="Hello",
            max_tokens=10,
            temperature=0.0
        ):
            pass
        print("✅ 预热完成")

        # 正式测试
        print("\n开始正式测试（3 轮）...")
        tg_times = []
        token_counts = []

        for i in range(3):
            print(f"\n  Round {i+1}/3:")

            first_token_time = None
            last_token_time = None
            token_count = 0

            async for output in engine.stream_generate(
                prompt=short_prompt,
                max_tokens=512,
                temperature=0.7
            ):
                if output.new_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    last_token_time = time.perf_counter()
                    token_count += 1

                    if token_count % 100 == 0:
                        print(f"    生成: {token_count} tokens", end="\r")

            if first_token_time and last_token_time:
                gen_time = last_token_time - first_token_time
                tg_times.append(gen_time)
                token_counts.append(token_count)
                print(f"    完成: {token_count} tokens in {gen_time:.3f}s")
                print(f"    TPS: {token_count / gen_time:.1f} tok/s")

        # 计算结果
        avg_gen_time = sum(tg_times) / len(tg_times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tg_tps = avg_tokens / avg_gen_time

        print("\n" + "=" * 70)
        print("📊 TG 性能结果")
        print("=" * 70)
        print(f"平均生成 tokens: {avg_tokens:.1f}")
        print(f"平均 Generation 时间: {avg_gen_time:.3f}s")
        print(f"TG TPS: {tg_tps:.1f} tok/s")
        print(f"社区基线: 60-80+ tok/s")

        if tg_tps >= 60:
            print(f"✅ 达到基线！(+{tg_tps - 70:.1f} tok/s from 70)")
        else:
            print(f"⚠️  低于基线 ({tg_tps - 60:.1f} tok/s)")

        return tg_tps >= 60

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
