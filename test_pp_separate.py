#!/usr/bin/env python3
"""
PP (Prompt Processing) 性能测试 - 独立版本

测试 prefill 性能，避免 Metal 并发问题
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_long_prompt(target_tokens: int) -> str:
    """生成指定长度的 prompt"""
    base_text = (
        "In software engineering, design patterns are typical solutions to common problems "
        "in software design. Each pattern is like a blueprint that you can customize to solve "
        "a particular design problem in your code. Design patterns are formalized best practices "
        "that the programmer can use to solve common problems when designing an application or system. "
        "Object-oriented design patterns typically show relationships and interactions between classes "
        "or objects, without specifying the final application classes or objects that are involved. "
    )
    repeats = (target_tokens // 100) + 1
    return (base_text + "\n") * repeats


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 70)
    print("📊 PP (Prompt Processing) 性能测试")
    print("=" * 70)
    print("目标: 8192 tokens prompt")
    print("生成: 1 token（只测 prefill）")
    print("社区基线: 600-800 tok/s")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    long_prompt = generate_long_prompt(8192)

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
            max_tokens=1,
            temperature=0.0
        ):
            pass
        print("✅ 预热完成")

        # 正式测试
        print("\n开始正式测试（3 轮）...")
        pp_times = []

        for i in range(3):
            print(f"\n  Round {i+1}/3:")

            start_time = time.perf_counter()
            first_token_time = None

            async for output in engine.stream_generate(
                prompt=long_prompt,
                max_tokens=1,
                temperature=0.0
            ):
                if output.new_text and first_token_time is None:
                    first_token_time = time.perf_counter()
                    prefill_time = first_token_time - start_time
                    pp_times.append(prefill_time)
                    print(f"    Prefill 时间: {prefill_time:.3f}s")

        # 计算结果
        avg_prefill = sum(pp_times) / len(pp_times)
        prompt_tokens = len(long_prompt) // 4  # 估算
        pp_tps = prompt_tokens / avg_prefill

        print("\n" + "=" * 70)
        print("📊 PP 性能结果")
        print("=" * 70)
        print(f"Prompt tokens (估算): {prompt_tokens}")
        print(f"平均 Prefill 时间: {avg_prefill:.3f}s")
        print(f"PP TPS: {pp_tps:.1f} tok/s")
        print(f"社区基线: 600-800 tok/s")

        if pp_tps >= 600:
            print(f"✅ 达到基线！(+{pp_tps - 600:.1f} tok/s)")
        else:
            print(f"⚠️  低于基线 ({pp_tps - 600:.1f} tok/s)")

        return pp_tps >= 600

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
