#!/usr/bin/env python3
"""
完整的 PP + TG 性能测试

分别测试：
1. PP (Prompt Processing): 处理长 prompt 的速度
2. TG (Token Generation): 生成 token 的速度

社区基线：
- PP: 600-800 tok/s
- TG: 60-80+ tok/s
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def generate_long_prompt(target_tokens: int) -> str:
    """
    生成指定长度的 prompt

    使用重复的技术文本确保达到目标 token 数
    平均每个单词 ~1.3 tokens
    """
    base_text = (
        "In software engineering, design patterns are typical solutions to common problems "
        "in software design. Each pattern is like a blueprint that you can customize to solve "
        "a particular design problem in your code. Design patterns are formalized best practices "
        "that the programmer can use to solve common problems when designing an application or system. "
        "Object-oriented design patterns typically show relationships and interactions between classes "
        "or objects, without specifying the final application classes or objects that are involved. "
        "Patterns that imply mutable state may be unsuited for functional programming languages. "
        "Some patterns can be rendered unnecessary in languages that have built-in support for solving "
        "the problem they are trying to solve, and object-oriented patterns are not necessarily suitable "
        "for non-object-oriented languages. Design patterns may be viewed as a structured approach to "
        "computer programming intermediate between the levels of a programming paradigm and a concrete algorithm. "
    )

    # 每个 base_text 约 130 tokens
    repeats = (target_tokens // 130) + 1
    long_text = (base_text + "\n") * repeats

    return long_text


async def test_pp_performance(engine, target_tokens: int = 8192):
    """
    测试 PP (Prompt Processing) 性能

    Args:
        engine: BatchedEngine 实例
        target_tokens: 目标 prompt 长度

    Returns:
        pp_tps: Prompt Processing TPS
    """
    print("\n" + "=" * 80)
    print("📊 PP (Prompt Processing) 性能测试")
    print("=" * 80)
    print(f"目标 prompt 长度: {target_tokens} tokens")
    print(f"生成长度: 1 token（只测 prefill）")
    print(f"社区基线: 600-800 tok/s")
    print("")

    # 生成长 prompt
    long_prompt = generate_long_prompt(target_tokens)
    print(f"✅ 生成了长 prompt（预估 ~{target_tokens} tokens）")
    print("")

    # 预热（第一次会有 Metal shader 编译开销）
    print("🔥 预热中...")
    async for _ in engine.stream_generate(
        prompt=long_prompt,
        max_tokens=1,
        temperature=0.0
    ):
        pass
    print("✅ 预热完成")
    await asyncio.sleep(2)  # Metal 并发规避

    print("\n开始正式测试...")

    # 正式测试（3 次取平均）
    pp_times = []
    actual_prompt_tokens = None

    for i in range(3):
        print(f"\n  Round {i+1}/3:")

        # 记录开始时间
        start_time = time.perf_counter()

        first_token_time = None
        token_count = 0

        async for output in engine.stream_generate(
            prompt=long_prompt,
            max_tokens=1,
            temperature=0.0
        ):
            if output.new_text and first_token_time is None:
                first_token_time = time.perf_counter()

                # TTFT (Time To First Token) = Prefill time
                prefill_time = first_token_time - start_time
                pp_times.append(prefill_time)

                # 尝试获取实际的 prompt tokens 数
                if hasattr(output, 'prompt_tokens'):
                    actual_prompt_tokens = output.prompt_tokens

                print(f"    TTFT (Prefill): {prefill_time:.3f}s")

            if output.new_text:
                token_count += 1

        if first_token_time is None:
            print(f"    ⚠️  未收到 token，跳过")
            continue

        # 等待清理
        await asyncio.sleep(2)

    # 计算平均值
    avg_prefill_time = sum(pp_times) / len(pp_times)

    # 使用实际 token 数或估计值
    if actual_prompt_tokens:
        prompt_tokens = actual_prompt_tokens
        print(f"\n✅ 实际 prompt tokens: {prompt_tokens}")
    else:
        # 估算：long_prompt 字符数 / 4（GPT 估算）
        prompt_tokens = len(long_prompt) // 4
        print(f"\n⚠️  估算 prompt tokens: {prompt_tokens}")

    pp_tps = prompt_tokens / avg_prefill_time

    print("\n" + "─" * 80)
    print("PP 性能结果:")
    print("─" * 80)
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"平均 Prefill 时间: {avg_prefill_time:.3f}s")
    print(f"PP TPS: {pp_tps:.1f} tok/s")
    print(f"社区基线: 600-800 tok/s")

    if pp_tps >= 600:
        delta = pp_tps - 600
        print(f"✅ 达到基线！({delta:+.1f} tok/s)")
    else:
        delta = pp_tps - 600
        print(f"⚠️  低于基线 ({delta:+.1f} tok/s)")

    return pp_tps


async def test_tg_performance(engine, output_tokens: int = 512):
    """
    测试 TG (Token Generation) 性能

    Args:
        engine: BatchedEngine 实例
        output_tokens: 目标生成长度

    Returns:
        tg_tps: Token Generation TPS
    """
    print("\n" + "=" * 80)
    print("📊 TG (Token Generation) 性能测试")
    print("=" * 80)
    print(f"Prompt: 短文本 (~50 tokens)")
    print(f"生成长度: {output_tokens} tokens")
    print(f"社区基线: 60-80+ tok/s")
    print("")

    short_prompt = (
        "Explain the key differences between Python and JavaScript in detail, "
        "covering syntax, runtime environment, type systems, and common use cases."
    )

    # 预热
    print("🔥 预热中...")
    async for _ in engine.stream_generate(
        prompt=short_prompt,
        max_tokens=32,
        temperature=0.0
    ):
        pass
    print("✅ 预热完成")
    await asyncio.sleep(2)

    print("\n开始正式测试...")

    # 正式测试（3 次取平均）
    tg_times = []
    token_counts = []

    for i in range(3):
        print(f"\n  Round {i+1}/3:")

        # 跳过 prefill，只计时 generation
        first_token_time = None
        last_token_time = None
        token_count = 0

        async for output in engine.stream_generate(
            prompt=short_prompt,
            max_tokens=output_tokens,
            temperature=0.7  # 非确定性，避免 cache
        ):
            if output.new_text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                last_token_time = time.perf_counter()
                token_count += 1

                if token_count % 100 == 0:
                    print(f"    生成中: {token_count} tokens", end="\r")

        if first_token_time and last_token_time:
            generation_time = last_token_time - first_token_time
            tg_times.append(generation_time)
            token_counts.append(token_count)

            print(f"    生成: {token_count} tokens in {generation_time:.3f}s")
            print(f"    TG TPS: {token_count / generation_time:.1f} tok/s")

        # 等待清理
        await asyncio.sleep(2)

    # 计算平均值
    avg_generation_time = sum(tg_times) / len(tg_times)
    avg_token_count = sum(token_counts) / len(token_counts)
    tg_tps = avg_token_count / avg_generation_time

    print("\n" + "─" * 80)
    print("TG 性能结果:")
    print("─" * 80)
    print(f"平均生成 tokens: {avg_token_count:.1f}")
    print(f"平均 Generation 时间: {avg_generation_time:.3f}s")
    print(f"TG TPS: {tg_tps:.1f} tok/s")
    print(f"社区基线: 60-80+ tok/s")

    if tg_tps >= 60:
        delta = tg_tps - 70  # 中间值
        print(f"✅ 达到基线！({delta:+.1f} tok/s from 70)")
    else:
        delta = tg_tps - 60
        print(f"⚠️  低于基线 ({delta:+.1f} tok/s)")

    return tg_tps


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🧪 完整 PP + TG 性能测试")
    print("=" * 80)
    print("")
    print("社区基线:")
    print("  - PP (Prompt Processing): 600-800 tok/s")
    print("  - TG (Token Generation): 60-80+ tok/s")
    print("")
    print("已知限制:")
    print("  - 请求间需等待 2s（Metal 并发规避）")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print("初始化引擎...")
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()
    print("✅ 引擎启动完成")

    try:
        # 测试 1: PP
        pp_tps = await test_pp_performance(engine, target_tokens=8192)

        # 等待清理
        await asyncio.sleep(3)

        # 测试 2: TG
        tg_tps = await test_tg_performance(engine, output_tokens=512)

        # 综合报告
        print("\n" + "=" * 80)
        print("📊 综合性能报告")
        print("=" * 80)
        print("")
        print(f"PP (Prompt Processing):  {pp_tps:7.1f} tok/s  (基线: 600-800)")
        print(f"TG (Token Generation):   {tg_tps:7.1f} tok/s  (基线: 60-80+)")
        print("")

        # 判断是否达标
        pp_pass = pp_tps >= 600
        tg_pass = tg_tps >= 60

        if pp_pass and tg_pass:
            print("🎉 两项指标均达到社区基线！")
            return True
        elif pp_pass or tg_pass:
            print("✅ 部分指标达到基线")
            if not pp_pass:
                print(f"   ⚠️  PP 需要提升: {pp_tps:.1f} → 600+ tok/s")
            if not tg_pass:
                print(f"   ⚠️  TG 需要提升: {tg_tps:.1f} → 60+ tok/s")
            return True
        else:
            print("⚠️  两项指标均低于基线")
            return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n停止引擎...")
        await engine.stop()
        print("✅ 测试完成")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
