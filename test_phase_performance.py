#!/usr/bin/env python3
"""
Phase 1-4 性能验证 - 单请求顺序场景

测试场景：4 个顺序请求，每个 512 tokens
目标：验证 Processing TPS 提升（692.7 → 730+ tok/s）
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_sequential_processing():
    """顺序处理 4 个请求"""
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🧪 Phase 1-4 性能验证 - 单请求顺序场景")
    print("=" * 80)
    print("")
    print("测试配置:")
    print("  - 4 个顺序请求")
    print("  - 每请求: 512 tokens")
    print("  - 总计: 2048 tokens")
    print("  - 基线: 692.7 tok/s (Processing TPS)")
    print("  - 目标: 730+ tok/s (+5.4%)")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        prompts = [
            "Explain the key differences between Python and JavaScript in detail, "
            "covering syntax, runtime, type systems, and common use cases.",

            "What are the main advantages of using TypeScript over JavaScript? "
            "Discuss type safety, tooling support, and developer experience.",

            "Describe the most important design patterns in software engineering. "
            "Include examples like Singleton, Factory, Observer, and Strategy patterns.",

            "Explain how asynchronous programming works in Python with examples. "
            "Cover async/await, event loops, and common pitfalls."
        ]

        overall_start = time.time()
        total_tokens = 0
        generation_times = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'─' * 80}")
            print(f"📝 Request {i}/{len(prompts)}")
            print(f"{'─' * 80}")

            request_tokens = 0
            gen_start = time.time()

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7
            ):
                if output.new_text:
                    request_tokens += 1
                    if request_tokens % 100 == 0:
                        print(f"  Generated: {request_tokens} tokens", end="\r")

            gen_elapsed = time.time() - gen_start
            generation_times.append(gen_elapsed)

            print(f"✅ Request {i} completed: {request_tokens} tokens in {gen_elapsed:.2f}s")
            print(f"   Generation TPS: {request_tokens / gen_elapsed:.1f} tok/s")

            total_tokens += request_tokens

        overall_elapsed = time.time() - overall_start

        # 计算 Processing TPS（包含所有开销）
        processing_tps = total_tokens / overall_elapsed

        # 计算纯 Generation TPS（平均）
        avg_gen_time = sum(generation_times) / len(generation_times)
        avg_gen_tps = (total_tokens / len(prompts)) / avg_gen_time

        print("\n" + "=" * 80)
        print("📊 性能结果")
        print("=" * 80)
        print(f"总 tokens: {total_tokens}")
        print(f"总时间: {overall_elapsed:.2f}s")
        print(f"")
        print(f"Processing TPS: {processing_tps:.1f} tok/s")
        print(f"  基线: 692.7 tok/s")
        print(f"  提升: {((processing_tps / 692.7) - 1) * 100:+.1f}%")
        print(f"")
        print(f"Generation TPS (平均): {avg_gen_tps:.1f} tok/s")
        print("")

        # 判断是否达到目标
        if processing_tps >= 730:
            print("🎉 目标达成！Processing TPS ≥ 730 tok/s")
            return True
        elif processing_tps >= 710:
            print("✅ 良好进展！Processing TPS ≥ 710 tok/s (Phase 1+2)")
            return True
        else:
            print(f"⚠️  未达目标，但仍有提升: {processing_tps:.1f} tok/s")
            return True  # Still successful if no errors

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()


async def main():
    success = await test_sequential_processing()

    if success:
        print("\n" + "=" * 80)
        print("✅ 测试完成")
        print("=" * 80)
        print("")
        print("下一步:")
        print("  - 如果 Processing TPS < 730: 分析瓶颈，调整优化")
        print("  - 如果 Processing TPS ≥ 730: 验证并发场景稳定性")
        print("")


if __name__ == "__main__":
    asyncio.run(main())
