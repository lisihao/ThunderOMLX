#!/usr/bin/env python3
"""
Phase 1-4 完整性能验证

测试场景：4 个顺序请求，每个 512 tokens
目标：验证 Processing TPS 从 692.7 → 730+ tok/s (+5.4%)

已知限制：
- 请求间需要等待 2s（Metal 并发问题的临时规避）
- 并发度限制为 1（顺序处理）
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🧪 Phase 1-4 完整性能验证")
    print("=" * 80)
    print("")
    print("测试配置:")
    print("  - 4 个顺序请求")
    print("  - 每请求: 512 tokens")
    print("  - 总计: 2048 tokens")
    print("  - 请求间等待: 2s（Metal 并发规避）")
    print("")
    print("性能目标:")
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
            "covering syntax, runtime environment, type systems, and common use cases. "
            "Discuss how each language handles asynchronous programming.",

            "What are the main advantages of using TypeScript over JavaScript? "
            "Discuss type safety, tooling support, developer experience, and potential drawbacks. "
            "Provide examples of when TypeScript is most beneficial.",

            "Describe the most important design patterns in software engineering. "
            "Include detailed examples like Singleton, Factory, Observer, Strategy, and Decorator patterns. "
            "Explain when to use each pattern.",

            "Explain how asynchronous programming works in Python with comprehensive examples. "
            "Cover async/await, event loops, asyncio, common pitfalls, and best practices. "
            "Compare with threading and multiprocessing."
        ]

        overall_start = time.time()
        total_tokens = 0
        generation_times = []
        wait_time_total = 0

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
                        print(f"  生成中: {request_tokens} tokens", end="\r")

            gen_elapsed = time.time() - gen_start
            generation_times.append(gen_elapsed)

            print(f"✅ Request {i} 完成: {request_tokens} tokens in {gen_elapsed:.2f}s")
            print(f"   Generation TPS: {request_tokens / gen_elapsed:.1f} tok/s")

            total_tokens += request_tokens

            # Metal 并发规避：请求间等待
            if i < len(prompts):
                print("   等待清理完成（2s）...")
                await asyncio.sleep(2)
                wait_time_total += 2

        overall_elapsed = time.time() - overall_start

        # 计算 Processing TPS（包含所有开销，含等待时间）
        processing_tps_with_wait = total_tokens / overall_elapsed

        # 计算实际处理时间的 Processing TPS（不含等待）
        actual_processing_time = overall_elapsed - wait_time_total
        processing_tps_actual = total_tokens / actual_processing_time

        # 计算平均 Generation TPS
        avg_gen_time = sum(generation_times) / len(generation_times)
        avg_gen_tps = (total_tokens / len(prompts)) / avg_gen_time

        print("\n" + "=" * 80)
        print("📊 性能结果")
        print("=" * 80)
        print(f"总 tokens: {total_tokens}")
        print(f"总时间（含等待）: {overall_elapsed:.2f}s")
        print(f"实际处理时间: {actual_processing_time:.2f}s")
        print(f"等待时间: {wait_time_total:.2f}s")
        print("")
        print(f"Processing TPS（实际）: {processing_tps_actual:.1f} tok/s")
        print(f"  基线: 692.7 tok/s")
        print(f"  提升: {((processing_tps_actual / 692.7) - 1) * 100:+.1f}%")
        print("")
        print(f"Processing TPS（含等待）: {processing_tps_with_wait:.1f} tok/s")
        print(f"Generation TPS（平均）: {avg_gen_tps:.1f} tok/s")
        print("")

        # 判断是否达到目标
        if processing_tps_actual >= 730:
            print("🎉 目标达成！Processing TPS ≥ 730 tok/s")
            print("   Phase 1-4 优化生效！")
            success = True
        elif processing_tps_actual >= 710:
            print("✅ 良好进展！Processing TPS ≥ 710 tok/s")
            print("   已达到 Phase 1+2 目标")
            success = True
        elif processing_tps_actual > 692.7:
            print(f"✓  有提升: {processing_tps_actual:.1f} tok/s (基线 692.7)")
            print("   Phase 1-4 部分生效")
            success = True
        else:
            print(f"⚠️  性能未提升: {processing_tps_actual:.1f} tok/s")
            success = False

        print("")
        print("=" * 80)
        print("📝 Phase 测试总结")
        print("=" * 80)
        for phase, desc in [
            ("Phase 1", "异步 Tensor 提取 (+2.8% 预期)"),
            ("Phase 2", "异步 save_block (+2.1% 预期)"),
            ("Phase 3", "减少调度间隙 (+1.0% 预期)"),
            ("Phase 4", "批量 Metal 操作 (+0.3% 预期)"),
        ]:
            print(f"  {phase}: {desc}")
        print("")

        return success

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()


if __name__ == "__main__":
    success = asyncio.run(main())

    print("\n" + "=" * 80)
    print("🏁 测试完成")
    print("=" * 80)

    if success:
        print("")
        print("下一步:")
        print("  - 如果 Processing TPS < 730: 分析 Phase 瓶颈，进一步优化")
        print("  - 如果 Processing TPS ≥ 730: 验证并发场景稳定性（已知 Metal 限制）")
        print("  - 文档化 Metal 并发限制和使用建议")
        print("")

    sys.exit(0 if success else 1)
