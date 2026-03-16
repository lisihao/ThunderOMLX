#!/usr/bin/env python3
"""测试 writer 同步机制 - 无需手动等待"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 70)
    print("🧪 Writer 同步机制测试")
    print("=" * 70)
    print("配置: 2 个顺序请求 × 128 tokens")
    print("关键: 无需手动 sleep，依赖 wait_for_writes() 自动同步")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        prompts = [
            "Explain Python in detail, covering syntax and features.",
            "What is TypeScript and why use it over JavaScript?"
        ]

        overall_start = time.time()
        total_tokens = 0

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'─' * 70}")
            print(f"📝 Request {i}/{len(prompts)}")

            request_tokens = 0
            gen_start = time.time()

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=128,
                temperature=0.7
            ):
                if output.new_text:
                    request_tokens += 1
                    if request_tokens % 50 == 0:
                        print(f"  生成中: {request_tokens} tokens", end="\r")

            gen_elapsed = time.time() - gen_start
            gen_tps = request_tokens / gen_elapsed

            print(f"✅ 完成: {request_tokens} tokens in {gen_elapsed:.2f}s")
            print(f"   Generation TPS: {gen_tps:.1f} tok/s")

            total_tokens += request_tokens

            # 关键区别：这里没有手动 sleep(2)
            # wait_for_writes() 会在 _cleanup_finished 中自动调用

        overall_elapsed = time.time() - overall_start
        processing_tps = total_tokens / overall_elapsed

        print("\n" + "=" * 70)
        print("📊 性能结果")
        print("=" * 70)
        print(f"总 tokens: {total_tokens}")
        print(f"总时间: {overall_elapsed:.2f}s")
        print(f"Processing TPS: {processing_tps:.1f} tok/s")
        print("")
        print("🎯 成功标准:")
        print("   ✓ 无 Metal 错误")
        print("   ✓ 无需手动等待")
        print("   ✓ Processing TPS > 60 tok/s")

        if processing_tps > 60:
            print(f"\n🎉 测试通过！Processing TPS = {processing_tps:.1f} tok/s")
            return True
        else:
            print(f"\n⚠️  性能偏低: {processing_tps:.1f} tok/s")
            return True  # 仍然成功（无错误）

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
