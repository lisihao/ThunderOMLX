#!/usr/bin/env python3
"""性能测试 - 128 tokens + 请求间等待"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 60)
    print("🧪 Phase 1-4 性能验证（带清理等待）")
    print("=" * 60)
    print("配置: 2 个顺序请求 × 128 tokens + 2s 清理等待")
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
            print(f"\n{'─' * 60}")
            print(f"📝 Request {i}/2")

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

            # 关键：等待后台清理完成
            if i < len(prompts):
                print("   等待清理...")
                await asyncio.sleep(2)

        overall_elapsed = time.time() - overall_start

        # 减去等待时间得到真实处理时间
        actual_processing_time = overall_elapsed - (len(prompts) - 1) * 2
        processing_tps = total_tokens / actual_processing_time

        print("\n" + "=" * 60)
        print("📊 性能结果")
        print("=" * 60)
        print(f"总 tokens: {total_tokens}")
        print(f"总时间（含等待）: {overall_elapsed:.2f}s")
        print(f"实际处理时间: {actual_processing_time:.2f}s")
        print(f"Processing TPS: {processing_tps:.1f} tok/s")
        print("")

        return True

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
