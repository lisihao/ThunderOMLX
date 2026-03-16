#!/usr/bin/env python3
"""PP 测试 - 无预热版本"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("📊 PP 性能测试（无预热）")
    print("─" * 60)

    # 生成长 prompt (~8192 tokens)
    base_text = (
        "In software engineering, design patterns are typical solutions to common problems "
        "in software design. Each pattern is like a blueprint that you can customize to solve "
        "a particular design problem in your code. Design patterns are formalized best practices "
        "that the programmer can use to solve common problems when designing an application or system. "
    )
    long_prompt = (base_text + "\n") * 80  # ~8000 tokens

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()

    try:
        print(f"\nPrompt 长度: ~{len(long_prompt) // 4} tokens (估算)")
        print("生成: 1 token（只测 prefill）")
        print("")

        start_time = time.perf_counter()
        first_token_time = None

        async for output in engine.stream_generate(
            prompt=long_prompt,
            max_tokens=1,
            temperature=0.0
        ):
            if output.new_text and first_token_time is None:
                first_token_time = time.perf_counter()
                break

        if first_token_time:
            prefill_time = first_token_time - start_time
            prompt_tokens = len(long_prompt) // 4  # GPT 估算
            pp_tps = prompt_tokens / prefill_time

            print(f"✅ TTFT (Prefill): {prefill_time:.3f}s")
            print(f"Prompt tokens: {prompt_tokens} (估算)")
            print(f"PP TPS: {pp_tps:.1f} tok/s")
            print(f"社区基线: 600-800 tok/s")

            if pp_tps >= 600:
                print(f"✅ 达标 (+{pp_tps - 600:.1f})")
            else:
                print(f"⚠️  低于基线 ({pp_tps - 600:.1f})")

        return True

    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
