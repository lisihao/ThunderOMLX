#!/usr/bin/env python3
"""TG 测试 - 无预热版本"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("📊 TG 性能测试（无预热）")
    print("─" * 60)

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()

    try:
        prompt = "Explain Python and JavaScript differences."

        print("\n测试 1 次（512 tokens）...")
        first_token_time = None
        last_token_time = None
        token_count = 0

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        ):
            if output.new_text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                last_token_time = time.perf_counter()
                token_count += 1

                if token_count % 100 == 0:
                    print(f"  {token_count} tokens", end="\r")

        if first_token_time and last_token_time:
            gen_time = last_token_time - first_token_time
            tg_tps = token_count / gen_time

            print(f"\n✅ 完成: {token_count} tokens in {gen_time:.2f}s")
            print(f"TG TPS: {tg_tps:.1f} tok/s")
            print(f"社区基线: 60-80+ tok/s")

            if tg_tps >= 60:
                print(f"✅ 达标 (+{tg_tps - 70:.1f} from 70)")
            else:
                print(f"⚠️  低于基线 ({tg_tps - 60:.1f})")

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
