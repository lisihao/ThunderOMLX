#!/usr/bin/env python3
"""
8K PP 测试 - 单次测试，避免 Metal 并发问题
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 70)
    print("📊 8K PP 性能测试（单次）")
    print("=" * 70)
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 生成 8K tokens prompt
    base = "word " * 100
    repeats = 8000 // 100
    text = base * repeats

    print("初始化引擎...")
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()
    print("✅ 引擎启动\n")

    try:
        print(f"测试 ~8K tokens...")
        print(f"Prompt 长度: ~{len(text) // 4} tokens (估算)")
        print("")

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

            print("=" * 70)
            print("✅ 结果")
            print("=" * 70)
            print(f"Tokens: {actual_tokens}")
            print(f"Prefill 时间: {prefill:.3f}s")
            print(f"PP TPS: {pp_tps:.1f} tok/s")
            print(f"社区基线: 637-693 tok/s")
            print("")

            if pp_tps >= 637:
                print(f"✅ 达到基线！(+{pp_tps - 665:.1f} tok/s from 平均)")
            else:
                print(f"⚠️  低于基线 ({pp_tps - 637:.1f} tok/s)")

            return True
        else:
            print("❌ 未生成 token")
            return False

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
