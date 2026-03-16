#!/usr/bin/env python3
"""快速 Metal 并发测试 - 2x32 tokens"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 60)
    print("🧪 快速 Metal 并发测试 (2x32)")
    print("=" * 60)

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(model_name=str(model_path), trust_remote_code=True)
    await engine.start()

    try:
        prompts = [
            "Hello, how are you?",
            "What is Python?"
        ]

        async def generate(prompt, rid):
            tokens = 0
            async for output in engine.stream_generate(prompt=prompt, max_tokens=32, temperature=0.0):
                if output.new_text:
                    tokens += 1
            return rid, tokens

        print("🚀 开始 2 个并发请求...")
        results = await asyncio.gather(
            generate(prompts[0], "R1"),
            generate(prompts[1], "R2")
        )

        for rid, tokens in results:
            print(f"✅ {rid}: {tokens} tokens")

        print("\n🎉 测试通过！无 Metal 错误")
        return True

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
