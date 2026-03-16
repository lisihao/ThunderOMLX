#!/usr/bin/env python3
"""最小化测试 - 单请求"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("🔧 初始化引擎...")
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )

    print("🚀 启动引擎...")
    await engine.start()

    try:
        print("📝 生成测试...")
        tokens = 0
        async for output in engine.stream_generate(
            prompt="Say hello",
            max_tokens=10,
            temperature=0.0
        ):
            if output.new_text:
                tokens += 1
                print(f"Token {tokens}: {output.new_text}", end="", flush=True)

        print(f"\n✅ 完成: {tokens} tokens")
        return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("🛑 停止引擎...")
        await engine.stop()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
