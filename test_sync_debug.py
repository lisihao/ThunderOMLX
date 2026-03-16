#!/usr/bin/env python3
"""测试 writer 同步 - 带调试输出"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("🧪 Writer 同步测试（调试模式）")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print("[1/5] 创建引擎...")
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )

    print("[2/5] 启动引擎...")
    await engine.start()

    try:
        # 第一个请求
        print("\n[3/5] 第 1 个请求（32 tokens）...")
        tokens_1 = 0
        async for output in engine.stream_generate(
            prompt="Say hello",
            max_tokens=32,
            temperature=0.0
        ):
            if output.new_text:
                tokens_1 += 1
        print(f"✅ 第 1 个请求完成: {tokens_1} tokens")

        # 第二个请求（立即开始，无手动等待）
        print("\n[4/5] 第 2 个请求（32 tokens）- 无手动等待...")
        tokens_2 = 0
        async for output in engine.stream_generate(
            prompt="Say goodbye",
            max_tokens=32,
            temperature=0.0
        ):
            if output.new_text:
                tokens_2 += 1
        print(f"✅ 第 2 个请求完成: {tokens_2} tokens")

        print(f"\n🎉 成功！总计 {tokens_1 + tokens_2} tokens，无 Metal 错误")
        return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n[5/5] 停止引擎...")
        await engine.stop()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
