#!/usr/bin/env python3
"""详细性能测试 - 每步都打印状态"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 60)
    print("🔍 详细性能测试（逐步追踪）")
    print("=" * 60)

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print("\n[1/6] 创建引擎...")
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )

    print("[2/6] 启动引擎...")
    await engine.start()
    print("✅ 引擎启动成功")

    try:
        # 第一个请求
        print("\n[3/6] 执行第 1 个请求...")
        tokens_1 = 0
        async for output in engine.stream_generate(
            prompt="Hello",
            max_tokens=32,
            temperature=0.0
        ):
            if output.new_text:
                tokens_1 += 1

        print(f"✅ 第 1 个请求完成: {tokens_1} tokens")

        # 等待一下让清理完成
        print("[4/6] 等待清理...")
        await asyncio.sleep(2)
        print("✅ 清理完成")

        # 第二个请求
        print("\n[5/6] 执行第 2 个请求...")
        tokens_2 = 0
        async for output in engine.stream_generate(
            prompt="World",
            max_tokens=32,
            temperature=0.0
        ):
            if output.new_text:
                tokens_2 += 1

        print(f"✅ 第 2 个请求完成: {tokens_2} tokens")

        print(f"\n总计: {tokens_1 + tokens_2} tokens")
        return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n[6/6] 停止引擎...")
        await engine.stop()
        print("✅ 完成")

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
