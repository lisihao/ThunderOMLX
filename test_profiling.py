#!/usr/bin/env python3
"""
简单的性能分析测试 - 使用修改后的 scheduler
"""
import asyncio
import sys
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🧪 性能分析测试 - pp8192/tg128")
    print("=" * 80)

    # 创建引擎
    print("\n📦 加载模型...")
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        # 生成 8192 token prompt
        print("📝 生成 8192 token prompt...")
        filler = "The quick brown fox jumps over the lazy dog. " * 1000
        tokens = engine.tokenizer.encode(filler)[:8192]
        prompt = engine.tokenizer.decode(tokens)

        actual_tokens = len(engine.tokenizer.encode(prompt))
        print(f"✅ Prompt tokens: {actual_tokens}")

        # 生成 128 tokens
        print(f"\n🚀 开始生成 128 tokens...")
        print(f"   每 50 tokens 会打印性能统计\n")

        count = 0
        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=128,
            temperature=0.0
        ):
            if output.new_text:
                count += 1
                if count % 10 == 0:
                    print(f"  生成进度: {count} tokens", end="\r")

        print(f"\n\n✅ 生成完成! 总共生成: {count} tokens")

    finally:
        await engine.stop()

    print("\n" + "=" * 80)
    print("📊 查看 /tmp/omlx-server-profiling.log 中的性能统计")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
