#!/usr/bin/env python3
"""测试：缓存禁用（理论上限）"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    print("="*80)
    print("📊 基准测试：缓存禁用（理论上限）")
    print("="*80)
    print()

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    # 生成 8K prompt
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
    ]
    target_tokens = 8192
    text = " ".join(base_words * (target_tokens // len(base_words) + 1))
    tokens = tokenizer.encode(text)[:target_tokens]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)
    print(f"✅ Prompt: {actual_tokens} tokens\n")

    # 配置：禁用缓存
    scheduler_config = SchedulerConfig()
    scheduler_config.paged_ssd_cache_dir = None  # 禁用缓存
    print("🔍 缓存配置: 禁用\n")

    # 初始化引擎
    print("初始化引擎...")
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    await engine.start()
    print("✅ 引擎启动\n")

    try:
        # 预热
        print("预热...")
        async for output in engine.stream_generate(
            prompt="Hello",
            max_tokens=1,
            temperature=0.0
        ):
            pass
        print("✅ 预热完成\n")

        # 测试 Prefill（3次）
        print("测试 8K Prefill (3次)...\n")
        times = []

        for trial in range(3):
            start_time = time.perf_counter()
            first_token_time = None

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=1,
                temperature=0.0
            ):
                if output.new_text:
                    first_token_time = time.perf_counter()
                    break

            if first_token_time:
                prefill_time = first_token_time - start_time
                pp_tps = actual_tokens / prefill_time
                times.append((prefill_time, pp_tps))
                print(f"  Trial {trial + 1}: {prefill_time:.3f}s, {pp_tps:.1f} tok/s")

        if times:
            avg_time = sum(t[0] for t in times) / len(times)
            avg_tps = sum(t[1] for t in times) / len(times)

            print()
            print("="*80)
            print("📊 结果（缓存禁用）")
            print("="*80)
            print()
            print(f"平均 Prefill 时间: {avg_time:.3f}s")
            print(f"平均 PP TPS:       {avg_tps:.1f} tok/s")
            print()
            print("💡 这是理论上限（无缓存 overhead）")

    finally:
        print()
        print("关闭引擎...")
        await engine.stop()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    asyncio.run(main())
