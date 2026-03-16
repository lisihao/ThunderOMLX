#!/usr/bin/env python3
"""快速 PP 性能测试（无 profiling）"""
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
    print("🔍 快速 PP 测试 - 无 Profiling")
    print("="*80)

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    # 生成 8K prompt
    print("生成 8K prompt...")
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
    ]
    text = " ".join(base_words * 1000)
    tokens = tokenizer.encode(text)[:8192]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)
    print(f"✅ Prompt: {actual_tokens} tokens\n")

    # 初始化引擎（缓存禁用）
    print("初始化引擎（缓存禁用）...")
    scheduler_config = SchedulerConfig(
        paged_ssd_cache_dir=None,
        hot_cache_max_size=0
    )
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    await engine.start()
    print("✅ 引擎启动\n")

    try:
        # Test 1: Prefill
        print("测试 Prefill...")
        start_time = time.perf_counter()
        first_token_time = None

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=1,  # 只生成 1 个 token，测 prefill
            temperature=0.0
        ):
            if output.new_text:
                first_token_time = time.perf_counter()
                break

        if first_token_time:
            prefill_time = first_token_time - start_time
            pp_tps = actual_tokens / prefill_time

            print(f"\n✅ Prefill 完成")
            print(f"   时间: {prefill_time:.3f}s")
            print(f"   PP TPS: {pp_tps:.1f} tok/s\n")

        # Test 2: Generation
        print("测试 Generation...")
        gen_start = time.perf_counter()
        tokens_generated = []

        async for output in engine.stream_generate(
            prompt="Hello",  # 短 prompt
            max_tokens=128,
            temperature=0.7
        ):
            if output.new_text:
                tokens_generated.append(time.perf_counter())

        gen_time = time.perf_counter() - gen_start
        tg_tps = len(tokens_generated) / gen_time if gen_time > 0 else 0

        print(f"\n✅ Generation 完成")
        print(f"   生成 tokens: {len(tokens_generated)}")
        print(f"   时间: {gen_time:.3f}s")
        print(f"   TG TPS: {tg_tps:.1f} tok/s\n")

    finally:
        await engine.stop()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    asyncio.run(main())
