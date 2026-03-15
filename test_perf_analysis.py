#!/usr/bin/env python3
"""性能分析：找出 FULL SKIP 的额外开销"""

import asyncio
import logging
import time
from pathlib import Path

from mlx_lm import load

# 设置日志级别
logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')


async def test_perf_analysis():
    """详细的性能分析"""

    print("=" * 80)
    print("ThunderOMLX 性能分析")
    print("=" * 80)

    # 1. 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))
    print(f"✅ 模型加载完成")

    # 2. 创建 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "perf_test"),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 3. 测试用例
    prompt = "请详细解释人工智能的发展历史" * 10  # 构造长 prompt
    tokens = tokenizer.encode(prompt)
    print(f"\nPrompt tokens: {len(tokens)}")

    sampling_params = SamplingParams(max_tokens=1)  # 只生成 1 token

    # Test 1: Cold Start
    print("\n" + "─" * 80)
    print("Test 1: Cold Start")
    print("─" * 80)

    start = time.perf_counter()
    output1 = await engine.generate(prompt=prompt, sampling_params=sampling_params)
    time1 = (time.perf_counter() - start) * 1000

    print(f"总时间: {time1:.2f} ms")
    print(f"输出: {output1.output_text[:20]}...")

    # Test 2: FULL SKIP
    print("\n" + "─" * 80)
    print("Test 2: FULL SKIP (100% 重复)")
    print("─" * 80)

    start = time.perf_counter()
    output2 = await engine.generate(prompt=prompt, sampling_params=sampling_params)
    time2 = (time.perf_counter() - start) * 1000

    print(f"总时间: {time2:.2f} ms")
    print(f"输出: {output2.output_text[:20]}...")

    # 4. 性能分析
    print("\n" + "=" * 80)
    print("性能分析")
    print("=" * 80)

    speedup = time1 / time2

    # 按 ThunderLLAMA 的比例推算
    expected_prefill = time1 * 0.979  # 97.9%
    expected_generate = time1 * 0.021  # 2.1%

    # FULL SKIP 应该的时间
    expected_cache_load = 8  # ~8ms (ThunderLLAMA)
    expected_full_skip = expected_cache_load + expected_generate

    # 额外开销
    extra_overhead = time2 - expected_full_skip

    print(f"\n推算（基于 ThunderLLAMA 97.9%/2.1% 比例）:")
    print(f"  Prefill 应该是: {expected_prefill:.2f} ms")
    print(f"  Generate 1 token 应该是: {expected_generate:.2f} ms")
    print(f"  FULL SKIP 应该是: {expected_full_skip:.2f} ms (缓存加载 {expected_cache_load}ms + 生成 {expected_generate:.2f}ms)")

    print(f"\n实际测量:")
    print(f"  Cold Start: {time1:.2f} ms")
    print(f"  FULL SKIP: {time2:.2f} ms")
    print(f"  加速比: {speedup:.2f}x")

    print(f"\n额外开销分析:")
    print(f"  期望时间: {expected_full_skip:.2f} ms")
    print(f"  实际时间: {time2:.2f} ms")
    print(f"  额外开销: {extra_overhead:.2f} ms")
    print(f"  开销占比: {(extra_overhead / time2) * 100:.1f}%")

    # 5. ThunderLLAMA 对比
    print(f"\n与 ThunderLLAMA 对比:")
    print(f"  ThunderLLAMA FULL SKIP: 23ms (加速比 30.56x)")
    print(f"  ThunderOMLX FULL SKIP: {time2:.2f}ms (加速比 {speedup:.2f}x)")
    print(f"  性能差距: {time2 / 23:.2f}x")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_perf_analysis())
