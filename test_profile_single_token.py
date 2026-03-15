#!/usr/bin/env python3
"""性能 profiling: 单 token 生成"""

import asyncio
import time
from pathlib import Path

from mlx_lm import load


async def profile_single_token():
    """专注于单 token 生成的性能分析"""

    # 1. 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

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
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "profile"),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 3. Warm up (建立缓存)
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    await engine.generate(prompt=prompt, sampling_params=sampling_params)

    # 4. Profile 第二次生成 (FULL SKIP)
    print("开始 profiling (FULL SKIP)...")
    print("运行 10 次取平均...")

    times = []
    for i in range(10):
        start = time.perf_counter()
        await engine.generate(prompt=prompt, sampling_params=sampling_params)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  第 {i+1} 次: {elapsed:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"\n平均时间: {avg_time:.2f} ms")


if __name__ == "__main__":
    asyncio.run(profile_single_token())
