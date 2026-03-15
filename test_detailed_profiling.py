#!/usr/bin/env python3
"""详细性能分析：分解每个步骤的时间"""

import asyncio
import time
from pathlib import Path
import mlx.core as mx

from mlx_lm import load


async def detailed_profiling():
    """详细分解每个步骤的时间"""

    print("=" * 80)
    print("详细性能分析")
    print("=" * 80)

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

    # 3. Warm up
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    await engine.generate(prompt=prompt, sampling_params=sampling_params)

    # 4. 详细 profiling
    print("\n开始详细 profiling (运行 5 次取平均)...")

    # 收集时间数据
    timings = {
        'total': [],
        'scheduler_step': [],
        'model_forward': [],
        'sampling': [],
        'cache_ops': [],
    }

    # 注入计时器到 scheduler
    original_step = engine.scheduler._step

    def timed_step(*args, **kwargs):
        start = time.perf_counter()
        result = original_step(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        timings['scheduler_step'].append(elapsed)
        return result

    engine.scheduler._step = timed_step

    for i in range(5):
        print(f"\n第 {i+1} 次:")

        # 总时间
        start_total = time.perf_counter()

        # 执行生成
        await engine.generate(prompt=prompt, sampling_params=sampling_params)

        total_time = (time.perf_counter() - start_total) * 1000
        timings['total'].append(total_time)

        print(f"  总时间: {total_time:.2f} ms")
        if timings['scheduler_step']:
            print(f"  _step() 调用: {timings['scheduler_step'][-1]:.2f} ms")

    # 5. 分析结果
    print("\n" + "=" * 80)
    print("性能分解（平均值）")
    print("=" * 80)

    avg_total = sum(timings['total']) / len(timings['total'])
    avg_step = sum(timings['scheduler_step']) / len(timings['scheduler_step']) if timings['scheduler_step'] else 0

    print(f"\n总时间: {avg_total:.2f} ms")
    print(f"_step() 调用: {avg_step:.2f} ms ({(avg_step/avg_total)*100:.1f}%)")
    print(f"其他开销: {avg_total - avg_step:.2f} ms ({((avg_total - avg_step)/avg_total)*100:.1f}%)")

    # 6. MLX 操作 profiling
    print("\n" + "=" * 80)
    print("MLX 操作性能测试")
    print("=" * 80)

    # 测试 mx.eval 开销
    print("\n测试 mx.eval() 开销:")
    cache_states = [engine.scheduler._prompt_cache[0].state for _ in range(5)]

    times = []
    for _ in range(10):
        start = time.perf_counter()
        mx.eval(cache_states)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"  平均: {sum(times)/len(times):.2f} ms")

    # 测试 mx.clear_cache 开销
    print("\n测试 mx.clear_cache() 开销:")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        mx.clear_cache()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"  平均: {sum(times)/len(times):.2f} ms")


if __name__ == "__main__":
    asyncio.run(detailed_profiling())
