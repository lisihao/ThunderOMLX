#!/usr/bin/env python3
"""Python cProfile 性能分析"""

import asyncio
import cProfile
import pstats
import io
from pathlib import Path

from mlx_lm import load


async def run_profiling():
    """使用 cProfile 进行性能分析"""

    # 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    # 创建 EngineCore
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

    # Warm up
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    await engine.generate(prompt=prompt, sampling_params=sampling_params)

    # Profiling 运行
    print("\n" + "=" * 80)
    print("开始 cProfile 分析...")
    print("=" * 80)

    # 创建 profiler
    profiler = cProfile.Profile()

    # 开始 profiling
    profiler.enable()

    # 运行 5 次 FULL SKIP 生成
    for i in range(5):
        await engine.generate(prompt=prompt, sampling_params=sampling_params)
        print(f"  完成第 {i+1} 次生成")

    # 停止 profiling
    profiler.disable()

    # 输出结果
    print("\n" + "=" * 80)
    print("cProfile 分析结果（按累计时间排序）")
    print("=" * 80)

    # 创建统计对象
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    # 按累计时间排序，显示前 50 个函数
    stats.sort_stats('cumulative')
    stats.print_stats(50)

    # 输出到屏幕
    print(stream.getvalue())

    # 保存到文件
    with open('profiling/cprofile_output.txt', 'w') as f:
        f.write(stream.getvalue())

    print("\n" + "=" * 80)
    print("分析结果已保存到: profiling/cprofile_output.txt")
    print("=" * 80)

    # 按自身时间排序
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats('tottime')
    stats2.print_stats(50)

    with open('profiling/cprofile_tottime.txt', 'w') as f:
        f.write(stream2.getvalue())

    print("\n按自身时间排序的结果已保存到: profiling/cprofile_tottime.txt")


if __name__ == "__main__":
    asyncio.run(run_profiling())
