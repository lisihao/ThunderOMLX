#!/usr/bin/env python3
"""使用 line_profiler 进行逐行性能分析"""

import asyncio
from pathlib import Path
from mlx_lm import load


async def run_profiling():
    """使用 line_profiler 进行逐行性能分析"""

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

    print("\n" + "=" * 80)
    print("开始 line_profiler 分析...")
    print("=" * 80)

    # 注入 @profile 装饰器到 _process_prompts
    from line_profiler import LineProfiler

    profiler = LineProfiler()
    # batch_generator 在 warm up 后已创建
    if engine.scheduler.batch_generator is not None:
        profiler.add_function(engine.scheduler.batch_generator._process_prompts)
    else:
        print("⚠️  batch_generator 未初始化，无法 profile _process_prompts")
        return

    # 启用 profiler
    profiler.enable()

    # 运行 5 次 FULL SKIP 生成
    for i in range(5):
        await engine.generate(prompt=prompt, sampling_params=sampling_params)
        print(f"  完成第 {i+1} 次生成")

    # 停止 profiler
    profiler.disable()

    # 输出结果
    print("\n" + "=" * 80)
    print("line_profiler 分析结果")
    print("=" * 80)

    # 保存到文件
    with open('profiling/line_profiler_output.txt', 'w') as f:
        profiler.print_stats(stream=f)

    # 同时输出到屏幕
    profiler.print_stats()

    print("\n" + "=" * 80)
    print("分析结果已保存到: profiling/line_profiler_output.txt")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_profiling())
