#!/usr/bin/env python3
"""调试 FULL SKIP 模式未触发的原因"""

import asyncio
import logging
from pathlib import Path
from mlx_lm import load

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def debug_full_skip():
    """调试 FULL SKIP 模式"""

    # 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    # 创建 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    paged_ssd_cache_dir = str(Path.home() / ".cache" / "omlx" / "profile_debug")

    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=paged_ssd_cache_dir,
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 测试 prompt（故意使用相同的 prompt）
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    print("\n" + "=" * 80)
    print("🔍 调试信息：")
    print(f"Paged SSD Cache Dir: {paged_ssd_cache_dir}")
    print(f"Prompt: {prompt[:50]}... (长度: {len(prompt)})")
    print("=" * 80)

    # Warm up
    print("\n🔥 Warm-up 生成（建立初始缓存）")
    result = await engine.generate(prompt=prompt, sampling_params=sampling_params)
    print(f"✅ Warm-up 完成，输出: {result.text[:50]}")

    # 检查 scheduler 状态
    print("\n🔍 检查 Scheduler 状态:")
    print(f"  - Scheduler class: {type(engine.scheduler).__name__}")
    print(f"  - Batch generator: {type(engine.scheduler.batch_generator).__name__}")
    print(f"  - Has paged_ssd_cache: {hasattr(engine.scheduler, 'paged_ssd_cache')}")
    if hasattr(engine.scheduler, 'paged_ssd_cache'):
        cache = engine.scheduler.paged_ssd_cache
        print(f"  - Paged SSD cache class: {type(cache).__name__}")
        print(f"  - Cache dir: {cache.cache_dir if hasattr(cache, 'cache_dir') else 'N/A'}")

    # 运行 5 次相同 prompt 的生成（应该触发 FULL SKIP）
    print("\n" + "=" * 80)
    print("🔁 开始 5 次相同 prompt 生成（期望触发 FULL SKIP）")
    print("=" * 80)

    for i in range(5):
        print(f"\n📍 第 {i+1} 次生成")
        print("-" * 40)

        # 检查生成前的缓存状态
        if hasattr(engine.scheduler, 'paged_ssd_cache'):
            cache = engine.scheduler.paged_ssd_cache
            if hasattr(cache, '_blocks'):
                print(f"  🗄️  缓存块数: {len(cache._blocks)}")

        result = await engine.generate(prompt=prompt, sampling_params=sampling_params)

        # 检查是否触发了 FULL SKIP
        if hasattr(engine.scheduler, 'batch_generator'):
            bg = engine.scheduler.batch_generator
            if hasattr(bg, '_last_batch_was_full_skip'):
                print(f"  ✨ Full Skip: {bg._last_batch_was_full_skip}")

        print(f"  ✅ 输出: {result.text[:50]}")

    print("\n" + "=" * 80)
    print("🏁 调试完成")
    print("=" * 80)

    # 检查最终缓存状态
    print("\n🗄️  最终缓存状态:")
    if hasattr(engine.scheduler, 'paged_ssd_cache'):
        cache = engine.scheduler.paged_ssd_cache
        if hasattr(cache, '_blocks'):
            print(f"  - 总缓存块数: {len(cache._blocks)}")
        if hasattr(cache, 'cache_dir'):
            cache_dir_path = Path(cache.cache_dir)
            if cache_dir_path.exists():
                cache_files = list(cache_dir_path.glob('*'))
                print(f"  - 缓存文件数: {len(cache_files)}")
                if cache_files:
                    print(f"  - 示例缓存文件: {cache_files[0].name}")


if __name__ == "__main__":
    asyncio.run(debug_full_skip())
