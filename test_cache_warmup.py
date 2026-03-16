#!/usr/bin/env python3
"""
正确的缓存预热脚本：
1. 第一次运行：warm-up，建立缓存
2. 等待缓存写入完成
3. 第二次运行：5 次生成，应全部触发 FULL SKIP
"""

import asyncio
import time
import logging
from pathlib import Path
from mlx_lm import load

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,  # 使用 INFO 级别，避免太多 DEBUG 信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def warmup_cache():
    """第一步：Warm-up，建立缓存"""

    print("\n" + "=" * 80)
    print("🔥 第一步：Warm-up（建立缓存）")
    print("=" * 80)

    # 清空缓存目录
    cache_dir = Path.home() / ".cache" / "omlx" / "profile"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✅ 清空缓存目录: {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)

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
        paged_ssd_cache_dir=str(cache_dir),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 测试 prompt
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    # Warm-up 生成
    print(f"\n📍 Warm-up 生成（prompt 长度: {len(prompt)} 字符）")
    result = await engine.generate(prompt=prompt, sampling_params=sampling_params)
    print(f"✅ Warm-up 完成")

    # 检查缓存写入队列
    print("\n⏳ 等待缓存写入到 SSD...")
    if hasattr(engine.scheduler, 'paged_ssd_cache'):
        cache_manager = engine.scheduler.paged_ssd_cache
        if hasattr(cache_manager, 'async_writer'):
            writer = cache_manager.async_writer
            # 等待写入队列为空
            max_wait = 10  # 最多等待 10 秒
            for i in range(max_wait * 10):
                if hasattr(writer, 'write_queue'):
                    queue_size = writer.write_queue.qsize()
                    if queue_size == 0:
                        print(f"✅ 写入队列已清空")
                        break
                    else:
                        print(f"  - 队列中还有 {queue_size} 个块待写入...")
                await asyncio.sleep(0.1)

    # 额外等待 1 秒确保所有异步写入完成
    await asyncio.sleep(1.0)

    # 检查缓存文件
    cache_files = list(cache_dir.rglob('*.safetensors*'))
    print(f"\n📊 缓存文件检查:")
    print(f"  - 缓存目录: {cache_dir}")
    print(f"  - 文件数量: {len(cache_files)}")
    if cache_files:
        for f in cache_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name}: {size_mb:.2f} MB")
    else:
        print("  ⚠️  警告：没有找到缓存文件！")

    # 关闭 engine（触发最终的缓存写入）
    # await engine.shutdown()  # 如果有 shutdown 方法

    print("\n" + "=" * 80)
    print("🏁 Warm-up 完成，缓存已写入")
    print("=" * 80)


async def test_full_skip():
    """第二步：5 次生成，测试 FULL SKIP"""

    print("\n" + "=" * 80)
    print("🔁 第二步：5 次生成（测试 FULL SKIP）")
    print("=" * 80)

    cache_dir = Path.home() / ".cache" / "omlx" / "profile"

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
        paged_ssd_cache_dir=str(cache_dir),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 测试 prompt（与 warm-up 相同）
    prompt = "请详细解释人工智能" * 2
    sampling_params = SamplingParams(max_tokens=1)

    # 运行 5 次生成
    full_skip_count = 0
    for i in range(5):
        print(f"\n📍 第 {i+1} 次生成")
        result = await engine.generate(prompt=prompt, sampling_params=sampling_params)
        print(f"✅ 生成完成")

        # 注意：这里无法直接检测 FULL SKIP，需要从日志中查看

    print("\n" + "=" * 80)
    print("🏁 5 次生成完成")
    print("=" * 80)
    print("\n💡 提示：检查日志中是否有 '✨ [Full Skip Batch]' 字样")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 第二步：测试 FULL SKIP
        asyncio.run(test_full_skip())
    else:
        # 第一步：Warm-up
        asyncio.run(warmup_cache())
