#!/usr/bin/env python3
"""测试 vocab 磁盘缓存功能"""

import asyncio
from pathlib import Path

from mlx_lm import load


async def test_vocab_cache():
    """测试 vocab 缓存"""

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig

    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    print("=" * 80)
    print("第一次启动（应该从 tokenizer 加载并缓存）")
    print("=" * 80)

    engine1 = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine1.start()

    print("\n" + "=" * 80)
    print("第二次启动（应该从磁盘缓存加载，非常快）")
    print("=" * 80)

    engine2 = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine2.start()

    print("\n" + "=" * 80)
    print("测试完成！检查日志中的加载时间")
    print("=" * 80)

    # 检查缓存文件
    cache_dir = Path.home() / ".cache" / "omlx" / "vocab_cache"
    model_name = str(model_path).replace("/", "_").replace("\\", "_")
    cache_file = cache_dir / f"{model_name}_vocab.pkl"
    cache_meta = cache_dir / f"{model_name}_vocab.meta"

    if cache_file.exists():
        size_kb = cache_file.stat().st_size / 1024
        print(f"\n✅ 缓存文件已创建: {cache_file} ({size_kb:.1f} KB)")

        # 读取元数据
        if cache_meta.exists():
            with open(cache_meta, 'r') as f:
                lines = f.readlines()
                vocab_size = lines[0].strip()
                vocab_hash = lines[1].strip()
                print(f"   Vocab size: {vocab_size}")
                print(f"   Hash: {vocab_hash}")
    else:
        print(f"\n❌ 缓存文件未创建")


if __name__ == "__main__":
    asyncio.run(test_vocab_cache())
