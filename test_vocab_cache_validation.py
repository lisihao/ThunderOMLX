#!/usr/bin/env python3
"""测试 vocab 缓存校验机制"""

import asyncio
import logging
from pathlib import Path

from mlx_lm import load

# 设置日志级别为 INFO，确保能看到警告信息
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


async def test_vocab_cache_validation():
    """测试 vocab 缓存校验"""

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
    print("测试：缓存元数据已被破坏，应该从 tokenizer 重新加载并修复缓存")
    print("=" * 80)

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    print("\n" + "=" * 80)
    print("检查缓存文件是否已修复")
    print("=" * 80)

    # 检查缓存文件
    cache_dir = Path.home() / ".cache" / "omlx" / "vocab_cache"
    model_name = str(model_path).replace("/", "_").replace("\\", "_")
    cache_meta = cache_dir / f"{model_name}_vocab.meta"

    if cache_meta.exists():
        with open(cache_meta, 'r') as f:
            lines = f.readlines()
            vocab_size = lines[0].strip()
            vocab_hash = lines[1].strip()
            print(f"✅ 缓存元数据已修复:")
            print(f"   Vocab size: {vocab_size}")
            print(f"   Hash: {vocab_hash[:16]}...")

            if vocab_size == "99999":
                print("❌ 缓存未修复，仍然是错误的值！")
            else:
                print("✅ 缓存已成功修复！")


if __name__ == "__main__":
    asyncio.run(test_vocab_cache_validation())
