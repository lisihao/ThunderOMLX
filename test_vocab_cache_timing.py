#!/usr/bin/env python3
"""测试 vocab 磁盘缓存的性能"""

import asyncio
import time
from pathlib import Path

from mlx_lm import load


async def test_vocab_cache_timing():
    """测试 vocab 缓存性能"""

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
    print("第一次启动（缓存已存在，应该从磁盘加载）")
    print("=" * 80)

    start = time.perf_counter()
    engine1 = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine1.start()
    elapsed1 = (time.perf_counter() - start) * 1000
    print(f"第一次启动耗时: {elapsed1:.2f} ms")

    # 清除第一个 engine 的 vocab 缓存，强制第二次从磁盘加载
    del engine1

    print("\n" + "=" * 80)
    print("第二次启动（也应该从磁盘缓存加载）")
    print("=" * 80)

    start = time.perf_counter()
    engine2 = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine2.start()
    elapsed2 = (time.perf_counter() - start) * 1000
    print(f"第二次启动耗时: {elapsed2:.2f} ms")

    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    print(f"第一次启动: {elapsed1:.2f} ms")
    print(f"第二次启动: {elapsed2:.2f} ms")
    print(f"加速比: {elapsed1 / elapsed2:.1f}x" if elapsed2 > 0 else "N/A")

    # 检查缓存文件
    cache_dir = Path.home() / ".cache" / "omlx" / "vocab_cache"
    model_name = str(model_path).replace("/", "_").replace("\\", "_")
    cache_file = cache_dir / f"{model_name}_vocab.pkl"
    cache_meta = cache_dir / f"{model_name}_vocab.meta"

    if cache_file.exists():
        size_kb = cache_file.stat().st_size / 1024
        print(f"\n✅ 缓存文件: {cache_file} ({size_kb:.1f} KB)")

        # 读取元数据
        if cache_meta.exists():
            with open(cache_meta, 'r') as f:
                lines = f.readlines()
                vocab_size = lines[0].strip()
                vocab_hash = lines[1].strip()
                print(f"   Vocab size: {vocab_size}")
                print(f"   Hash: {vocab_hash[:16]}...")

    # 手动测试直接从磁盘加载的性能
    print("\n" + "=" * 80)
    print("直接从磁盘加载 pickle 的性能")
    print("=" * 80)

    import pickle

    times = []
    for i in range(5):
        start = time.perf_counter()
        with open(cache_file, 'rb') as f:
            vocab = pickle.load(f)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"第 {i+1} 次: {elapsed:.2f} ms")

    print(f"平均: {sum(times)/len(times):.2f} ms")


if __name__ == "__main__":
    asyncio.run(test_vocab_cache_timing())
