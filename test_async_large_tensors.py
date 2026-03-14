"""测试大张量场景下的异步 I/O 性能"""
import asyncio
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.thunder_config import SerializationConfig


async def benchmark_large_tensors():
    """测试大张量（I/O bound）场景"""
    print("=== 大张量异步 I/O 基准测试 ===\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建缓存管理器（L2: 10MB，L3: 2GB）
        config = SerializationConfig(compression="zlib", enable_checksum=True)
        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=10,
            l3_cache_path=tmpdir / "l3_cache",
            l3_max_size_gb=2,
            serialization_config=config,
        )

        # 准备测试数据：创建 8 个大张量（每个 10MB）
        print("[准备阶段: 创建大张量]")
        num_tensors = 8
        tensor_size_mb = 10
        keys = []
        tensors = []

        for i in range(num_tensors):
            # 10MB 张量：1600x1600 float32 = ~10MB
            tensor = mx.random.normal(shape=(1600, 1600))
            key = f"large_tensor_{i}"
            cache_mgr.store(key, tensor)
            keys.append(key)
            tensors.append(tensor)

        # 清空 L2，强制从 L3 读取
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        print(f"  创建 {num_tensors} 个张量，每个 ~{tensor_size_mb} MB")
        print(f"  L3 大小: {cache_mgr.l3_size_bytes / (1024**2):.2f} MB\n")

        # 测试 1: 串行加载
        print("[测试 1: 串行加载]")

        start = time.perf_counter()
        serial_results = []
        for key in keys:
            value, hit = cache_mgr.fetch(key)
            serial_results.append((value, hit))
        serial_time = (time.perf_counter() - start) * 1000

        print(f"  串行加载时间: {serial_time:.2f} ms")
        print(f"  平均每个: {serial_time / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (serial_time / 1000):.1f} MB/s\n")

        # 测试 2: 线程池并行加载
        print("[测试 2: 线程池并行加载（8 线程）]")

        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        parallel_results = cache_mgr.batch_fetch_parallel(keys, max_workers=8)
        parallel_time = (time.perf_counter() - start) * 1000

        print(f"  并行加载时间: {parallel_time:.2f} ms")
        print(f"  平均每个: {parallel_time / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (parallel_time / 1000):.1f} MB/s")

        # 性能对比
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"  加速比: {speedup:.2f}x")
        print(f"  验收标准: {'✅ 通过' if speedup >= 3 else '⚠️ 未达标'}  (目标 ≥ 3x)\n")

        # 验证数据正确性
        all_correct = all(
            mx.allclose(tensors[i], parallel_results[i][0])
            for i in range(num_tensors)
            if parallel_results[i][0] is not None
        )
        print(f"  数据正确性: {'✅' if all_correct else '❌'}\n")

    print("=== 大张量测试完成 ===")


if __name__ == "__main__":
    asyncio.run(benchmark_large_tensors())
