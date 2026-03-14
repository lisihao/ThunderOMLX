"""P3-4: 异步 I/O 批量加载基准测试"""
import asyncio
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.thunder_config import SerializationConfig


async def benchmark_async_io():
    """基准测试：串行 vs 并行加载"""
    print("=== P3-4: 异步 I/O 批量加载基准测试 ===\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建缓存管理器（L2: 4MB，L3: 1GB）
        config = SerializationConfig(compression="zlib", enable_checksum=True)
        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=4,  # 小 L2，强制使用 L3
            l3_cache_path=tmpdir / "l3_cache",
            l3_max_size_gb=1,
            serialization_config=config,
        )

        # 准备测试数据：创建 16 个张量，全部存入 L3
        print("[准备阶段: 创建测试数据]")
        num_tensors = 16
        tensor_size_mb = 1  # 每个 1MB
        keys = []
        tensors = []

        for i in range(num_tensors):
            tensor = mx.random.normal(shape=(512, 512))  # ~1MB
            key = f"async_test_tensor_{i}"
            cache_mgr.store(key, tensor)
            keys.append(key)
            tensors.append(tensor)

        # 清空 L2，强制从 L3 读取
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        print(f"  创建 {num_tensors} 个张量，总大小 ~{num_tensors * tensor_size_mb} MB")
        print(f"  L3 大小: {cache_mgr.l3_size_bytes / (1024**2):.2f} MB")
        print(f"  L3 条目数: {len(cache_mgr.l3_index)}\n")

        # 测试 1: 串行加载（8 个张量）
        print("[测试 1: 串行加载 8 个张量]")
        batch_size = 8
        test_keys = keys[:batch_size]

        # 清空 L2，重置统计
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        serial_results = []
        for key in test_keys:
            value, hit = cache_mgr.fetch(key)
            serial_results.append((value, hit))
        serial_time = (time.perf_counter() - start) * 1000

        serial_hits = sum(1 for _, hit in serial_results if hit)

        print(f"  串行加载时间: {serial_time:.2f} ms")
        print(f"  命中数: {serial_hits}/{batch_size}")
        print(f"  平均每个: {serial_time / batch_size:.2f} ms\n")

        # 测试 2: 并行加载（8 个张量）
        print("[测试 2: 异步并行加载 8 个张量]")

        # 清空 L2，重置统计
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        parallel_results = await cache_mgr.batch_fetch(test_keys)
        parallel_time = (time.perf_counter() - start) * 1000

        parallel_hits = sum(1 for _, hit in parallel_results if hit)

        print(f"  并行加载时间: {parallel_time:.2f} ms")
        print(f"  命中数: {parallel_hits}/{batch_size}")
        print(f"  平均每个: {parallel_time / batch_size:.2f} ms\n")

        # 性能对比
        print("[性能对比]")
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        print(f"  加速比: {speedup:.2f}x")
        print(f"  时间节省: {serial_time - parallel_time:.2f} ms ({(1 - parallel_time/serial_time) * 100:.1f}%)")
        print(f"  验收标准: {'✅ 通过' if speedup >= 10 else '⚠️ 未达标'}  (目标 ≥ 10x)\n")

        # 验证数据正确性
        print("[数据正确性验证]")
        all_correct = True
        for i, ((serial_val, _), (parallel_val, _)) in enumerate(zip(serial_results, parallel_results)):
            if serial_val is None or parallel_val is None:
                all_correct = False
                print(f"  ❌ 张量 {i}: 加载失败")
            elif not mx.allclose(serial_val, parallel_val):
                all_correct = False
                print(f"  ❌ 张量 {i}: 数据不一致")
            elif not mx.allclose(tensors[i], parallel_val):
                all_correct = False
                print(f"  ❌ 张量 {i}: 与原始数据不一致")

        if all_correct:
            print(f"  ✅ 所有 {batch_size} 个张量数据正确")
        print()

        # 测试 3: 线程池并行加载（8 个张量）
        print("[测试 3: 线程池并行加载 8 个张量]")

        # 清空 L2
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        thread_pool_results = cache_mgr.batch_fetch_parallel(test_keys, max_workers=8)
        thread_pool_time = (time.perf_counter() - start) * 1000

        thread_pool_hits = sum(1 for _, hit in thread_pool_results if hit)

        print(f"  线程池加载时间: {thread_pool_time:.2f} ms")
        print(f"  命中数: {thread_pool_hits}/{batch_size}")
        print(f"  平均每个: {thread_pool_time / batch_size:.2f} ms")

        # 与串行对比
        thread_pool_speedup = serial_time / thread_pool_time if thread_pool_time > 0 else 0
        print(f"  vs 串行加速比: {thread_pool_speedup:.2f}x")
        print(f"  验收标准: {'✅ 通过' if thread_pool_speedup >= 10 else '⚠️ 未达标'}  (目标 ≥ 10x)\n")

        # 测试 4: 更大批量（16 个张量）
        print("[测试 4: 线程池并行加载 16 个张量]")

        # 清空 L2
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        large_batch_results = cache_mgr.batch_fetch_parallel(keys, max_workers=16)
        large_batch_time = (time.perf_counter() - start) * 1000

        print(f"  线程池加载时间: {large_batch_time:.2f} ms")
        print(f"  命中数: {sum(1 for _, hit in large_batch_results if hit)}/{num_tensors}")
        print(f"  平均每个: {large_batch_time / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors / (large_batch_time / 1000):.1f} tensors/s\n")

    print("=== 基准测试完成 ===")


if __name__ == "__main__":
    asyncio.run(benchmark_async_io())
