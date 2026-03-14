"""多进程并行加载基准测试

验证 ProcessPoolExecutor 能否绕过 Python GIL 限制，实现真正并行。

预期结果：
- 多进程加速比 > 3x（vs 串行）
- 数据正确性验证通过
"""
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.thunder_config import SerializationConfig


def benchmark_multiprocess():
    """测试多进程并行加载（绕过 GIL）"""
    print("=== 多进程并行加载基准测试 ===\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建缓存管理器
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

        # 测试 2: 多进程并行加载（4 进程）
        print("[测试 2: 多进程并行加载（4 进程）]")

        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        mp4_results = cache_mgr.batch_fetch_multiprocess(keys, max_workers=4)
        mp4_time = (time.perf_counter() - start) * 1000

        print(f"  并行加载时间: {mp4_time:.2f} ms")
        print(f"  平均每个: {mp4_time / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (mp4_time / 1000):.1f} MB/s")

        # 性能对比
        speedup_4 = serial_time / mp4_time if mp4_time > 0 else 0
        print(f"  加速比: {speedup_4:.2f}x")
        print(f"  验收标准: {'✅ 通过' if speedup_4 >= 3 else '⚠️ 未达标'}  (目标 ≥ 3x)\n")

        # 测试 3: 多进程并行加载（8 进程）
        print("[测试 3: 多进程并行加载（8 进程）]")

        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        start = time.perf_counter()
        mp8_results = cache_mgr.batch_fetch_multiprocess(keys, max_workers=8)
        mp8_time = (time.perf_counter() - start) * 1000

        print(f"  并行加载时间: {mp8_time:.2f} ms")
        print(f"  平均每个: {mp8_time / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (mp8_time / 1000):.1f} MB/s")

        # 性能对比
        speedup_8 = serial_time / mp8_time if mp8_time > 0 else 0
        print(f"  加速比: {speedup_8:.2f}x")
        print(f"  验收标准: {'✅ 通过' if speedup_8 >= 3 else '⚠️ 未达标'}  (目标 ≥ 3x)\n")

        # 验证数据正确性
        all_correct_4 = all(
            mx.allclose(tensors[i], mp4_results[i][0])
            for i in range(num_tensors)
            if mp4_results[i][0] is not None
        )
        all_correct_8 = all(
            mx.allclose(tensors[i], mp8_results[i][0])
            for i in range(num_tensors)
            if mp8_results[i][0] is not None
        )
        print(f"  数据正确性（4 进程）: {'✅' if all_correct_4 else '❌'}")
        print(f"  数据正确性（8 进程）: {'✅' if all_correct_8 else '❌'}\n")

        # 性能总结
        print("=== 性能总结 ===")
        print(f"  串行加载:     {serial_time:.2f} ms")
        print(f"  多进程(4核):  {mp4_time:.2f} ms  (加速 {speedup_4:.2f}x)")
        print(f"  多进程(8核):  {mp8_time:.2f} ms  (加速 {speedup_8:.2f}x)")
        print()

        # 结论
        best_speedup = max(speedup_4, speedup_8)
        if best_speedup >= 3:
            print(f"✅ 多进程成功绕过 GIL，最高加速 {best_speedup:.2f}x")
            print("   推荐使用多进程方案替代 asyncio/threading")
        elif best_speedup >= 2:
            print(f"⚠️ 多进程有一定加速（{best_speedup:.2f}x），但未达预期")
            print("   可能原因：进程间通信开销、序列化开销")
            print("   建议：尝试 C++ 扩展方案")
        else:
            print(f"❌ 多进程加速不明显（{best_speedup:.2f}x）")
            print("   可能原因：进程启动开销过大、序列化开销过大")
            print("   建议：必须使用 C++ 扩展方案")

    print("\n=== 多进程测试完成 ===")


if __name__ == "__main__":
    benchmark_multiprocess()
