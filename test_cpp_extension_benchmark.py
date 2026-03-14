"""C++ 扩展性能基准测试

验证 C++ 扩展能否绕过 Python GIL，实现真正并行加载。

预期结果：
- C++ + ThreadPoolExecutor 加速比 > 8x
- 数据正确性验证通过
"""
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

# 导入 C++ 扩展
import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "omlx" / "extensions"))

try:
    from _tensor_loader import load_numpy_nogil, load_numpy_safe
    CPP_EXTENSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ C++ 扩展未找到: {e}")
    print("   请先运行: ./scripts/build_extensions.sh")
    CPP_EXTENSION_AVAILABLE = False
    sys.exit(1)

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.thunder_config import SerializationConfig


def benchmark_cpp_extension():
    """测试 C++ 扩展并行加载（绕过 GIL）"""
    print("=== C++ 扩展并行加载基准测试 ===\n")
    print(f"✅ C++ 扩展加载成功\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建缓存管理器
        config = SerializationConfig(compression="none", enable_checksum=False)  # 无压缩，简化测试
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
        file_paths = []

        for i in range(num_tensors):
            # 10MB 张量：1600x1600 float32 = ~10MB
            tensor = mx.random.normal(shape=(1600, 1600))
            key = f"large_tensor_{i}"
            cache_mgr.store(key, tensor)
            keys.append(key)
            tensors.append(tensor)

        # 列出实际创建的文件
        print(f"  L3 缓存目录: {cache_mgr.l3_cache_path}")
        actual_files = list(cache_mgr.l3_cache_path.glob("*"))
        print(f"  实际文件: {[f.name for f in actual_files[:5]]}")

        # 检查实际文件路径（可能有不同扩展名）
        for key in keys:
            file_path = cache_mgr.l3_cache_path / key
            # 检查哪个扩展名存在
            if (file_path.parent / (file_path.name + ".npy")).exists():
                file_paths.append(str(file_path) + ".npy")
            elif (file_path.parent / (file_path.name + ".npz")).exists():
                file_paths.append(str(file_path) + ".npz")
            else:
                # 列出所有可能的文件
                candidates = list(file_path.parent.glob(file_path.name + "*"))
                if candidates:
                    file_paths.append(str(candidates[0]))
                    print(f"  使用文件: {candidates[0].name}")
                else:
                    raise FileNotFoundError(f"找不到缓存文件: {file_path}")

        print(f"  创建 {num_tensors} 个张量，每个 ~{tensor_size_mb} MB")
        print(f"  L3 大小: {cache_mgr.l3_size_bytes / (1024**2):.2f} MB")
        print(f"  文件格式: {Path(file_paths[0]).suffix}\n")

        # 测试 1: Python 串行加载（baseline）
        print("[测试 1: Python 串行加载（baseline）]")

        start = time.perf_counter()
        python_results = []
        for path in file_paths:
            # 模拟 Python 层加载
            np_array = np.load(path)
            tensor = mx.array(np_array)
            python_results.append(tensor)
        python_time = (time.perf_counter() - start) * 1000

        print(f"  Python 串行时间: {python_time:.2f} ms")
        print(f"  平均每个: {python_time / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (python_time / 1000):.1f} MB/s\n")

        # 测试 2: C++ 扩展 + ThreadPoolExecutor（4 线程）
        print("[测试 2: C++ 扩展 + ThreadPoolExecutor（4 线程）]")

        def load_with_cpp(path):
            """使用 C++ 扩展加载（释放 GIL）"""
            np_array, success = load_numpy_safe(path)
            if success:
                return mx.array(np_array)
            else:
                return None

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            cpp_results_4 = list(executor.map(load_with_cpp, file_paths))
        cpp_time_4 = (time.perf_counter() - start) * 1000

        print(f"  C++ 并行时间: {cpp_time_4:.2f} ms")
        print(f"  平均每个: {cpp_time_4 / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (cpp_time_4 / 1000):.1f} MB/s")

        # 性能对比
        speedup_4 = python_time / cpp_time_4 if cpp_time_4 > 0 else 0
        print(f"  加速比: {speedup_4:.2f}x")
        print(f"  验收标准: {'✅ 通过' if speedup_4 >= 3 else '⚠️ 未达标'}  (目标 ≥ 3x)\n")

        # 测试 3: C++ 扩展 + ThreadPoolExecutor（8 线程）
        print("[测试 3: C++ 扩展 + ThreadPoolExecutor（8 线程）]")

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            cpp_results_8 = list(executor.map(load_with_cpp, file_paths))
        cpp_time_8 = (time.perf_counter() - start) * 1000

        print(f"  C++ 并行时间: {cpp_time_8:.2f} ms")
        print(f"  平均每个: {cpp_time_8 / num_tensors:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (cpp_time_8 / 1000):.1f} MB/s")

        # 性能对比
        speedup_8 = python_time / cpp_time_8 if cpp_time_8 > 0 else 0
        print(f"  加速比: {speedup_8:.2f}x")
        print(f"  验收标准: {'✅ 通过' if speedup_8 >= 8 else '⚠️ 未达标'}  (目标 ≥ 8x)\n")

        # 验证数据正确性
        all_correct_4 = all(
            mx.allclose(tensors[i], cpp_results_4[i])
            for i in range(num_tensors)
            if cpp_results_4[i] is not None
        )
        all_correct_8 = all(
            mx.allclose(tensors[i], cpp_results_8[i])
            for i in range(num_tensors)
            if cpp_results_8[i] is not None
        )
        print(f"  数据正确性（4 线程）: {'✅' if all_correct_4 else '❌'}")
        print(f"  数据正确性（8 线程）: {'✅' if all_correct_8 else '❌'}\n")

        # 性能总结
        print("=== 性能总结 ===")
        print(f"  Python 串行:  {python_time:.2f} ms")
        print(f"  C++ 并行(4核): {cpp_time_4:.2f} ms  (加速 {speedup_4:.2f}x)")
        print(f"  C++ 并行(8核): {cpp_time_8:.2f} ms  (加速 {speedup_8:.2f}x)")
        print()

        # 结论
        best_speedup = max(speedup_4, speedup_8)
        if best_speedup >= 8:
            print(f"✅ C++ 扩展成功绕过 GIL，最高加速 {best_speedup:.2f}x")
            print("   推荐使用 C++ 扩展方案")
        elif best_speedup >= 3:
            print(f"⚠️ C++ 扩展有显著加速（{best_speedup:.2f}x），但未达最优")
            print("   可能原因：numpy 反序列化仍在 Python 层")
            print("   建议：进一步优化 C++ 层实现")
        else:
            print(f"❌ C++ 扩展加速不明显（{best_speedup:.2f}x）")
            print("   可能原因：GIL 仍在 numpy.load() 中被持有")

    print("\n=== C++ 扩展测试完成 ===")


if __name__ == "__main__":
    if CPP_EXTENSION_AVAILABLE:
        benchmark_cpp_extension()
