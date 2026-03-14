"""C++ 扩展简单测试

直接测试 C++ 扩展的性能，不依赖缓存管理器。
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
    print("✅ C++ 扩展加载成功\n")
except ImportError as e:
    print(f"⚠️ C++ 扩展未找到: {e}")
    sys.exit(1)


def main():
    print("=== C++ 扩展性能测试 ===\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 准备测试数据
        print("[准备阶段]")
        num_tensors = 8
        tensor_size_mb = 10
        tensors = []
        file_paths = []

        for i in range(num_tensors):
            # 创建张量
            tensor = mx.random.normal(shape=(1600, 1600))
            tensors.append(tensor)

            # 保存为 .npy 文件
            file_path = tmpdir / f"tensor_{i}.npy"
            np_array = np.array(tensor)
            np.save(file_path, np_array)
            file_paths.append(str(file_path))

        print(f"  创建 {num_tensors} 个张量，每个 ~{tensor_size_mb} MB\n")

        # 测试 1: Python 串行加载
        print("[测试 1: Python 串行加载]")
        start = time.perf_counter()
        python_results = []
        for path in file_paths:
            np_array = np.load(path)
            tensor = mx.array(np_array)
            python_results.append(tensor)
        python_time = (time.perf_counter() - start) * 1000

        print(f"  时间: {python_time:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (python_time / 1000):.1f} MB/s\n")

        # 测试 2: C++ + 单线程
        print("[测试 2: C++ 单线程加载]")
        start = time.perf_counter()
        cpp_single_results = []
        for path in file_paths:
            np_array, success = load_numpy_safe(path)
            if success:
                tensor = mx.array(np_array)
                cpp_single_results.append(tensor)
        cpp_single_time = (time.perf_counter() - start) * 1000

        print(f"  时间: {cpp_single_time:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (cpp_single_time / 1000):.1f} MB/s")
        print(f"  vs Python: {python_time / cpp_single_time:.2f}x\n")

        # 测试 3: C++ + ThreadPoolExecutor (4 线程)
        print("[测试 3: C++ + ThreadPoolExecutor (4 线程)]")

        def load_with_cpp(path):
            np_array, success = load_numpy_safe(path)
            if success:
                return mx.array(np_array)
            return None

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            cpp_results_4 = list(executor.map(load_with_cpp, file_paths))
        cpp_time_4 = (time.perf_counter() - start) * 1000

        print(f"  时间: {cpp_time_4:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (cpp_time_4 / 1000):.1f} MB/s")
        speedup_4 = python_time / cpp_time_4
        print(f"  vs Python: {speedup_4:.2f}x")
        print(f"  验收: {'✅ 通过' if speedup_4 >= 3 else '⚠️ 未达标'} (目标 ≥ 3x)\n")

        # 测试 4: C++ + ThreadPoolExecutor (8 线程)
        print("[测试 4: C++ + ThreadPoolExecutor (8 线程)]")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            cpp_results_8 = list(executor.map(load_with_cpp, file_paths))
        cpp_time_8 = (time.perf_counter() - start) * 1000

        print(f"  时间: {cpp_time_8:.2f} ms")
        print(f"  吞吐量: {num_tensors * tensor_size_mb / (cpp_time_8 / 1000):.1f} MB/s")
        speedup_8 = python_time / cpp_time_8
        print(f"  vs Python: {speedup_8:.2f}x")
        print(f"  验收: {'✅ 通过' if speedup_8 >= 8 else '⚠️ 未达标'} (目标 ≥ 8x)\n")

        # 验证数据正确性
        all_correct_4 = all(mx.allclose(tensors[i], cpp_results_4[i]) for i in range(num_tensors))
        all_correct_8 = all(mx.allclose(tensors[i], cpp_results_8[i]) for i in range(num_tensors))

        print(f"数据正确性: 4线程 {'✅' if all_correct_4 else '❌'} | 8线程 {'✅' if all_correct_8 else '❌'}\n")

        # 总结
        print("=== 性能总结 ===")
        print(f"Python 串行:    {python_time:.2f} ms")
        print(f"C++ 单线程:     {cpp_single_time:.2f} ms  ({python_time/cpp_single_time:.2f}x)")
        print(f"C++ 4线程:      {cpp_time_4:.2f} ms  ({speedup_4:.2f}x)")
        print(f"C++ 8线程:      {cpp_time_8:.2f} ms  ({speedup_8:.2f}x)")
        print()

        best_speedup = max(speedup_4, speedup_8)
        if best_speedup >= 8:
            print(f"✅ C++ 扩展成功！加速 {best_speedup:.2f}x，GIL 已绕过")
        elif best_speedup >= 3:
            print(f"⚠️ C++ 扩展部分成功（{best_speedup:.2f}x），但未达最优")
            print("   可能受限于 numpy.load() 仍需 GIL")
        else:
            print(f"❌ C++ 扩展未达预期（{best_speedup:.2f}x）")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
