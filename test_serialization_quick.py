"""快速验证张量序列化功能"""
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from omlx.serialization import TensorSerializer
from omlx.thunder_config import SerializationConfig


def quick_test():
    """快速功能测试"""
    print("=== ThunderOMLX 张量序列化快速测试 ===\n")

    # 创建测试张量（4MB）
    tensor = mx.random.normal(shape=(1024, 1024))  # ~4MB (float32)
    print(f"张量大小: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"数据量: ~4MB\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 测试 1: 无压缩 + Checksum
        print("[测试 1: 无压缩 + Checksum]")
        config = SerializationConfig(compression="none", enable_checksum=True)
        serializer = TensorSerializer(config)
        file_path = tmpdir / "test_none"

        start = time.perf_counter()
        metadata = serializer.save(tensor, file_path)
        save_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        loaded = serializer.load(file_path)
        load_time = (time.perf_counter() - start) * 1000

        data_size = Path(str(file_path) + ".npy").stat().st_size / (1024 * 1024)

        print(f"  保存时间: {save_time:.2f} ms")
        print(f"  加载时间: {load_time:.2f} ms")
        print(f"  文件大小: {data_size:.2f} MB")
        print(f"  数据正确: {'✅' if mx.allclose(tensor, loaded) else '❌'}")
        print(f"  验收标准: {'✅ 通过' if save_time < 100 and load_time < 100 else '⚠️ 超时'}\n")

        # 测试 2: zlib 压缩
        print("[测试 2: zlib 压缩 + Checksum]")
        config = SerializationConfig(compression="zlib", enable_checksum=True)
        serializer = TensorSerializer(config)
        file_path = tmpdir / "test_zlib"

        start = time.perf_counter()
        metadata = serializer.save(tensor, file_path)
        save_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        loaded = serializer.load(file_path)
        load_time = (time.perf_counter() - start) * 1000

        data_size = Path(str(file_path) + ".npz").stat().st_size / (1024 * 1024)
        compression_ratio = 4.0 / data_size

        print(f"  保存时间: {save_time:.2f} ms")
        print(f"  加载时间: {load_time:.2f} ms")
        print(f"  文件大小: {data_size:.2f} MB")
        print(f"  压缩比: {compression_ratio:.2f}x")
        print(f"  数据正确: {'✅' if mx.allclose(tensor, loaded) else '❌'}")
        print(f"  验收标准: {'✅ 通过' if compression_ratio > 1.2 else '⚠️ 压缩不足'}\n")

        # 测试 3: 元数据
        print("[测试 3: 元数据读取]")
        meta = serializer.get_metadata(file_path)
        print(f"  Shape: {meta.shape}")
        print(f"  DType: {meta.dtype}")
        print(f"  Checksum: {meta.checksum[:16]}...")
        print(f"  Compression: {meta.compression}")
        print(f"  验收标准: {'✅ 通过' if meta.shape == (1024, 1024) else '❌ 失败'}\n")

    print("=== 快速测试完成 ===")


if __name__ == "__main__":
    quick_test()
