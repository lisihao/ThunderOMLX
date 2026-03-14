"""快速测试 lz4 压缩功能"""
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np

from omlx.serialization import TensorSerializer
from omlx.thunder_config import SerializationConfig


def test_lz4_compression():
    """测试 lz4 压缩的保存和加载"""
    print("=" * 60)
    print("测试 lz4 压缩功能")
    print("=" * 60)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建配置
        config = SerializationConfig(
            compression="lz4",
            enable_checksum=True,
            checksum_algorithm="xxh64"
        )

        serializer = TensorSerializer(config)

        # 生成测试张量（16MB，模拟 KV Cache）
        shape = (1, 1024, 4096)  # batch=1, seq_len=1024, hidden_dim=4096
        print(f"\n生成测试张量: shape={shape}")
        tensor = mx.random.normal(shape=shape)
        size_mb = tensor.nbytes / (1024 ** 2)
        print(f"张量大小: {size_mb:.2f} MB")

        # 测试保存
        file_path = tmpdir / "test_tensor"
        print(f"\n保存张量到: {file_path}")

        start = time.perf_counter()
        metadata = serializer.save(tensor, file_path)
        save_time = (time.perf_counter() - start) * 1000
        print(f"保存时间: {save_time:.2f} ms")

        # 检查文件
        data_file = file_path.with_suffix(".lz4")
        meta_file = file_path.with_suffix(".meta.json")

        print(f"\n生成文件:")
        print(f"  数据文件: {data_file.name} ({data_file.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  元数据文件: {meta_file.name}")

        # 计算压缩比
        compressed_size_mb = data_file.stat().st_size / (1024 ** 2)
        compression_ratio = size_mb / compressed_size_mb
        print(f"\n压缩比: {compression_ratio:.2f}x")

        # 测试加载
        print(f"\n加载张量...")
        start = time.perf_counter()
        loaded_tensor = serializer.load(file_path)
        load_time = (time.perf_counter() - start) * 1000
        print(f"加载时间: {load_time:.2f} ms")

        # 验证一致性
        np_original = np.array(tensor)
        np_loaded = np.array(loaded_tensor)

        if np.allclose(np_original, np_loaded):
            print("\n✅ 数据一致性验证通过")
        else:
            print("\n❌ 数据不一致！")
            return False

        # 性能总结
        print("\n" + "=" * 60)
        print("性能总结")
        print("=" * 60)
        print(f"张量大小: {size_mb:.2f} MB")
        print(f"压缩后大小: {compressed_size_mb:.2f} MB")
        print(f"压缩比: {compression_ratio:.2f}x")
        print(f"保存时间: {save_time:.2f} ms")
        print(f"加载时间: {load_time:.2f} ms")
        print(f"吞吐量 (保存): {size_mb / (save_time / 1000):.1f} MB/s")
        print(f"吞吐量 (加载): {size_mb / (load_time / 1000):.1f} MB/s")

        return True


def compare_compression_methods():
    """对比 zlib vs lz4 性能"""
    print("\n" + "=" * 60)
    print("对比 zlib vs lz4 压缩性能")
    print("=" * 60)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 生成测试张量（16MB）
        shape = (1, 1024, 4096)
        tensor = mx.random.normal(shape=shape)
        size_mb = tensor.nbytes / (1024 ** 2)

        results = {}

        for compression in ["zlib", "lz4"]:
            print(f"\n【{compression} 压缩】")

            config = SerializationConfig(
                compression=compression,
                enable_checksum=True
            )
            serializer = TensorSerializer(config)

            file_path = tmpdir / f"test_{compression}"

            # 保存
            start = time.perf_counter()
            serializer.save(tensor, file_path)
            save_time = (time.perf_counter() - start) * 1000

            # 加载
            start = time.perf_counter()
            serializer.load(file_path)
            load_time = (time.perf_counter() - start) * 1000

            # 文件大小
            if compression == "zlib":
                data_file = file_path.with_suffix(".npz")
            else:
                data_file = file_path.with_suffix(".lz4")

            compressed_size = data_file.stat().st_size / (1024 ** 2)

            results[compression] = {
                "save_time": save_time,
                "load_time": load_time,
                "compressed_size": compressed_size,
                "compression_ratio": size_mb / compressed_size
            }

            print(f"  保存时间: {save_time:.2f} ms")
            print(f"  加载时间: {load_time:.2f} ms")
            print(f"  压缩后大小: {compressed_size:.2f} MB")
            print(f"  压缩比: {results[compression]['compression_ratio']:.2f}x")

        # 性能对比
        print("\n" + "=" * 60)
        print("性能对比总结")
        print("=" * 60)

        zlib_result = results["zlib"]
        lz4_result = results["lz4"]

        save_speedup = zlib_result["save_time"] / lz4_result["save_time"]
        load_speedup = zlib_result["load_time"] / lz4_result["load_time"]

        print(f"\n保存性能:")
        print(f"  zlib: {zlib_result['save_time']:.2f} ms")
        print(f"  lz4:  {lz4_result['save_time']:.2f} ms")
        print(f"  加速比: {save_speedup:.2f}x ✅")

        print(f"\n加载性能:")
        print(f"  zlib: {zlib_result['load_time']:.2f} ms")
        print(f"  lz4:  {lz4_result['load_time']:.2f} ms")
        print(f"  加速比: {load_speedup:.2f}x ✅")

        print(f"\n压缩比:")
        print(f"  zlib: {zlib_result['compression_ratio']:.2f}x")
        print(f"  lz4:  {lz4_result['compression_ratio']:.2f}x")

        print(f"\n总结:")
        if save_speedup > 2 and load_speedup > 2:
            print(f"  ✅ lz4 在保存和加载上均有 2x+ 提升")
        else:
            print(f"  ⚠️ lz4 提升未达预期 2x 目标")


if __name__ == "__main__":
    # 基础功能测试
    if test_lz4_compression():
        print("\n✅ lz4 压缩功能测试通过\n")

        # 性能对比测试
        compare_compression_methods()
    else:
        print("\n❌ lz4 压缩功能测试失败")
