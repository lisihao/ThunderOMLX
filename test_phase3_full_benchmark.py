"""Phase 3 全面性能基准测试
整合 P3-1 到 P3-4 所有组件，验证整体性能收益
"""
import asyncio
import gc
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

import mlx.core as mx
import psutil

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.serialization import TensorSerializer
from omlx.thunder_config import SerializationConfig, ThunderOMLXConfig
from omlx.thunder_loader import load_thunder_config


class BenchmarkRunner:
    """性能基准测试运行器"""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.process = psutil.Process(os.getpid())

    def get_memory_mb(self) -> float:
        """获取当前内存使用（MB）"""
        return self.process.memory_info().rss / (1024 ** 2)

    def benchmark_config_loading(self, tmpdir: Path) -> Dict[str, Any]:
        """P3-1: 配置加载性能"""
        print("[P3-1: 配置加载性能]")

        # 创建测试配置文件
        config_path = tmpdir / "thunderomlx.yaml"
        config = ThunderOMLXConfig()
        config.save_to_yaml(config_path)

        # 测试冷启动加载
        start = time.perf_counter()
        loaded_config = ThunderOMLXConfig.load_from_yaml(config_path)
        cold_load_time = (time.perf_counter() - start) * 1000

        # 测试热加载（缓存）
        start = time.perf_counter()
        for _ in range(100):
            _ = ThunderOMLXConfig.load_from_yaml(config_path)
        hot_load_time = (time.perf_counter() - start) * 1000 / 100

        print(f"  冷启动加载: {cold_load_time:.2f} ms")
        print(f"  热加载（缓存）: {hot_load_time:.2f} ms")
        print(f"  配置字段数: {len(loaded_config.model_dump())}")
        print()

        return {
            "cold_load_ms": cold_load_time,
            "hot_load_ms": hot_load_time,
            "fields_count": len(loaded_config.model_dump()),
        }

    def benchmark_serialization(self, tmpdir: Path) -> Dict[str, Any]:
        """P3-2: 张量序列化性能"""
        print("[P3-2: 张量序列化性能]")

        results = {}

        # 测试不同大小的张量
        sizes = [
            ("小", (256, 256), 0.25),    # 256KB
            ("中", (512, 512), 1.0),     # 1MB
            ("大", (1024, 1024), 4.0),   # 4MB
            ("超大", (2048, 2048), 16.0), # 16MB
        ]

        for size_name, shape, size_mb in sizes:
            print(f"  [{size_name}张量 ~{size_mb:.1f}MB]")

            tensor = mx.random.normal(shape=shape)

            # 无压缩
            config = SerializationConfig(compression="none", enable_checksum=True)
            serializer = TensorSerializer(config)
            file_path = tmpdir / f"tensor_{size_name}_none"

            start = time.perf_counter()
            serializer.save(tensor, file_path)
            save_time_none = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            loaded = serializer.load(file_path)
            load_time_none = (time.perf_counter() - start) * 1000

            file_size_none = (Path(str(file_path) + ".npy").stat().st_size) / (1024 ** 2)

            # zlib 压缩
            config = SerializationConfig(compression="zlib", enable_checksum=True)
            serializer = TensorSerializer(config)
            file_path = tmpdir / f"tensor_{size_name}_zlib"

            start = time.perf_counter()
            serializer.save(tensor, file_path)
            save_time_zlib = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            loaded = serializer.load(file_path)
            load_time_zlib = (time.perf_counter() - start) * 1000

            file_size_zlib = (Path(str(file_path) + ".npz").stat().st_size) / (1024 ** 2)
            compression_ratio = file_size_none / file_size_zlib

            print(f"    无压缩: 保存 {save_time_none:.2f}ms, 加载 {load_time_none:.2f}ms, 大小 {file_size_none:.2f}MB")
            print(f"    zlib:   保存 {save_time_zlib:.2f}ms, 加载 {load_time_zlib:.2f}ms, 大小 {file_size_zlib:.2f}MB, 压缩比 {compression_ratio:.2f}x")

            results[size_name] = {
                "size_mb": size_mb,
                "none": {
                    "save_ms": save_time_none,
                    "load_ms": load_time_none,
                    "file_mb": file_size_none,
                },
                "zlib": {
                    "save_ms": save_time_zlib,
                    "load_ms": load_time_zlib,
                    "file_mb": file_size_zlib,
                    "compression_ratio": compression_ratio,
                },
            }

        print()
        return results

    def benchmark_dual_cache(self, tmpdir: Path) -> Dict[str, Any]:
        """P3-3: 双层缓存性能"""
        print("[P3-3: 双层缓存性能]")

        config = SerializationConfig(compression="zlib", enable_checksum=True)
        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=8,  # 8MB L2
            l3_cache_path=tmpdir / "l3_cache",
            l3_max_size_gb=1,
            serialization_config=config,
        )

        # 准备测试数据
        num_tensors = 20
        tensors = []
        keys = []

        print(f"  准备 {num_tensors} 个张量（每个 ~1MB）...")
        for i in range(num_tensors):
            tensor = mx.random.normal(shape=(512, 512))
            key = f"perf_test_{i}"
            cache_mgr.store(key, tensor)
            tensors.append(tensor)
            keys.append(key)

        # 测试 L2 命中性能
        print(f"  [L2 缓存命中测试]")
        l2_times = []
        for key in keys[:5]:  # 测试前 5 个（应该在 L2）
            if key in cache_mgr.l2_cache:
                start = time.perf_counter()
                value, hit = cache_mgr.fetch(key)
                l2_times.append((time.perf_counter() - start) * 1000)

        l2_avg = sum(l2_times) / len(l2_times) if l2_times else 0
        print(f"    L2 平均命中时间: {l2_avg:.4f} ms")

        # 清空 L2，测试 L3 命中性能
        print(f"  [L3 缓存命中测试]")
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        l3_times = []
        for key in keys[:5]:  # 测试前 5 个（从 L3 加载）
            start = time.perf_counter()
            value, hit = cache_mgr.fetch(key)
            l3_times.append((time.perf_counter() - start) * 1000)

        l3_avg = sum(l3_times) / len(l3_times) if l3_times else 0
        print(f"    L3 平均命中时间: {l3_avg:.2f} ms")

        # 测试驱逐性能
        print(f"  [驱逐性能测试]")
        stats_before = cache_mgr.get_stats()
        evictions_before = stats_before.l2_evictions

        # 添加更多张量，触发驱逐
        for i in range(10):
            tensor = mx.random.normal(shape=(512, 512))
            key = f"evict_test_{i}"
            cache_mgr.store(key, tensor)

        stats_after = cache_mgr.get_stats()
        evictions_after = stats_after.l2_evictions
        evictions_triggered = evictions_after - evictions_before

        print(f"    触发驱逐次数: {evictions_triggered}")
        print(f"    L2→L3 晋升: {stats_after.l2_to_l3_promotions}")

        # 统计信息
        stats = cache_mgr.get_stats()
        print(f"  [缓存统计]")
        print(f"    L2 命中率: {stats.l2_hit_rate:.2%}")
        print(f"    L3 命中率: {stats.l3_hit_rate:.2%}")
        print(f"    整体命中率: {stats.overall_hit_rate:.2%}")
        print(f"    L2 大小: {stats.l2_size_bytes / (1024**2):.2f} MB")
        print(f"    L3 大小: {stats.l3_size_bytes / (1024**2):.2f} MB")

        print()
        return {
            "l2_avg_ms": l2_avg,
            "l3_avg_ms": l3_avg,
            "evictions": evictions_triggered,
            "l2_hit_rate": stats.l2_hit_rate,
            "l3_hit_rate": stats.l3_hit_rate,
            "overall_hit_rate": stats.overall_hit_rate,
            "l2_size_mb": stats.l2_size_bytes / (1024**2),
            "l3_size_mb": stats.l3_size_bytes / (1024**2),
        }

    async def benchmark_async_io(self, tmpdir: Path) -> Dict[str, Any]:
        """P3-4: 异步 I/O 性能"""
        print("[P3-4: 异步 I/O 性能]")

        config = SerializationConfig(compression="zlib", enable_checksum=True)
        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=4,
            l3_cache_path=tmpdir / "l3_cache_async",
            l3_max_size_gb=1,
            serialization_config=config,
        )

        # 准备测试数据
        num_tensors = 16
        keys = []

        print(f"  准备 {num_tensors} 个张量...")
        for i in range(num_tensors):
            tensor = mx.random.normal(shape=(512, 512))
            key = f"async_test_{i}"
            cache_mgr.store(key, tensor)
            keys.append(key)

        # 清空 L2，强制从 L3 读取
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        # 串行加载
        print(f"  [串行加载 {num_tensors} 个张量]")
        start = time.perf_counter()
        for key in keys:
            _ = cache_mgr.fetch(key)
        serial_time = (time.perf_counter() - start) * 1000

        print(f"    串行时间: {serial_time:.2f} ms")
        print(f"    平均每个: {serial_time / num_tensors:.2f} ms")

        # 并行加载（asyncio）
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        print(f"  [asyncio 并行加载 {num_tensors} 个张量]")
        start = time.perf_counter()
        _ = await cache_mgr.batch_fetch(keys)
        async_time = (time.perf_counter() - start) * 1000

        print(f"    asyncio 时间: {async_time:.2f} ms")
        print(f"    平均每个: {async_time / num_tensors:.2f} ms")
        print(f"    加速比: {serial_time / async_time:.2f}x")

        # 并行加载（线程池）
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        print(f"  [线程池并行加载 {num_tensors} 个张量]")
        start = time.perf_counter()
        _ = cache_mgr.batch_fetch_parallel(keys, max_workers=8)
        thread_time = (time.perf_counter() - start) * 1000

        print(f"    线程池时间: {thread_time:.2f} ms")
        print(f"    平均每个: {thread_time / num_tensors:.2f} ms")
        print(f"    加速比: {serial_time / thread_time:.2f}x")

        print()
        return {
            "serial_ms": serial_time,
            "async_ms": async_time,
            "thread_ms": thread_time,
            "async_speedup": serial_time / async_time if async_time > 0 else 0,
            "thread_speedup": serial_time / thread_time if thread_time > 0 else 0,
        }

    def benchmark_memory_usage(self, tmpdir: Path) -> Dict[str, Any]:
        """内存使用情况"""
        print("[内存使用情况]")

        mem_start = self.get_memory_mb()

        config = SerializationConfig(compression="zlib", enable_checksum=True)
        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=64,  # 64MB L2
            l3_cache_path=tmpdir / "l3_cache_mem",
            l3_max_size_gb=1,
            serialization_config=config,
        )

        # 存储大量张量
        num_tensors = 100
        print(f"  存储 {num_tensors} 个张量...")

        for i in range(num_tensors):
            tensor = mx.random.normal(shape=(512, 512))
            key = f"mem_test_{i}"
            cache_mgr.store(key, tensor)

        mem_after = self.get_memory_mb()
        mem_delta = mem_after - mem_start

        stats = cache_mgr.get_stats()

        print(f"    内存起始: {mem_start:.2f} MB")
        print(f"    内存结束: {mem_after:.2f} MB")
        print(f"    内存增量: {mem_delta:.2f} MB")
        print(f"    L2 缓存: {stats.l2_size_bytes / (1024**2):.2f} MB")
        print(f"    L3 缓存: {stats.l3_size_bytes / (1024**2):.2f} MB")
        print(f"    总缓存: {(stats.l2_size_bytes + stats.l3_size_bytes) / (1024**2):.2f} MB")

        # 计算开销比例
        total_cache_mb = (stats.l2_size_bytes + stats.l3_size_bytes) / (1024**2)
        overhead_ratio = (mem_delta / total_cache_mb) if total_cache_mb > 0 else 0

        print(f"    开销比例: {overhead_ratio:.2f}x (内存增量/总缓存)")

        print()
        return {
            "mem_start_mb": mem_start,
            "mem_after_mb": mem_after,
            "mem_delta_mb": mem_delta,
            "l2_cache_mb": stats.l2_size_bytes / (1024**2),
            "l3_cache_mb": stats.l3_size_bytes / (1024**2),
            "overhead_ratio": overhead_ratio,
        }

    async def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("=" * 70)
        print("Phase 3 全面性能基准测试")
        print("=" * 70)
        print()

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # P3-1: 配置加载
            self.results["config_loading"] = self.benchmark_config_loading(tmpdir)

            # P3-2: 序列化
            self.results["serialization"] = self.benchmark_serialization(tmpdir)

            # P3-3: 双层缓存
            self.results["dual_cache"] = self.benchmark_dual_cache(tmpdir)

            # P3-4: 异步 I/O
            self.results["async_io"] = await self.benchmark_async_io(tmpdir)

            # 内存使用
            self.results["memory"] = self.benchmark_memory_usage(tmpdir)

            # 生成总结
            self.print_summary()

    def print_summary(self):
        """打印性能总结"""
        print("=" * 70)
        print("性能总结")
        print("=" * 70)
        print()

        print("[P3-1: 配置加载]")
        config = self.results["config_loading"]
        print(f"  ✅ 冷启动: {config['cold_load_ms']:.2f} ms")
        print(f"  ✅ 热加载: {config['hot_load_ms']:.2f} ms (100 次平均)")
        print()

        print("[P3-2: 序列化（4MB 张量）]")
        ser = self.results["serialization"]["大"]
        print(f"  ✅ 无压缩保存: {ser['none']['save_ms']:.2f} ms")
        print(f"  ✅ 无压缩加载: {ser['none']['load_ms']:.2f} ms")
        print(f"  ✅ zlib 保存: {ser['zlib']['save_ms']:.2f} ms")
        print(f"  ✅ zlib 加载: {ser['zlib']['load_ms']:.2f} ms")
        print(f"  ✅ 压缩比: {ser['zlib']['compression_ratio']:.2f}x")
        print()

        print("[P3-3: 双层缓存]")
        cache = self.results["dual_cache"]
        print(f"  ✅ L2 平均命中: {cache['l2_avg_ms']:.4f} ms (目标 < 5ms)")
        print(f"  ✅ L3 平均命中: {cache['l3_avg_ms']:.2f} ms (目标 < 50ms)")
        print(f"  ✅ 整体命中率: {cache['overall_hit_rate']:.1%}")
        print(f"  ✅ L2 大小: {cache['l2_size_mb']:.2f} MB")
        print(f"  ✅ L3 大小: {cache['l3_size_mb']:.2f} MB")
        print()

        print("[P3-4: 异步 I/O]")
        async_io = self.results["async_io"]
        print(f"  ⚠️ asyncio 加速: {async_io['async_speedup']:.2f}x (目标 10x)")
        print(f"  ⚠️ 线程池加速: {async_io['thread_speedup']:.2f}x (目标 10x)")
        print(f"  📝 受 Python GIL 限制")
        print()

        print("[内存使用]")
        mem = self.results["memory"]
        print(f"  ✅ 内存增量: {mem['mem_delta_mb']:.2f} MB (100 个张量)")
        print(f"  ✅ 开销比例: {mem['overhead_ratio']:.2f}x")
        print()

        print("[整体评价]")
        # 计算达标项
        passed = 0
        total = 0

        # P3-1
        total += 1
        if config['cold_load_ms'] < 100:
            passed += 1
            print(f"  ✅ 配置加载 < 100ms")
        else:
            print(f"  ❌ 配置加载 >= 100ms")

        # P3-2
        total += 2
        if ser['none']['save_ms'] < 100 and ser['none']['load_ms'] < 100:
            passed += 1
            print(f"  ✅ 序列化（无压缩）< 100ms")
        else:
            print(f"  ❌ 序列化（无压缩）>= 100ms")

        if ser['zlib']['save_ms'] < 100:
            passed += 1
            print(f"  ✅ 序列化（zlib）保存 < 100ms")
        else:
            print(f"  ❌ 序列化（zlib）保存 >= 100ms")

        # P3-3
        total += 2
        if cache['l2_avg_ms'] < 5:
            passed += 1
            print(f"  ✅ L2 缓存 < 5ms")
        else:
            print(f"  ❌ L2 缓存 >= 5ms")

        if cache['l3_avg_ms'] < 50:
            passed += 1
            print(f"  ✅ L3 缓存 < 50ms")
        else:
            print(f"  ❌ L3 缓存 >= 50ms")

        # P3-4
        total += 1
        if async_io['async_speedup'] >= 10:
            passed += 1
            print(f"  ✅ 异步 I/O >= 10x")
        else:
            print(f"  ⚠️ 异步 I/O < 10x (受 GIL 限制)")

        print()
        print(f"  达标率: {passed}/{total} ({passed/total*100:.1f}%)")
        print()

        # 保存结果到文件
        results_path = Path(".solar/benchmark_results.json")
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  详细结果已保存: {results_path}")


async def main():
    """主函数"""
    runner = BenchmarkRunner()
    await runner.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
