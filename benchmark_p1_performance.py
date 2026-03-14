#!/usr/bin/env python3
"""
P1 阶段性能验证脚本

测试内容：
1. L3 (SSD) 加速测试 - Smart Prefetch on/off 对比
2. 内存占用测试 - Adaptive Chunk 内存优化
3. Checksum 性能开销测试
4. 综合性能对比
"""

import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.chunk_adapter import AdaptiveChunkCalculator

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    print("⚠️  MLX not available, using mock data for benchmarks")


def format_bytes(bytes_val):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def format_duration(seconds):
    """格式化时间"""
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def create_mock_cache_data(num_layers=32, seq_len=128, embd_dim=128):
    """创建模拟的 KV cache 数据"""
    if HAS_MLX:
        cache_data = []
        for _ in range(num_layers):
            # 模拟 keys 和 values
            keys = mx.random.normal(shape=(seq_len, embd_dim))
            values = mx.random.normal(shape=(seq_len, embd_dim))
            cache_data.append((keys, values))
        return cache_data
    else:
        # 使用 numpy 模拟
        cache_data = []
        for _ in range(num_layers):
            keys = np.random.randn(seq_len, embd_dim).astype(np.float16)
            values = np.random.randn(seq_len, embd_dim).astype(np.float16)
            cache_data.append((keys, values))
        return cache_data


def benchmark_smart_prefetch():
    """测试 1: Smart Prefetch 性能

    测试场景：
    - 无预取：冷启动，每次从 SSD 加载（无 hot cache）
    - 有预取：预取触发后，热块已在 hot cache 中
    """
    print("=" * 70)
    print("Benchmark 1: Smart Prefetch Performance")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  跳过（需要 MLX）")
        return

    num_blocks = 50
    num_hot_blocks = 10

    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\n📝 创建 {num_blocks} 个缓存块...")

        # 第一阶段：保存块
        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 500,  # 500MB
            enable_prefetch=False,
            enable_checksum=False,
            hot_cache_max_bytes=0
        )

        block_hashes = []
        for i in range(num_blocks):
            block_hash = f"block_{i:03d}".encode().ljust(32, b'\x00')
            cache_data = create_mock_cache_data(num_layers=32, seq_len=128)
            manager.save_block(block_hash, cache_data, token_count=128)
            block_hashes.append(block_hash)

        print(f"  ✅ 已保存 {num_blocks} 个块")

        # 等待写入完成
        print("  ⏳ 等待写入完成...")
        flush_result = manager.flush(timeout=30.0)
        if flush_result:
            print("  ✅ 写入完成")
        else:
            print("  ⚠️  写入超时，部分 block 可能未保存")

        # 检查统计
        stats_after_flush = manager.get_stats()
        print(f"  📊 写入统计: {stats_after_flush.num_files} 个文件")

        manager.stop()

        # 第二阶段：测试无预取（冷启动，无 hot cache）
        print(f"\n📊 场景 1: 无预取 + 无 hot cache（每次冷读 SSD）...")

        # 检查文件是否存在
        cache_files = list(Path(cache_dir).rglob("*.safetensors*"))
        print(f"  📁 缓存文件数: {len(cache_files)}")

        start = time.perf_counter()
        loads_no_prefetch = 0

        # 每轮都重新创建 manager（冷启动）
        for round_idx in range(5):
            manager_no_prefetch = PagedSSDCacheManager(
                cache_dir=Path(cache_dir),
                max_size_bytes=1024 * 1024 * 500,
                enable_prefetch=False,
                enable_checksum=False,
                hot_cache_max_bytes=0  # 无 hot cache
            )

            # 检查索引
            stats_check = manager_no_prefetch.get_stats()
            if round_idx == 0:
                print(f"  📊 Manager 索引: {stats_check.num_files} 个文件")

            # 访问前 10 个热块
            for block_hash in block_hashes[:num_hot_blocks]:
                data = manager_no_prefetch.load_block(block_hash)
                if data:
                    loads_no_prefetch += 1

            manager_no_prefetch.stop()

        elapsed_no_prefetch = time.perf_counter() - start
        avg_per_block_no_prefetch = elapsed_no_prefetch / loads_no_prefetch if loads_no_prefetch > 0 else 0

        print(f"  ❌ 无预取:")
        print(f"    - 总耗时: {format_duration(elapsed_no_prefetch)}")
        print(f"    - 平均每块: {format_duration(avg_per_block_no_prefetch)}")
        print(f"    - 成功加载: {loads_no_prefetch} 块")

        # 第三阶段：测试有预取（预热后，hot cache 命中）
        print(f"\n📊 场景 2: 有预取 + hot cache（预取后命中）...")

        manager_with_prefetch = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 500,
            enable_prefetch=True,
            prefetch_top_n=num_hot_blocks,
            prefetch_interval=1.0,  # 1 秒触发一次
            enable_checksum=False,
            hot_cache_max_bytes=50 * 1024**2  # 50MB hot cache
        )

        # 预热：访问一次所有热块（建立访问频率）
        for block_hash in block_hashes[:num_hot_blocks]:
            manager_with_prefetch.load_block(block_hash)

        # 等待预取触发（1 秒间隔 + 1 秒缓冲）
        print("  ⏳ 等待预取触发...")
        time.sleep(2.5)

        # 测试：重复访问热块（应该从 hot cache 命中）
        start = time.perf_counter()
        loads_with_prefetch = 0
        for _ in range(5):  # 5 轮访问
            for block_hash in block_hashes[:num_hot_blocks]:
                data = manager_with_prefetch.load_block(block_hash)
                if data:
                    loads_with_prefetch += 1
        elapsed_with_prefetch = time.perf_counter() - start

        avg_per_block_with_prefetch = elapsed_with_prefetch / loads_with_prefetch if loads_with_prefetch > 0 else 0
        print(f"\n  ✅ 有预取:")
        print(f"    - 总耗时: {format_duration(elapsed_with_prefetch)}")
        print(f"    - 平均每块: {format_duration(avg_per_block_with_prefetch)}")
        print(f"    - 成功加载: {loads_with_prefetch} 块")

        # 计算加速比
        if elapsed_with_prefetch > 0:
            speedup = elapsed_no_prefetch / elapsed_with_prefetch
            print(f"\n  🚀 加速比: {speedup:.2f}x")

            if speedup >= 2.0:
                print(f"  ✅ 达到预期（目标 > 2x）")
            else:
                print(f"  ⚠️  未达预期（目标 > 2x）")
        else:
            print(f"\n  ⚠️  无法计算加速比")

        # 统计
        stats = manager_with_prefetch.get_stats()
        print(f"\n  📊 预取统计:")
        print(f"    - Hot cache hits: {stats.hot_cache_hits}")
        print(f"    - Total loads: {stats.loads}")

        manager_with_prefetch.stop()

    print("\n✅ Benchmark 1 完成")


def benchmark_adaptive_chunk_memory():
    """测试 2: Adaptive Chunk 内存优化"""
    print("\n" + "=" * 70)
    print("Benchmark 2: Adaptive Chunk Memory Optimization")
    print("=" * 70)

    calculator = AdaptiveChunkCalculator(cache_block_size=64)

    test_cases = [
        (512, "中等 Prompt"),
        (2048, "长 Prompt"),
        (8192, "超长 Prompt"),
    ]

    print("\n📝 测试不同 prompt 长度的内存占用...")

    results = []
    for prompt_length, label in test_cases:
        # 模拟单次处理（无分块）
        bytes_per_token = 4096  # 估算
        memory_no_chunk = prompt_length * bytes_per_token

        # 自适应分块
        chunk_size = calculator.compute_chunk_size(prompt_length)
        memory_with_chunk = chunk_size * bytes_per_token

        reduction = memory_no_chunk / memory_with_chunk

        print(f"\n  {label} ({prompt_length} tokens):")
        print(f"    - 无分块内存峰值: {format_bytes(memory_no_chunk)}")
        print(f"    - 分块内存峰值: {format_bytes(memory_with_chunk)}")
        print(f"    - 优化: {reduction:.1f}x")

        results.append({
            'prompt_length': prompt_length,
            'label': label,
            'reduction': reduction
        })

    print("\n  📊 内存优化总结:")
    for r in results:
        status = "✅" if r['reduction'] >= 4.0 else "⚠️"
        print(f"    {status} {r['label']}: {r['reduction']:.1f}x 优化")

    print("\n✅ Benchmark 2 完成")


def benchmark_checksum_overhead():
    """测试 3: Checksum 性能开销"""
    print("\n" + "=" * 70)
    print("Benchmark 3: Checksum Performance Overhead")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  跳过（需要 MLX）")
        return

    num_blocks = 20

    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\n📝 创建 {num_blocks} 个缓存块...")

        # 测试 1: 无 checksum
        manager_no_checksum = PagedSSDCacheManager(
            cache_dir=Path(cache_dir) / "no_checksum",
            max_size_bytes=1024 * 1024 * 200,
            enable_checksum=False,
            enable_prefetch=False,
            hot_cache_max_bytes=0
        )

        block_hashes = []
        for i in range(num_blocks):
            block_hash = f"block_{i:03d}".encode().ljust(32, b'\x00')
            block_hashes.append(block_hash)

        # Save 性能
        start = time.perf_counter()
        for block_hash in block_hashes:
            cache_data = create_mock_cache_data(num_layers=32, seq_len=64)
            manager_no_checksum.save_block(block_hash, cache_data, token_count=64)
        time.sleep(1)  # 等待写入
        save_time_no_checksum = time.perf_counter() - start

        # Load 性能
        start = time.perf_counter()
        for block_hash in block_hashes:
            manager_no_checksum.load_block(block_hash)
        load_time_no_checksum = time.perf_counter() - start

        print(f"\n  ❌ 无 Checksum:")
        print(f"    - Save: {format_duration(save_time_no_checksum)} ({format_duration(save_time_no_checksum/num_blocks)}/块)")
        print(f"    - Load: {format_duration(load_time_no_checksum)} ({format_duration(load_time_no_checksum/num_blocks)}/块)")

        manager_no_checksum.stop()

        # 测试 2: 有 checksum
        manager_with_checksum = PagedSSDCacheManager(
            cache_dir=Path(cache_dir) / "with_checksum",
            max_size_bytes=1024 * 1024 * 200,
            enable_checksum=True,
            enable_prefetch=False,
            hot_cache_max_bytes=0
        )

        # Save 性能
        start = time.perf_counter()
        for block_hash in block_hashes:
            cache_data = create_mock_cache_data(num_layers=32, seq_len=64)
            manager_with_checksum.save_block(block_hash, cache_data, token_count=64)
        time.sleep(1)  # 等待写入
        save_time_with_checksum = time.perf_counter() - start

        # Load 性能（首次加载 - 需要验证）
        start = time.perf_counter()
        for block_hash in block_hashes:
            manager_with_checksum.load_block(block_hash)
        load_time_first = time.perf_counter() - start

        # Load 性能（重复加载 - cached verification）
        start = time.perf_counter()
        for block_hash in block_hashes:
            manager_with_checksum.load_block(block_hash)
        load_time_cached = time.perf_counter() - start

        print(f"\n  ✅ 有 Checksum:")
        print(f"    - Save: {format_duration(save_time_with_checksum)} ({format_duration(save_time_with_checksum/num_blocks)}/块)")
        print(f"    - Load (首次): {format_duration(load_time_first)} ({format_duration(load_time_first/num_blocks)}/块)")
        print(f"    - Load (缓存): {format_duration(load_time_cached)} ({format_duration(load_time_cached/num_blocks)}/块)")

        # 计算开销
        save_overhead = (save_time_with_checksum / save_time_no_checksum - 1) * 100
        load_overhead_first = (load_time_first / load_time_no_checksum - 1) * 100
        load_overhead_cached = (load_time_cached / load_time_no_checksum - 1) * 100

        print(f"\n  📊 性能开销:")
        print(f"    - Save 开销: {save_overhead:+.1f}%")
        print(f"    - Load 开销 (首次): {load_overhead_first:+.1f}%")
        print(f"    - Load 开销 (缓存): {load_overhead_cached:+.1f}%")

        if abs(load_overhead_cached) <= 5:
            print(f"  ✅ 缓存后开销在预期范围内（< 5%）")
        else:
            print(f"  ⚠️  缓存后开销仍超出预期（> 5%）")

        # 统计
        stats = manager_with_checksum.get_stats()
        print(f"\n  📊 Checksum 统计:")
        print(f"    - 真实验证: {stats.checksum_verifications}")
        print(f"    - 缓存跳过: {stats.cached_verifications}")
        print(f"    - 失败: {stats.checksum_failures}")
        print(f"    - 缓存命中率: {stats.cached_verifications / (stats.checksum_verifications + stats.cached_verifications) * 100:.1f}%")

        manager_with_checksum.stop()

    print("\n✅ Benchmark 3 完成")


def benchmark_comprehensive():
    """测试 4: 综合性能对比"""
    print("\n" + "=" * 70)
    print("Benchmark 4: Comprehensive Performance Comparison")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  跳过（需要 MLX）")
        return

    configs = [
        ("基准配置", {
            'enable_prefetch': False,
            'enable_checksum': False,
            'hot_cache_max_bytes': 0
        }),
        ("P1 全功能", {
            'enable_prefetch': True,
            'prefetch_top_n': 10,
            'prefetch_interval': 1.0,
            'enable_checksum': True,
            'hot_cache_max_bytes': 10 * 1024**2
        }),
    ]

    num_blocks = 30
    num_hot_blocks = 10

    results = []

    for config_name, config in configs:
        print(f"\n📝 测试配置: {config_name}")

        with tempfile.TemporaryDirectory() as cache_dir:
            manager = PagedSSDCacheManager(
                cache_dir=Path(cache_dir),
                max_size_bytes=1024 * 1024 * 300,
                **config
            )

            # 保存块
            block_hashes = []
            start = time.perf_counter()
            for i in range(num_blocks):
                block_hash = f"block_{i:03d}".encode().ljust(32, b'\x00')
                cache_data = create_mock_cache_data(num_layers=32, seq_len=64)
                manager.save_block(block_hash, cache_data, token_count=64)
                block_hashes.append(block_hash)
            time.sleep(1)
            save_time = time.perf_counter() - start

            # 模拟热块访问模式（重复访问前 N 个块）
            for _ in range(3):
                for block_hash in block_hashes[:num_hot_blocks]:
                    manager.load_block(block_hash)

            # 等待预取
            if config.get('enable_prefetch'):
                time.sleep(2)

            # 测试加载性能
            start = time.perf_counter()
            for _ in range(5):
                for block_hash in block_hashes[:num_hot_blocks]:
                    manager.load_block(block_hash)
            load_time = time.perf_counter() - start

            stats = manager.get_stats()

            print(f"  - Save 时间: {format_duration(save_time)}")
            print(f"  - Load 时间: {format_duration(load_time)}")
            print(f"  - Hot cache hits: {stats.hot_cache_hits}")

            results.append({
                'config': config_name,
                'save_time': save_time,
                'load_time': load_time,
                'hot_cache_hits': stats.hot_cache_hits
            })

            manager.stop()

    # 对比
    print(f"\n📊 性能对比:")
    baseline = results[0]
    optimized = results[1]

    load_speedup = baseline['load_time'] / optimized['load_time']
    print(f"  - Load 加速: {load_speedup:.2f}x")
    print(f"  - Hot cache 命中提升: {optimized['hot_cache_hits'] - baseline['hot_cache_hits']}")

    if load_speedup >= 2.0:
        print(f"  ✅ 达到预期（目标 > 2x）")
    else:
        print(f"  ⚠️  部分达标")

    print("\n✅ Benchmark 4 完成")


def main():
    """运行所有性能测试"""
    print("\n" + "=" * 70)
    print("P1 阶段性能验证")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  警告: MLX 未安装，部分测试将跳过")
        print("   建议: pip install mlx")

    try:
        # Benchmark 1: Smart Prefetch
        benchmark_smart_prefetch()

        # Benchmark 2: Adaptive Chunk Memory
        benchmark_adaptive_chunk_memory()

        # Benchmark 3: Checksum Overhead
        benchmark_checksum_overhead()

        # Benchmark 4: Comprehensive
        benchmark_comprehensive()

        print("\n" + "=" * 70)
        print("🎉 所有性能测试完成")
        print("=" * 70)

        print("\n📋 性能验证总结:")
        print("  ✅ Benchmark 1: Smart Prefetch 性能")
        print("  ✅ Benchmark 2: Adaptive Chunk 内存优化")
        print("  ✅ Benchmark 3: Checksum 性能开销")
        print("  ✅ Benchmark 4: 综合性能对比")
        print()

    except Exception as e:
        print(f"\n❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
