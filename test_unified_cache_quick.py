"""快速验证统一内存双层缓存功能"""
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.thunder_config import SerializationConfig


def quick_test():
    """快速功能测试"""
    print("=== ThunderOMLX 统一内存双层缓存快速测试 ===\n")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建缓存管理器（L2: 4MB, L3: 100MB）
        config = SerializationConfig(compression="zlib", enable_checksum=True)
        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=4,  # 小一点便于测试驱逐
            l3_cache_path=tmpdir / "l3_cache",
            l3_max_size_gb=1,  # 1GB
            serialization_config=config,
        )

        # 测试 1: L2 缓存命中
        print("[测试 1: L2 缓存命中]")
        tensor1 = mx.random.normal(shape=(256, 256))  # ~256KB
        key1 = "test_tensor_1"

        start = time.perf_counter()
        cache_mgr.store(key1, tensor1)
        store_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        loaded1, hit1 = cache_mgr.fetch(key1)
        fetch_time = (time.perf_counter() - start) * 1000

        print(f"  存储时间: {store_time:.2f} ms")
        print(f"  L2 命中时间: {fetch_time:.2f} ms")
        print(f"  数据正确: {'✅' if mx.allclose(tensor1, loaded1) else '❌'}")
        print(f"  L2 命中: {'✅' if hit1 else '❌'}")
        print(f"  验收标准: {'✅ 通过' if fetch_time < 5 else '⚠️ 超时'}  (< 5ms)\n")

        # 测试 2: L2 驱逐到 L3
        print("[测试 2: L2 驱逐到 L3]")
        # 创建多个大张量，撑爆 L2 (16MB)
        tensors = []
        keys = []
        for i in range(10):  # 10 个 2MB 张量 = 20MB > 16MB
            tensor = mx.random.normal(shape=(512, 512))  # ~1MB
            key = f"test_tensor_{i+2}"
            cache_mgr.store(key, tensor)
            tensors.append(tensor)
            keys.append(key)

        stats = cache_mgr.get_stats()
        print(f"  L2 驱逐次数: {stats.l2_evictions}")
        print(f"  L2→L3 晋升次数: {stats.l2_to_l3_promotions}")
        print(f"  L2 大小: {stats.l2_size_bytes / (1024**2):.2f} MB")
        print(f"  L3 大小: {stats.l3_size_bytes / (1024**2):.2f} MB")
        print(f"  验收标准: {'✅ 通过' if stats.l2_evictions > 0 else '❌ 失败'}  (发生驱逐)\n")

        # 测试 3: L3 缓存命中（从磁盘加载）
        print("[测试 3: L3 缓存命中]")
        # 清空 L2，强制从 L3 读取
        cache_mgr.l2_cache.clear()
        cache_mgr.l2_size_bytes = 0

        # 获取第一个被驱逐的 tensor
        evicted_key = keys[0]

        start = time.perf_counter()
        loaded_from_l3, hit_l3 = cache_mgr.fetch(evicted_key)
        l3_fetch_time = (time.perf_counter() - start) * 1000

        print(f"  L3 命中时间: {l3_fetch_time:.2f} ms")
        print(f"  L3 命中: {'✅' if hit_l3 else '❌'}")
        print(f"  数据正确: {'✅' if loaded_from_l3 is not None and mx.allclose(tensors[0], loaded_from_l3) else '❌'}")
        print(f"  验收标准: {'✅ 通过' if l3_fetch_time < 50 else '⚠️ 超时'}  (< 50ms)\n")

        # 测试 4: 跨会话恢复（模拟重启）
        print("[测试 4: 跨会话恢复]")
        l3_cache_path = tmpdir / "l3_cache"

        # 销毁第一个缓存管理器
        del cache_mgr

        # 创建新的缓存管理器（模拟重启）
        cache_mgr2 = UnifiedMemoryCacheManager(
            l2_max_size_mb=4,
            l3_cache_path=l3_cache_path,
            l3_max_size_gb=1,
            serialization_config=config,
        )

        # 验证 L3 数据已恢复
        print(f"  L3 恢复条目数: {len(cache_mgr2.l3_index)}")
        print(f"  L3 恢复大小: {cache_mgr2.l3_size_bytes / (1024**2):.2f} MB")

        # 尝试加载数据
        loaded_after_restart, hit_after_restart = cache_mgr2.fetch(evicted_key)
        print(f"  重启后 L3 命中: {'✅' if hit_after_restart else '❌'}")
        print(f"  数据正确: {'✅' if loaded_after_restart is not None and mx.allclose(tensors[0], loaded_after_restart) else '❌'}")
        print(f"  验收标准: {'✅ 通过' if len(cache_mgr2.l3_index) > 0 else '❌ 失败'}  (L3 持久化)\n")

        # 测试 5: 统计接口
        print("[测试 5: 统计接口]")
        stats2 = cache_mgr2.get_stats()
        stats_dict = stats2.to_dict()

        print(f"  L2 命中率: {stats2.l2_hit_rate:.2%}")
        print(f"  L3 命中率: {stats2.l3_hit_rate:.2%}")
        print(f"  整体命中率: {stats2.overall_hit_rate:.2%}")
        print(f"  L3→L2 晋升: {stats2.l3_to_l2_promotions}")
        print(f"  验收标准: {'✅ 通过' if 'l2_hit_rate' in stats_dict else '❌ 失败'}  (统计完整)\n")

    print("=== 快速测试完成 ===")


if __name__ == "__main__":
    quick_test()
