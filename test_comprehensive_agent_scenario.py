#!/usr/bin/env python3
"""
综合性能测试 - Agent 多轮对话场景

模拟真实 Agent 场景：
1. System prompt（512 tokens）每轮都重复使用
2. User query（256 tokens）每轮都不同
3. 多轮对话（10 轮）
4. 验证 Smart Prefetch + LRU-2 + Checksum 综合性能
"""

import sys
import tempfile
import time
from pathlib import Path
import hashlib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    import numpy as np


def create_mock_kv_cache(num_tokens, num_layers=32, embd_dim=128):
    """创建模拟的 KV cache 数据"""
    if HAS_MLX:
        cache_data = []
        for _ in range(num_layers):
            keys = mx.random.normal(shape=(num_tokens, embd_dim))
            values = mx.random.normal(shape=(num_tokens, embd_dim))
            cache_data.append((keys, values))
        return cache_data
    else:
        cache_data = []
        for _ in range(num_layers):
            keys = np.random.randn(num_tokens, embd_dim).astype(np.float16)
            values = np.random.randn(num_tokens, embd_dim).astype(np.float16)
            cache_data.append((keys, values))
        return cache_data


def simulate_agent_conversation(manager, num_rounds=10):
    """模拟 Agent 多轮对话

    场景：
    - System prompt: 512 tokens（每轮重复）
    - User query: 256 tokens（每轮不同）
    - 总共: 768 tokens/轮
    """
    print("\n📝 模拟 Agent 多轮对话场景...")
    print(f"  - 轮数: {num_rounds}")
    print(f"  - System prompt: 512 tokens（每轮重复）")
    print(f"  - User query: 256 tokens（每轮不同）")

    # 创建固定的 system prompt cache
    system_prompt_cache = create_mock_kv_cache(512)
    system_prompt_hash = hashlib.sha256(b"system_prompt_fixed").digest()

    save_times = []
    load_times = []
    hot_cache_hits = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n🔄 第 {round_num} 轮对话...")

        # 1. 保存 system prompt（应该只在第一轮保存，后续轮次缓存命中）
        start = time.time()
        manager.save_block(
            block_hash=system_prompt_hash,
            cache_data=system_prompt_cache,
            token_count=512
        )
        save_time = time.time() - start
        save_times.append(save_time)

        # 2. 创建当前轮次的 user query（每轮不同）
        user_query_cache = create_mock_kv_cache(256)
        user_query_hash = hashlib.sha256(f"user_query_{round_num}".encode()).digest()

        start = time.time()
        manager.save_block(
            block_hash=user_query_hash,
            cache_data=user_query_cache,
            token_count=256
        )
        save_time2 = time.time() - start
        save_times.append(save_time2)

        # 3. 加载 system prompt（测试预取效果）
        start = time.time()
        loaded_system = manager.load_block(system_prompt_hash)
        load_time = time.time() - start
        load_times.append(load_time)

        # 4. 检查是否命中 hot cache
        stats = manager.get_stats()
        hot_cache_hits.append(stats.hot_cache_hits)

        print(f"  - System prompt save: {save_time*1000:.2f} ms")
        print(f"  - User query save: {save_time2*1000:.2f} ms")
        print(f"  - System prompt load: {load_time*1000:.2f} ms")
        print(f"  - Hot cache hits: {stats.hot_cache_hits}")

        # 5. 等待一小段时间，让预取有机会触发
        if round_num == 1:
            print("  ⏳ 等待预取触发...")
            time.sleep(3)  # 预取间隔是 2 秒

    return {
        'save_times': save_times,
        'load_times': load_times,
        'hot_cache_hits': hot_cache_hits,
        'num_rounds': num_rounds
    }


def analyze_results(results):
    """分析测试结果"""
    print("\n" + "=" * 70)
    print("📊 综合性能分析")
    print("=" * 70)

    save_times = results['save_times']
    load_times = results['load_times']
    hot_cache_hits = results['hot_cache_hits']
    num_rounds = results['num_rounds']

    # 1. System prompt 加载性能分析
    print("\n📈 System Prompt 加载性能（重复访问场景）：")

    first_load = load_times[0] * 1000  # 第一次加载（冷启动）
    later_loads = [t * 1000 for t in load_times[1:]]  # 后续加载（预期命中预取）

    if later_loads:
        avg_later_load = sum(later_loads) / len(later_loads)
        speedup = first_load / avg_later_load if avg_later_load > 0 else 0

        print(f"  - 第一次加载（冷启动）: {first_load:.2f} ms")
        print(f"  - 后续加载平均: {avg_later_load:.2f} ms")
        print(f"  - 加速比: {speedup:.2f}x")

        if speedup > 2:
            print(f"  ✅ 预取效果显著（>{speedup:.1f}x 加速）")
        else:
            print(f"  ⚠️  预取效果不明显（仅 {speedup:.1f}x 加速）")

    # 2. Hot cache 命中率
    print("\n📊 Hot Cache 命中率：")
    print(f"  - 总加载次数: {len(load_times)}")
    print(f"  - 最终 hot cache hits: {hot_cache_hits[-1]}")
    print(f"  - 命中率: {hot_cache_hits[-1]/len(load_times)*100:.1f}%")

    # 3. 综合性能评估
    print("\n🎯 综合性能评估：")

    total_save_time = sum(save_times)
    total_load_time = sum(load_times)

    print(f"  - {num_rounds} 轮对话总耗时:")
    print(f"    - Save: {total_save_time:.2f} s")
    print(f"    - Load: {total_load_time:.2f} s")
    print(f"    - Total: {total_save_time + total_load_time:.2f} s")

    print(f"  - 平均每轮:")
    print(f"    - Save: {total_save_time/num_rounds*1000:.2f} ms")
    print(f"    - Load: {total_load_time/num_rounds*1000:.2f} ms")

    # 4. 验收标准检查
    print("\n✅ 验收标准检查：")

    checks = []

    # Check 1: 预取加速 > 2x
    if speedup > 2:
        checks.append(("预取加速 > 2x", True, f"{speedup:.2f}x"))
    else:
        checks.append(("预取加速 > 2x", False, f"{speedup:.2f}x"))

    # Check 2: Hot cache 命中率 > 30%
    hit_rate = hot_cache_hits[-1]/len(load_times)*100
    if hit_rate > 30:
        checks.append(("Hot cache 命中率 > 30%", True, f"{hit_rate:.1f}%"))
    else:
        checks.append(("Hot cache 命中率 > 30%", False, f"{hit_rate:.1f}%"))

    # Check 3: 后续加载延迟 < 10ms
    if later_loads and avg_later_load < 10:
        checks.append(("后续加载延迟 < 10ms", True, f"{avg_later_load:.2f}ms"))
    else:
        checks.append(("后续加载延迟 < 10ms", False, f"{avg_later_load:.2f}ms" if later_loads else "N/A"))

    for check_name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {value}")

    all_passed = all(c[1] for c in checks)

    return all_passed


def main():
    """主函数"""
    print("=" * 70)
    print("综合性能测试 - Agent 多轮对话场景")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  MLX 未安装，使用 NumPy 模拟")

    with tempfile.TemporaryDirectory() as cache_dir:
        print("\n📝 创建缓存管理器（Phase 1+2 全功能）...")

        # 创建全功能缓存管理器
        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=1024 * 1024 * 500,  # 500MB

            # Phase 1 优化
            enable_prefetch=True,              # P1-5: Smart Prefetch
            prefetch_top_n=10,
            prefetch_interval=2.0,
            enable_checksum=True,              # P1-6: Checksum
            enable_compression=True,           # P0-4: Compression

            # Hot cache
            hot_cache_max_bytes=10 * 1024**2,  # 10MB
        )

        print("  ✅ 缓存管理器已创建")
        print("  - Smart Prefetch: 已启用")
        print("  - Checksum Validation: 已启用")
        print("  - SSD Compression: 已启用")
        print("  - Hot Cache: 10MB")

        # 运行 Agent 对话场景
        results = simulate_agent_conversation(manager, num_rounds=10)

        # 等待所有写入完成
        print("\n⏳ 等待缓存写入完成...")
        manager.flush(timeout=30.0)

        # 分析结果
        all_passed = analyze_results(results)

        # 获取最终统计
        print("\n📊 最终统计：")
        stats = manager.get_stats()
        print(f"  Cache:")
        print(f"    - Total saves: {stats.saves}")
        print(f"    - Total loads: {stats.loads}")
        print(f"    - Hot cache hits: {stats.hot_cache_hits}")
        print(f"  Checksum:")
        print(f"    - Verifications: {stats.checksum_verifications}")
        print(f"    - Failures: {stats.checksum_failures}")

        prefetch_stats = manager.get_prefetch_stats()
        print(f"  Prefetch:")
        print(f"    - Tracked blocks: {prefetch_stats['access_tracker']['total_blocks']}")
        print(f"    - Total accesses: {prefetch_stats['access_tracker']['total_accesses']}")

        # 停止预取
        manager.stop()

        # 最终结论
        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 综合性能测试通过！")
            print("=" * 70)
            return 0
        else:
            print("⚠️  综合性能测试部分失败")
            print("=" * 70)
            return 1


if __name__ == "__main__":
    exit(main())
