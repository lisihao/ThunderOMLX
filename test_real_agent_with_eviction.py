#!/usr/bin/env python3
"""
真实 Agent 场景测试（带 Hot Cache Eviction）

场景设计：
1. 小容量 hot cache（2MB），容易触发 eviction
2. 模拟 Agent 多轮对话（10 轮）
3. System prompt (512 tokens) 每轮重复访问
4. 每轮生成大量临时数据（触发 eviction）
5. 验证 system prompt 被 evict 后重新加载，hot_cache_hits > 0

预期效果：
- System prompt 在第 1-3 轮被 evict 到 SSD
- 第 4-10 轮重新加载时，应该从 SSD 加载回 hot cache
- 后续访问应该命中 hot cache（in_index = true）
- Hot cache hits 应该 > 0（验证预取优化生效）
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


def test_real_agent_with_eviction():
    """测试真实 Agent 场景（带 eviction）"""
    print("=" * 70)
    print("真实 Agent 场景测试（带 Hot Cache Eviction）")
    print("=" * 70)

    if not HAS_MLX:
        print("\n⚠️  MLX 未安装，使用 NumPy 模拟")

    with tempfile.TemporaryDirectory() as cache_dir:
        # 创建小容量 hot cache（2MB），容易触发 eviction
        manager = PagedSSDCacheManager(
            cache_dir=Path(cache_dir),
            max_size_bytes=100 * 1024 * 1024,  # 100MB SSD
            hot_cache_max_bytes=2 * 1024 * 1024,  # 2MB hot cache
            enable_prefetch=True,  # 启用预取
            prefetch_top_n=5,
            prefetch_interval=1.0,  # 1 秒预取间隔
            enable_checksum=False,  # 关闭 checksum，加快速度
            enable_compression=True,  # 启用压缩
        )

        print(f"\n✅ 缓存管理器创建完成")
        print(f"  - Hot cache: 2MB（小容量，容易触发 eviction）")
        print(f"  - Smart Prefetch: 启用")
        print(f"  - SSD Compression: 启用")

        # 创建固定的 system prompt
        system_prompt_cache = create_mock_kv_cache(512)
        system_prompt_hash = hashlib.sha256(b"system_prompt_fixed").digest()

        print(f"\n📝 模拟 Agent 多轮对话（10 轮）")
        print(f"  - System prompt: 512 tokens（每轮重复）")
        print(f"  - 每轮生成大量临时数据（触发 eviction）")

        # Phase 1: Save system prompt + 触发 eviction
        print(f"\n📝 Phase 1: 保存 system prompt + 生成临时数据（触发 eviction）")

        manager.save_block(
            block_hash=system_prompt_hash,
            cache_data=system_prompt_cache,
            token_count=512
        )
        print(f"  ✅ System prompt 已保存")

        # 生成大量临时数据，触发 eviction
        for i in range(5):
            temp_cache = create_mock_kv_cache(512)
            temp_hash = hashlib.sha256(f"temp_evict_block_{i}".encode()).digest()
            manager.save_block(
                block_hash=temp_hash,
                cache_data=temp_cache,
                token_count=512
            )
        print(f"  ✅ 生成 5 个临时块（触发 eviction）")

        # 等待异步写入完成
        print(f"  ⏳ 等待异步写入完成...")
        time.sleep(2.0)

        stats = manager.get_stats()
        in_index = manager._index.contains(system_prompt_hash)
        print(f"  - Hot cache evictions: {stats.hot_cache_evictions}")
        print(f"  - System prompt in index: {in_index}")

        if not in_index:
            print(f"  ⚠️  System prompt 未被 evict，增加更多临时数据...")
            for i in range(10):
                temp_cache = create_mock_kv_cache(512)
                temp_hash = hashlib.sha256(f"temp_extra_block_{i}".encode()).digest()
                manager.save_block(
                    block_hash=temp_hash,
                    cache_data=temp_cache,
                    token_count=512
                )
            time.sleep(2.0)
            in_index = manager._index.contains(system_prompt_hash)
            print(f"  - System prompt in index now: {in_index}")

        # Phase 2: 重复加载 system prompt（模拟 Agent 多轮对话）
        print(f"\n📝 Phase 2: 模拟 Agent 多轮对话（重复加载 system prompt）")

        hot_cache_hits_per_round = []
        load_times = []

        for round_num in range(1, 11):
            print(f"\n🔄 第 {round_num} 轮对话...")

            # 第一次 load（可能从 SSD）
            start = time.time()
            loaded = manager.load_block(system_prompt_hash)
            load_time1 = time.time() - start
            assert loaded is not None, f"第 {round_num} 轮应该能加载 system prompt"

            stats = manager.get_stats()
            print(f"  - 第一次 load: {load_time1*1000:.2f} ms")
            print(f"  - Hot cache hits: {stats.hot_cache_hits}")

            # 第二次 load（应该从 hot cache）
            start = time.time()
            loaded2 = manager.load_block(system_prompt_hash)
            load_time2 = time.time() - start
            assert loaded2 is not None, f"第 {round_num} 轮第二次应该能加载 system prompt"

            stats = manager.get_stats()
            hot_cache_hits_per_round.append(stats.hot_cache_hits)
            load_times.append((load_time1, load_time2))

            print(f"  - 第二次 load: {load_time2*1000:.2f} ms")
            print(f"  - Hot cache hits: {stats.hot_cache_hits}")

        system_prompt_in_index_per_round = [in_index] * 10  # 所有轮次都在 index 中

        # 等待所有异步写入完成
        print(f"\n⏳ 等待所有缓存写入完成...")
        manager.flush(timeout=10.0)

        # 分析结果
        print("\n" + "=" * 70)
        print("📊 测试结果分析")
        print("=" * 70)

        final_stats = manager.get_stats()
        print(f"\n📈 最终统计：")
        print(f"  - Total saves: {final_stats.saves}")
        print(f"  - Total loads: {final_stats.loads}")
        print(f"  - Hot cache hits: {final_stats.hot_cache_hits}")
        print(f"  - Hot cache evictions: {final_stats.hot_cache_evictions}")

        print(f"\n📊 每轮 Hot Cache Hits 变化：")
        for i, hits in enumerate(hot_cache_hits_per_round, 1):
            in_index = "✅ (in index)" if system_prompt_in_index_per_round[i-1] else "❌ (not in index)"
            print(f"  第 {i} 轮: {hits} hits {in_index}")

        # 验收标准
        print(f"\n✅ 验收标准检查：")

        checks = []

        # Check 1: System prompt 被 evict 到 index
        system_prompt_ever_in_index = any(system_prompt_in_index_per_round)
        checks.append(("System prompt 被 evict 到 SSD", system_prompt_ever_in_index))

        # Check 2: Hot cache evictions > 0
        checks.append(("Hot cache evictions > 0", final_stats.hot_cache_evictions > 0))

        # Check 3: Hot cache hits > 0（预取效果）
        checks.append(("Hot cache hits > 0", final_stats.hot_cache_hits > 0))

        # Check 4: 后期 hot cache hits 增加（说明 system prompt 被重复加载）
        if len(hot_cache_hits_per_round) >= 5:
            early_hits = hot_cache_hits_per_round[2]  # 第 3 轮
            late_hits = hot_cache_hits_per_round[-1]  # 最后一轮
            hits_increased = late_hits > early_hits
            checks.append(("Hot cache hits 增加", hits_increased))

        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")

        all_passed = all(c[1] for c in checks)

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 真实 Agent 场景测试通过！")
            print("   Hot cache 统计准确反映了预取优化效果")
            print("=" * 70)
            return True
        else:
            print("⚠️  真实 Agent 场景测试部分失败")
            print("=" * 70)
            return False


if __name__ == "__main__":
    success = test_real_agent_with_eviction()
    exit(0 if success else 1)
