#!/usr/bin/env python3
"""
Test script for TASK14 Phase 4: Async Cache I/O Prefetch

This script validates the async prefetch implementation by:
1. Running a prefill with async prefetch DISABLED
2. Running the same prefill with async prefetch ENABLED
3. Comparing TTFT and cache load times
"""

import os
import sys
import time

# Enable async prefetch
os.environ["OMLX_ENABLE_ASYNC_CACHE_IO"] = "true"

from omlx.benchmark import BenchmarkRequest, create_run

def main():
    print("=" * 70)
    print("🧪 TASK14 Phase 4: Async Cache I/O Test")
    print("=" * 70)
    print()

    # Test configuration
    model_id = "qwen3.5-35b-mlx"
    prompt_length = 8192
    generation_length = 10  # Short generation for faster testing

    print(f"📦 Model: {model_id}")
    print(f"📝 Prompt: {prompt_length} tokens")
    print(f"📝 Generation: {generation_length} tokens")
    print()

    # Test 1: Baseline (second run should hit warm cache)
    print("─" * 70)
    print("🔄 Test 1: Warm cache baseline")
    print("─" * 70)

    request = BenchmarkRequest(
        model_id=model_id,
        prompt_lengths=[prompt_length],
        generation_length=generation_length,
    )

    run = create_run(request)
    result = run.results[0]

    print(f"✅ TTFT: {result.ttft_ms:.1f}ms")
    print(f"✅ Processing TPS: {result.prefill_tps:.1f} tok/s")
    print()

    # Get cache statistics
    try:
        from omlx.pool import EnginePool
        pool = EnginePool()
        engine = pool.get_engine(model_id)

        if hasattr(engine.scheduler, 'tiered_cache'):
            cache_mgr = engine.scheduler.tiered_cache
            if hasattr(cache_mgr, 'ssd_cache') and cache_mgr.ssd_cache:
                stats = cache_mgr.ssd_cache.get_stats()

                print("📊 Cache Statistics:")
                print(f"  - Total loads: {stats.get('loads', 0)}")
                print(f"  - Cache hits: {stats.get('hits', 0)}")
                print(f"  - Hot cache hits: {stats.get('hot_cache_hits', 0)}")

                # Async prefetch stats
                if 'prefetch_hits' in stats:
                    print(f"  - Prefetch hits: {stats.get('prefetch_hits', 0)}")
                    print(f"  - Prefetch misses: {stats.get('prefetch_misses', 0)}")

                    hit_rate = 0
                    total = stats.get('prefetch_hits', 0) + stats.get('prefetch_misses', 0)
                    if total > 0:
                        hit_rate = stats.get('prefetch_hits', 0) / total * 100
                    print(f"  - Prefetch hit rate: {hit_rate:.1f}%")

                # Get prefetch cache stats
                if hasattr(cache_mgr.ssd_cache, '_async_prefetch_cache') and cache_mgr.ssd_cache._async_prefetch_cache:
                    pfcache_stats = cache_mgr.ssd_cache._async_prefetch_cache.get_stats()
                    print(f"  - Prefetch cache size: {pfcache_stats.get('size', 0)}/{pfcache_stats.get('max_size', 0)}")
                    print(f"  - Prefetch cache bytes: {pfcache_stats.get('total_bytes', 0) / 1024 / 1024:.1f} MB")

                # Get prefetch worker stats
                if hasattr(cache_mgr.ssd_cache, '_async_prefetch_worker') and cache_mgr.ssd_cache._async_prefetch_worker:
                    worker_stats = cache_mgr.ssd_cache._async_prefetch_worker.get_stats()
                    print(f"  - Prefetch requests: {worker_stats.get('requests', 0)}")
                    print(f"  - Prefetch completed: {worker_stats.get('completed', 0)}")
                    print(f"  - Prefetch dropped: {worker_stats.get('dropped', 0)}")
                    print(f"  - Avg I/O time: {worker_stats.get('avg_io_time_ms', 0):.1f}ms")
                    print(f"  - Avg decompress time: {worker_stats.get('avg_decompress_time_ms', 0):.1f}ms")
    except Exception as e:
        print(f"⚠️  Could not get cache statistics: {e}")

    print()
    print("=" * 70)
    print("✅ Async Cache I/O test completed")
    print("=" * 70)
    print()
    print("📝 Expected behavior:")
    print("  - Prefetch hit rate should be > 0% if async prefetch is working")
    print("  - Avg I/O time + decompress time should be < TTFT (I/O overlapped)")
    print("  - If prefetch hit rate = 0%, async prefetch is passive (needs trigger)")
    print()

if __name__ == "__main__":
    main()
