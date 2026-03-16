#!/usr/bin/env python3
"""
深度端到端性能分析 - 缓存禁用
使用注入式 profiling 工具，分阶段分析性能瓶颈
"""
import asyncio
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入 profiling 工具
from profile_generation import inject_profiling, analyze_results, PERF_STATS


async def main():
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    print("=" * 80)
    print("🔍 深度端到端性能分析 - 缓存禁用")
    print("=" * 80)
    print()
    print("工具：注入式分阶段 profiling")
    print("目标：找出每个阶段的性能瓶颈")
    print()

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
    print("✅ Tokenizer 加载完成\n")

    # 生成 8K prompt
    print("生成 8K prompt...")
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
    ]
    text = " ".join(base_words * 1000)
    tokens = tokenizer.encode(text)[:8192]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)
    print(f"✅ Prompt: {actual_tokens} tokens\n")

    # 初始化引擎（缓存禁用）
    print("初始化引擎（缓存禁用）...")
    scheduler_config = SchedulerConfig(
        paged_ssd_cache_dir=None,  # 禁用 SSD 缓存
        hot_cache_max_size=0        # 禁用 hot cache
    )
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    # 注入性能分析
    print("注入性能分析...")
    inject_profiling()

    await engine.start()
    print("✅ 引擎启动（缓存已禁用）\n")

    try:
        # 清空之前的统计数据
        PERF_STATS.clear()

        # ================================================================
        # Test: 8K Prefill + 128 Token Generation
        # ================================================================
        print("=" * 80)
        print("📊 测试：8K Prefill + 128 Token Generation")
        print("=" * 80)
        print()

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        tokens_generated = []

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=128,
            temperature=0.7
        ):
            if output.new_text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                tokens_generated.append(time.perf_counter())

        end_time = time.perf_counter()

        # 计算指标
        if first_token_time:
            prefill_time = first_token_time - start_time
            pp_tps = actual_tokens / prefill_time
            total_time = end_time - start_time
            gen_time = end_time - first_token_time
            tg_tps = token_count / gen_time if gen_time > 0 else 0

            print()
            print("✅ 测试完成")
            print()
            print(f"Prefill (8K tokens):")
            print(f"  - 时间: {prefill_time:.3f}s")
            print(f"  - PP TPS: {pp_tps:.1f} tok/s")
            print()
            print(f"Token Generation ({token_count} tokens):")
            print(f"  - 时间: {gen_time:.3f}s")
            print(f"  - TG TPS: {tg_tps:.1f} tok/s")
            print()
            print(f"总时间: {total_time:.3f}s")
            print()

        # ================================================================
        # 分析性能数据
        # ================================================================
        analyze_results()

        # ================================================================
        # 额外分析：Token 间隔时间
        # ================================================================
        if len(tokens_generated) > 1:
            print("\n" + "=" * 80)
            print("📊 Token 间隔时间分析（TPOT）")
            print("=" * 80)
            print()

            intervals = []
            for i in range(1, len(tokens_generated)):
                interval = (tokens_generated[i] - tokens_generated[i-1]) * 1000
                intervals.append(interval)

            if intervals:
                avg_tpot = sum(intervals) / len(intervals)
                min_tpot = min(intervals)
                max_tpot = max(intervals)
                p50_tpot = sorted(intervals)[len(intervals) // 2]
                p95_tpot = sorted(intervals)[int(len(intervals) * 0.95)]

                print(f"平均 TPOT: {avg_tpot:.2f} ms/token")
                print(f"P50 TPOT:  {p50_tpot:.2f} ms/token")
                print(f"P95 TPOT:  {p95_tpot:.2f} ms/token")
                print(f"最小 TPOT: {min_tpot:.2f} ms/token")
                print(f"最大 TPOT: {max_tpot:.2f} ms/token")
                print()
                print(f"对应 TPS: {1000/avg_tpot:.1f} tok/s")
                print()

        # ================================================================
        # 瓶颈总结
        # ================================================================
        print("=" * 80)
        print("🎯 性能瓶颈总结")
        print("=" * 80)
        print()

        # 从 PERF_STATS 中提取关键指标
        if '4.batch_generator.next' in PERF_STATS:
            batch_gen_times = PERF_STATS['4.batch_generator.next']
            avg_batch_gen = sum(batch_gen_times) / len(batch_gen_times)
            total_step_times = PERF_STATS.get('0.total_step', [])
            avg_total = sum(total_step_times) / len(total_step_times) if total_step_times else 0

            if avg_total > 0:
                batch_gen_pct = (avg_batch_gen / avg_total) * 100
                print(f"1. batch_generator.next: {avg_batch_gen:.2f} ms ({batch_gen_pct:.1f}%)")
                print(f"   - 这是模型推理的核心瓶颈（MLX forward）")
                print()

        if '3.schedule_waiting' in PERF_STATS:
            schedule_times = PERF_STATS['3.schedule_waiting']
            avg_schedule = sum(schedule_times) / len(schedule_times)
            if avg_total > 0:
                schedule_pct = (avg_schedule / avg_total) * 100
                print(f"2. schedule_waiting: {avg_schedule:.2f} ms ({schedule_pct:.1f}%)")
                print()

        if '5.process_responses' in PERF_STATS:
            process_times = PERF_STATS['5.process_responses']
            avg_process = sum(process_times) / len(process_times)
            if avg_total > 0:
                process_pct = (avg_process / avg_total) * 100
                print(f"3. process_responses: {avg_process:.2f} ms ({process_pct:.1f}%)")
                print()

        if '6.cleanup' in PERF_STATS:
            cleanup_times = PERF_STATS['6.cleanup']
            avg_cleanup = sum(cleanup_times) / len(cleanup_times)
            if avg_total > 0:
                cleanup_pct = (avg_cleanup / avg_total) * 100
                if cleanup_pct > 1.0:
                    print(f"4. cleanup: {avg_cleanup:.2f} ms ({cleanup_pct:.1f}%)")
                    print()

        print("💡 优化建议:")
        print("  - 主要瓶颈在 MLX 模型推理（batch_generator.next）")
        print("  - 这部分无法优化（Metal GPU 计算）")
        print("  - 可优化的是其他阶段（schedule, process, cleanup）")
        print()

        return True

    finally:
        print("关闭引擎...")
        await engine.stop()
        gc.collect()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
