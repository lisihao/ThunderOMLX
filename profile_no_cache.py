#!/usr/bin/env python3
"""
深度性能分析 - 缓存禁用场景
使用 cProfile 找出纯推理的性能瓶颈
"""
import asyncio
import sys
import time
import gc
import cProfile
import pstats
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    print("=" * 80)
    print("🔍 深度性能分析 - 缓存禁用场景")
    print("=" * 80)
    print()
    print("目标：找出纯推理的性能瓶颈（无缓存写入干扰）")
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
    await engine.start()
    print("✅ 引擎启动（缓存已禁用）\n")

    try:
        # 预热（避免冷启动影响 profiling）
        print("预热引擎...")
        warmup_count = 0
        async for output in engine.stream_generate(
            prompt=prompt[:500],  # 短 prompt 预热
            max_tokens=10,
            temperature=0.0
        ):
            if output.new_text:
                warmup_count += 1
                if warmup_count >= 10:
                    break

        gc.collect()
        await asyncio.sleep(2)
        print("✅ 预热完成\n")

        # ================================================================
        # Phase 1: Prefill (PP) Profiling
        # ================================================================
        print("=" * 80)
        print("📊 Phase 1: Prefill (PP) 性能分析")
        print("=" * 80)
        print()

        profiler_pp = cProfile.Profile()
        profiler_pp.enable()

        start_pp = time.perf_counter()
        first_token_time = None

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=1,  # 只生成 1 个 token，专注 prefill
            temperature=0.0
        ):
            if output.new_text and first_token_time is None:
                first_token_time = time.perf_counter()
                break

        profiler_pp.disable()

        if first_token_time:
            prefill_time = first_token_time - start_pp
            pp_tps = actual_tokens / prefill_time

            print(f"✅ Prefill 完成")
            print(f"   时间: {prefill_time:.3f}s")
            print(f"   PP TPS: {pp_tps:.1f} tok/s")
            print()

        # 保存 PP profiling 结果
        print("分析 Prefill 性能瓶颈...\n")
        s = io.StringIO()
        ps = pstats.Stats(profiler_pp, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')

        # Top 30 函数（按累计时间）
        print("━" * 80)
        print("Top 30 函数（按累计时间）")
        print("━" * 80)
        ps.print_stats(30)

        # 保存到文件
        with open('/tmp/profile_pp_cumulative.txt', 'w') as f:
            ps2 = pstats.Stats(profiler_pp, stream=f)
            ps2.strip_dirs()
            ps2.sort_stats('cumulative')
            ps2.print_stats(100)

        # Top 30 函数（按总时间）
        print("\n" + "━" * 80)
        print("Top 30 函数（按总时间 tottime）")
        print("━" * 80)
        ps.sort_stats('tottime')
        ps.print_stats(30)

        with open('/tmp/profile_pp_tottime.txt', 'w') as f:
            ps2 = pstats.Stats(profiler_pp, stream=f)
            ps2.strip_dirs()
            ps2.sort_stats('tottime')
            ps2.print_stats(100)

        print()
        print("✅ Prefill profiling 结果已保存:")
        print("   - /tmp/profile_pp_cumulative.txt (按累计时间)")
        print("   - /tmp/profile_pp_tottime.txt (按总时间)")
        print()

        gc.collect()
        await asyncio.sleep(2)

        # ================================================================
        # Phase 2: Token Generation (TG) Profiling
        # ================================================================
        print("\n" + "=" * 80)
        print("📊 Phase 2: Token Generation (TG) 性能分析")
        print("=" * 80)
        print()

        profiler_tg = cProfile.Profile()

        # 先运行到 first token（不计入 profiling）
        start_tg = time.perf_counter()
        first_gen_token_time = None
        token_count = 0

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=1,
            temperature=0.7
        ):
            if output.new_text:
                first_gen_token_time = time.perf_counter()
                break

        # 从第 2 个 token 开始 profiling
        if first_gen_token_time:
            print("开始 profiling token generation...")
            profiler_tg.enable()

            gen_start = time.perf_counter()

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=128,
                temperature=0.7
            ):
                if output.new_text:
                    token_count += 1

            profiler_tg.disable()
            gen_time = time.perf_counter() - gen_start

            tg_tps = token_count / gen_time if gen_time > 0 else 0

            print(f"✅ Token Generation 完成")
            print(f"   生成 tokens: {token_count}")
            print(f"   时间: {gen_time:.3f}s")
            print(f"   TG TPS: {tg_tps:.1f} tok/s")
            print()

        # 保存 TG profiling 结果
        print("分析 Token Generation 性能瓶颈...\n")
        s_tg = io.StringIO()
        ps_tg = pstats.Stats(profiler_tg, stream=s_tg)
        ps_tg.strip_dirs()
        ps_tg.sort_stats('cumulative')

        print("━" * 80)
        print("Top 30 函数（按累计时间）")
        print("━" * 80)
        ps_tg.print_stats(30)

        with open('/tmp/profile_tg_cumulative.txt', 'w') as f:
            ps2 = pstats.Stats(profiler_tg, stream=f)
            ps2.strip_dirs()
            ps2.sort_stats('cumulative')
            ps2.print_stats(100)

        print("\n" + "━" * 80)
        print("Top 30 函数（按总时间 tottime）")
        print("━" * 80)
        ps_tg.sort_stats('tottime')
        ps_tg.print_stats(30)

        with open('/tmp/profile_tg_tottime.txt', 'w') as f:
            ps2 = pstats.Stats(profiler_tg, stream=f)
            ps2.strip_dirs()
            ps2.sort_stats('tottime')
            ps2.print_stats(100)

        print()
        print("✅ Token Generation profiling 结果已保存:")
        print("   - /tmp/profile_tg_cumulative.txt (按累计时间)")
        print("   - /tmp/profile_tg_tottime.txt (按总时间)")
        print()

        # ================================================================
        # 总结
        # ================================================================
        print("\n" + "=" * 80)
        print("📊 性能分析总结")
        print("=" * 80)
        print()
        print(f"Prefill (8K tokens):")
        print(f"  - 时间: {prefill_time:.3f}s")
        print(f"  - PP TPS: {pp_tps:.1f} tok/s")
        print()
        print(f"Token Generation (128 tokens):")
        print(f"  - 时间: {gen_time:.3f}s")
        print(f"  - TG TPS: {tg_tps:.1f} tok/s")
        print()
        print("详细 profiling 数据:")
        print("  - /tmp/profile_pp_cumulative.txt")
        print("  - /tmp/profile_pp_tottime.txt")
        print("  - /tmp/profile_tg_cumulative.txt")
        print("  - /tmp/profile_tg_tottime.txt")
        print()
        print("💡 下一步: 分析 profiling 数据，找出瓶颈函数")
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
