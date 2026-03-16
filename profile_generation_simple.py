#!/usr/bin/env python3
"""
简化版生成阶段性能分析

直接调用 engine 进行性能分析，不依赖 benchmark API
"""
import asyncio
import sys
import time
from pathlib import Path
from functools import wraps
from collections import defaultdict
from typing import Dict, List

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 全局性能统计
PERF_STATS: Dict[str, List[float]] = defaultdict(list)


def inject_profiling():
    """注入性能分析代码到 scheduler"""
    from omlx.scheduler import Scheduler, SchedulerOutput
    import logging

    logger = logging.getLogger(__name__)

    # 保存原始方法
    original_step = Scheduler.step

    @wraps(original_step)
    def profiled_step(self):
        """添加详细性能分析的 step 方法"""
        step_start = time.perf_counter()
        output = SchedulerOutput()

        # Phase 1: Process pending aborts
        t1 = time.perf_counter()
        self._process_pending_aborts()
        PERF_STATS['1.process_aborts'].append((time.perf_counter() - t1) * 1000)

        # Phase 2: Check memory pressure
        t2 = time.perf_counter()
        if self.memory_monitor is not None:
            self._check_memory_pressure()
        PERF_STATS['2.memory_check'].append((time.perf_counter() - t2) * 1000)

        try:
            # Phase 3: Schedule waiting requests
            t3 = time.perf_counter()
            scheduled = self._schedule_waiting()
            schedule_time = (time.perf_counter() - t3) * 1000
            PERF_STATS['3.schedule_waiting'].append(schedule_time)

            output.scheduled_request_ids = [r.request_id for r in scheduled]
            output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)

            # Phase 4: Run generation step (THE KEY BOTTLENECK)
            if self.batch_generator is not None and self.running:
                t4 = time.perf_counter()

                # 🎯 关键：测量 batch_generator.next() 的耗时
                responses = self.batch_generator.next()
                gen_time = (time.perf_counter() - t4) * 1000
                PERF_STATS['4.batch_generator.next'].append(gen_time)

                output.has_work = True

                if responses:
                    # Phase 5: Process batch responses
                    t5 = time.perf_counter()
                    outputs, finished_ids = self._process_batch_responses(responses)
                    process_time = (time.perf_counter() - t5) * 1000
                    PERF_STATS['5.process_responses'].append(process_time)

                    output.outputs = outputs
                    output.finished_request_ids = finished_ids

                    # Phase 6: Cleanup finished
                    t6 = time.perf_counter()
                    self._cleanup_finished(finished_ids)
                    PERF_STATS['6.cleanup'].append((time.perf_counter() - t6) * 1000)

        except Exception as e:
            # 简化异常处理，保持与原始代码一致
            import traceback
            logger.error(f"Error in profiled step: {e}\n{traceback.format_exc()}")
            # 调用原始方法处理异常
            return original_step(self)

        # Clear finished tracking
        self.finished_req_ids = set()

        # Periodic cleanup
        self._step_counter += 1
        if (
            self.config.mlx_cache_cleanup_interval > 0
            and self._step_counter % self.config.mlx_cache_cleanup_interval == 0
        ):
            import mlx.core as mx
            mx.clear_cache()
        if (
            self.config.gc_cleanup_interval > 0
            and self._step_counter % self.config.gc_cleanup_interval == 0
        ):
            import gc
            gc.collect()

        # 记录总耗时
        step_time = (time.perf_counter() - step_start) * 1000
        PERF_STATS['0.total_step'].append(step_time)

        return output

    # 替换方法
    Scheduler.step = profiled_step
    logger.info("✅ 性能分析已注入到 Scheduler.step()")


def analyze_results():
    """分析性能统计结果"""
    print("\n" + "="*80)
    print("🔍 生成阶段性能分析")
    print("="*80 + "\n")

    total_steps = len(PERF_STATS.get('0.total_step', []))
    if total_steps == 0:
        print("❌ 没有收集到性能数据")
        return

    print(f"📊 总执行 steps: {total_steps}\n")

    # 分阶段显示（按序号排序）
    phase_order = [
        '0.total_step',
        '1.process_aborts',
        '2.memory_check',
        '3.schedule_waiting',
        '4.batch_generator.next',
        '5.process_responses',
        '6.cleanup',
    ]

    total_avg = 0
    batch_gen_avg = 0

    for name in phase_order:
        times = PERF_STATS.get(name, [])
        if not times:
            continue

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        p50 = sorted(times)[len(times) // 2]
        p95 = sorted(times)[int(len(times) * 0.95)]

        # 记录关键指标
        if name == '0.total_step':
            total_avg = avg_time
        elif name == '4.batch_generator.next':
            batch_gen_avg = avg_time

        # 显示
        phase_name = name.split('.', 1)[1] if '.' in name else name
        print(f"📌 {phase_name}")
        print(f"   Avg: {avg_time:.2f} ms  |  P50: {p50:.2f} ms  |  P95: {p95:.2f} ms")

        # 对于总耗时和关键阶段，显示更多信息
        if name in ['0.total_step', '4.batch_generator.next']:
            print(f"   Min: {min_time:.2f} ms  |  Max: {max_time:.2f} ms")

        print()

    # 计算推理 TPS
    if total_avg > 0:
        tps = 1000 / total_avg
        print(f"⚡ ThunderOMLX TPS: {tps:.1f} tok/s")
        print(f"   TPOT: {total_avg:.2f} ms/token")

    # 对比 Native MLX
    print(f"\n📊 对比 Native MLX:")
    native_tps = 80.1  # Native MLX pp8192
    native_tpot = 1000 / native_tps
    print(f"   Native MLX TPS: {native_tps:.1f} tok/s")
    print(f"   Native MLX TPOT: {native_tpot:.2f} ms/token")

    if total_avg > 0:
        overhead = total_avg - native_tpot
        overhead_pct = (overhead / native_tpot) * 100
        print(f"\n⚠️  ThunderOMLX 开销: +{overhead:.2f} ms/token ({overhead_pct:.1f}%)")

    # 瓶颈分析
    if batch_gen_avg > 0:
        batch_gen_pct = (batch_gen_avg / total_avg) * 100 if total_avg > 0 else 0
        print(f"\n🎯 batch_generator.next() 占比: {batch_gen_pct:.1f}%")
        print(f"   ({batch_gen_avg:.2f} ms / {total_avg:.2f} ms)")

        if batch_gen_pct < 80:
            print(f"\n💡 瓶颈在 ThunderOMLX 层（不是 MLX 层）")
            print(f"   需要优化调度器、缓存管理等开销")
        else:
            print(f"\n💡 瓶颈在 MLX 生成层")
            print(f"   需要优化 KV cache、attention 等操作")


async def run_profile_test():
    """运行性能分析测试"""
    from omlx.engine.batched import BatchedEngine
    from omlx.config import OMLXConfig
    from omlx.settings_v2 import Settings

    # 注入性能分析
    inject_profiling()

    print("🧪 开始性能分析 - pp8192/tg128")
    print("="*80)

    # 加载设置
    settings = Settings.load()
    config = OMLXConfig()
    config.model.model_dirs = [Path(settings.model.model_dir)]
    config.scheduler.max_num_seqs = settings.scheduler.max_num_seqs

    # 创建 engine
    print("📦 加载模型...")
    engine = BatchedEngine(config=config)
    await engine.start()

    try:
        # 生成 8192 token prompt
        print("📝 生成 8192 token prompt...")
        filler = "The quick brown fox jumps over the lazy dog. " * 1000
        tokens = engine.tokenizer.encode(filler)[:8192]
        prompt = engine.tokenizer.decode(tokens)

        print(f"✅ Prompt tokens: {len(engine.tokenizer.encode(prompt))}")

        # 运行生成并收集性能数据
        print("\n🚀 开始生成 128 tokens...\n")
        start_time = time.perf_counter()
        token_count = 0

        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=128,
            temperature=0.0
        ):
            if output.new_text:
                token_count += len(output.new_text)
                if token_count %10 == 0:
                    print(f"  Generated {token_count} tokens...", end="\r")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"\n\n✅ 生成完成!")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   生成 tokens: {token_count}")
        print(f"   整体 TPS: {token_count / total_time:.1f} tok/s")

    finally:
        await engine.stop()

    # 分析性能统计
    analyze_results()


if __name__ == "__main__":
    asyncio.run(run_profile_test())
