#!/usr/bin/env python3
"""
test_async_overhead.py - 验证 scheduler.step() 的执行特性

假设1: scheduler.step 同步执行时间 < 50ms（如果超过，不能去除线程池）
假设2: 同步调用不会阻塞事件循环（slow_callback_duration 不会触发警告）

测试场景: FULL SKIP 模式，生成 1 token
"""

import asyncio
import time
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mlx_lm import load

# ============================================================================
# 配置
# ============================================================================

SLOW_CALLBACK_THRESHOLD = 0.05  # 50ms
NUM_ITERATIONS = 10


async def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("ASYNC OVERHEAD TEST - scheduler.step() Performance Analysis")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Slow callback threshold: {SLOW_CALLBACK_THRESHOLD*1000:.1f} ms")
    print(f"  - Iterations: {NUM_ITERATIONS}")
    print()

    # 加载模型
    print("Loading model...")
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    # 创建 engine
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig

    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        paged_cache_block_size=256,
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)
    await engine.start()

    # 准备请求（FULL SKIP 模式）
    prompt = "Hello"
    request_id = "test-sync-overhead"

    # 预热：生成一次，让后续请求进入 FULL SKIP 模式
    print("\nWarming up (first generation)...")
    await engine.generate(prompt=prompt, max_tokens=1, request_id=request_id + "-warmup")

    # -------------------------------------------------------------------------
    # 测试1: 测量 scheduler.step() 的同步执行时间
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 1: Sync Execution Time Measurement")
    print("-" * 70)

    times = []

    for i in range(NUM_ITERATIONS):
        # 创建新请求
        req_id = f"{request_id}-{i}"

        # 添加请求到 scheduler
        await engine.add_request(
            request_id=req_id,
            prompt=prompt,
            sampling_params={"max_tokens": 1},
        )

        # 测量单次 step 的同步执行时间
        start = time.perf_counter()

        # 直接调用 scheduler.step()（同步）
        outputs = engine.scheduler.step()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"  Iteration {i+1}: {elapsed*1000:.2f} ms")

        # 清理
        engine.scheduler._cleanup_finished()

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print(f"\n  Results:")
    print(f"    Average: {avg_time*1000:.2f} ms")
    print(f"    Min:     {min_time*1000:.2f} ms")
    print(f"    Max:     {max_time*1000:.2f} ms")

    hypothesis1_passed = avg_time < SLOW_CALLBACK_THRESHOLD

    # -------------------------------------------------------------------------
    # 测试2: 异步上下文中的阻塞检测
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 2: Event Loop Blocking Detection")
    print("-" * 70)

    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = SLOW_CALLBACK_THRESHOLD

    async_times = []

    for i in range(NUM_ITERATIONS):
        req_id = f"{request_id}-async-{i}"

        await engine.add_request(
            request_id=req_id,
            prompt=prompt,
            sampling_params={"max_tokens": 1},
        )

        start = time.perf_counter()

        # 在 async 上下文中同步调用
        outputs = engine.scheduler.step()

        elapsed = time.perf_counter() - start
        async_times.append(elapsed)

        # 让事件循环处理
        await asyncio.sleep(0)

        print(f"  Async iteration {i+1}: {elapsed*1000:.2f} ms")

        engine.scheduler._cleanup_finished()

    async_avg = sum(async_times) / len(async_times)

    print(f"\n  Results:")
    print(f"    Average: {async_avg*1000:.2f} ms")

    hypothesis2_passed = async_avg < SLOW_CALLBACK_THRESHOLD

    # -------------------------------------------------------------------------
    # 测试3: run_in_executor 开销（对比基准）
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST 3: run_in_executor Overhead (Current Implementation)")
    print("-" * 70)

    executor_times = []

    for i in range(NUM_ITERATIONS):
        req_id = f"{request_id}-executor-{i}"

        await engine.add_request(
            request_id=req_id,
            prompt=prompt,
            sampling_params={"max_tokens": 1},
        )

        start = time.perf_counter()

        # 使用 run_in_executor（当前实现）
        outputs = await loop.run_in_executor(None, engine.scheduler.step)

        elapsed = time.perf_counter() - start
        executor_times.append(elapsed)

        print(f"  Executor iteration {i+1}: {elapsed*1000:.2f} ms")

        engine.scheduler._cleanup_finished()

    executor_avg = sum(executor_times) / len(executor_times)

    print(f"\n  Results:")
    print(f"    Average: {executor_avg*1000:.2f} ms")
    print(f"    Overhead vs sync: {(executor_avg - avg_time)*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 最终报告
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print(f"\n✅ Hypothesis 1: sync execution < 50ms")
    print(f"     Value: {avg_time*1000:.2f} ms (threshold: {SLOW_CALLBACK_THRESHOLD*1000:.0f} ms)")
    print(f"     Status: {'PASS ✅' if hypothesis1_passed else 'FAIL ❌'}")

    print(f"\n✅ Hypothesis 2: no event loop blocking")
    print(f"     Value: {async_avg*1000:.2f} ms (threshold: {SLOW_CALLBACK_THRESHOLD*1000:.0f} ms)")
    print(f"     Status: {'PASS ✅' if hypothesis2_passed else 'FAIL ❌'}")

    print(f"\n📊 Executor overhead analysis:")
    print(f"     Sync execution:     {avg_time*1000:.2f} ms")
    print(f"     Executor execution: {executor_avg*1000:.2f} ms")
    print(f"     Overhead:           {(executor_avg - avg_time)*1000:.2f} ms ({(executor_avg/avg_time - 1)*100:.1f}%)")

    # 结论
    print("\n" + "-" * 70)
    print("CONCLUSION")
    print("-" * 70)

    if hypothesis1_passed and hypothesis2_passed:
        print("""
✅ 两个假设都通过验证！

结论: scheduler.step() 可以安全地在 async 上下文中同步调用。

推荐操作:
  - 可以考虑移除 run_in_executor 包装
  - 预期收益: -{overhead:.1f}ms/generation
  - 需要在生产环境监控实际延迟
""".format(overhead=(executor_avg - avg_time)*1000))
    else:
        print("""
❌ 一个或多个假设未通过验证！

结论: scheduler.step() 可能会阻塞事件循环，
      建议保留 run_in_executor 或寻找其他优化方案。
""")
        if not hypothesis1_passed:
            print(f"  - 同步执行时间 ({avg_time*1000:.2f}ms) 超过阈值 ({SLOW_CALLBACK_THRESHOLD*1000:.0f}ms)")
        if not hypothesis2_passed:
            print(f"  - 在异步上下文中执行时间 ({async_avg*1000:.2f}ms) 超过阈值")

    print("\n" + "=" * 70)

    # 停止 engine
    await engine.stop()

    return hypothesis1_passed and hypothesis2_passed


if __name__ == "__main__":
    result = asyncio.run(run_tests(), debug=True)
    sys.exit(0 if result else 1)
