#!/usr/bin/env python3
"""
并发 Processing TPS 测试 - 直接使用 BatchedEngine

模拟原始基准场景：
- 多个并发请求
- 触发 save_block 操作
- 测量 Processing TPS（包含 cleanup）
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def run_concurrent_requests(engine, prompts: list, max_tokens: int):
    """并发运行多个请求"""

    async def single_request(prompt_text: str, request_id: str):
        """单个请求"""
        start_time = time.time()
        generated_tokens = 0

        async for output in engine.stream_generate(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=0.0
        ):
            if output.new_text:
                generated_tokens += 1

        elapsed = time.time() - start_time
        return {
            "request_id": request_id,
            "tokens": generated_tokens,
            "time_s": elapsed
        }

    # 并发运行所有请求
    tasks = [
        single_request(prompts[i], f"R{i+1}")
        for i in range(len(prompts))
    ]

    wall_start = time.time()
    results = await asyncio.gather(*tasks)
    wall_end = time.time()

    return results, wall_end - wall_start


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🎯 Concurrent Processing TPS Test - Phase 1-4 Validation")
    print("=" * 80)
    print("")

    # 创建引擎
    print("📦 Loading model...")
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        # 准备并发请求（模拟 Agent 场景）
        prompts = [
            "Explain the key differences between Python and JavaScript in detail.",
            "What are the main advantages of using TypeScript over JavaScript?",
            "Describe the most important design patterns in software engineering.",
            "Explain how asynchronous programming works in Python with examples."
        ]

        num_requests = 4
        max_tokens_per_request = 128

        print(f"📋 Configuration:")
        print(f"  Concurrent requests: {num_requests}")
        print(f"  Tokens per request: {max_tokens_per_request}")
        print(f"  Total expected tokens: {num_requests * max_tokens_per_request}")
        print("")

        # 运行并发测试
        print("🚀 Running concurrent requests...")
        print("  (This will trigger save_block operations)")
        print("")

        results, wall_time = await run_concurrent_requests(
            engine,
            prompts[:num_requests],
            max_tokens_per_request
        )

        # 计算 Processing TPS
        total_tokens = sum(r["tokens"] for r in results)
        processing_tps = total_tokens / wall_time

        # 输出结果
        print("=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        print("")

        print(f"Total requests: {num_requests}")
        print(f"Total tokens: {total_tokens}")
        print(f"Wall time: {wall_time:.2f}s")
        print("")
        print(f"🎯 Processing TPS: {processing_tps:.1f} tok/s")
        print("")

        # 显示每个请求的详情
        print("Individual request details:")
        for r in results:
            tps = r["tokens"] / r["time_s"] if r["time_s"] > 0 else 0
            print(f"  {r['request_id']}: {r['tokens']} tokens in {r['time_s']:.2f}s ({tps:.1f} tok/s)")

        print("")
        print("=" * 80)
        print("📈 Performance Comparison")
        print("=" * 80)
        print("")

        baseline_tps = 692.7
        target_tps = 730.0

        improvement = (processing_tps - baseline_tps) / baseline_tps * 100

        print(f"Baseline (Phase 0):  {baseline_tps} tok/s")
        print(f"Current (Phase 1-4): {processing_tps:.1f} tok/s")
        print(f"Improvement:         {improvement:+.1f}%")
        print(f"Target:              {target_tps} tok/s (+5.4%)")
        print("")

        if processing_tps >= target_tps:
            print("🎉 TARGET ACHIEVED!")
            print("   Phase 1-4 optimizations successfully improved Processing TPS")
        else:
            shortfall = target_tps - processing_tps
            print(f"⚠️  Below target by {shortfall:.1f} tok/s")

    finally:
        await engine.stop()

    print("")
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
