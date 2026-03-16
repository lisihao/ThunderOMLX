#!/usr/bin/env python3
"""
Processing TPS 测试 - 测量包含 cleanup 的总体性能

Phase 1-4 优化主要体现在请求完成时的 cleanup 阶段：
- Phase 1: 异步 tensor 提取（节省 1.035s）
- Phase 2: 异步 save_block 调用（节省 0.781s）
- Phase 3: 减少调度间隙（节省 0.370s）
- Phase 4: 批量 Metal 操作（节省 0.104s）

此测试会：
1. 运行多个请求（触发 cleanup_finished）
2. 测量总 walltime 和总 tokens
3. 计算 Processing TPS = 总tokens / walltime
4. 检查 Phase 3/4 的日志输出
"""
import asyncio
import sys
import time
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def run_single_request(engine, prompt_text: str, max_tokens: int, request_id: str):
    """运行单个请求"""
    print(f"\n🔵 Request {request_id}: 开始生成 {max_tokens} tokens")

    start_time = time.time()
    generated_tokens = 0

    async for output in engine.stream_generate(
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=0.0
    ):
        if output.new_text:
            generated_tokens += 1
            if generated_tokens % 20 == 0:
                print(f"  {request_id}: {generated_tokens} tokens", end="\r")

    elapsed = time.time() - start_time
    print(f"\n✅ Request {request_id}: 完成 {generated_tokens} tokens in {elapsed:.2f}s")

    return {
        "request_id": request_id,
        "tokens": generated_tokens,
        "time_s": elapsed
    }


async def main():
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🧪 Processing TPS 测试 - 多请求场景")
    print("=" * 80)

    # 创建引擎
    print("\n📦 加载模型...")
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        # 准备测试请求
        print("\n📝 准备测试请求...")

        # 4 个请求，每个生成 128 tokens
        prompts = [
            "Explain the key differences between Python and JavaScript.",
            "What are the main advantages of using TypeScript?",
            "Describe the most important design patterns in software.",
            "Explain how asynchronous programming works in Python."
        ]

        num_requests = 4
        max_tokens_per_request = 128

        print(f"  请求数量: {num_requests}")
        print(f"  每个请求生成: {max_tokens_per_request} tokens")
        print(f"  预期总 tokens: {num_requests * max_tokens_per_request}")

        # 运行请求（顺序执行，模拟agent场景）
        print("\n🚀 开始执行请求...")
        wall_start = time.time()

        results = []
        for i, prompt in enumerate(prompts[:num_requests]):
            result = await run_single_request(
                engine,
                prompt,
                max_tokens_per_request,
                f"R{i+1}"
            )
            results.append(result)

            # 短暂等待，让 cleanup 有时间完成
            await asyncio.sleep(0.5)

        wall_end = time.time()
        wall_time = wall_end - wall_start

        # 计算 Processing TPS
        total_tokens = sum(r["tokens"] for r in results)
        processing_tps = total_tokens / wall_time

        # 输出结果
        print("\n" + "=" * 80)
        print("📊 Processing TPS 结果")
        print("=" * 80)

        print(f"\n总请求数: {num_requests}")
        print(f"总 tokens: {total_tokens}")
        print(f"总 walltime: {wall_time:.2f}s")
        print(f"\n🎯 Processing TPS: {processing_tps:.1f} tok/s")

        # 显示每个请求的详情
        print("\n单个请求详情:")
        for r in results:
            tps = r["tokens"] / r["time_s"] if r["time_s"] > 0 else 0
            print(f"  {r['request_id']}: {r['tokens']} tokens in {r['time_s']:.2f}s ({tps:.1f} tok/s)")

        # 提示检查日志
        print("\n" + "=" * 80)
        print("📋 检查优化日志:")
        print("  Phase 3 队列延迟: grep 'queue latency' /tmp/omlx-server-profiling.log")
        print("  Phase 4 批量 eval: grep 'Phase 4' /tmp/omlx-server-profiling.log")
        print("=" * 80)

    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
