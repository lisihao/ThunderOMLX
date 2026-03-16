#!/usr/bin/env python3
"""
Metal 并发压力测试 - 限制版本（只测试 2x64 和 4x64）
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def stress_test(num_requests: int, tokens_per_request: int):
    """压力测试"""
    from omlx.engine.batched import BatchedEngine

    print(f"\n{'=' * 80}")
    print(f"🧪 压力测试")
    print(f"{'=' * 80}")
    print(f"  并发请求: {num_requests}")
    print(f"  每请求tokens: {tokens_per_request}")
    print(f"  总tokens: {num_requests * tokens_per_request}")
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        base_prompts = [
            "Explain the key differences between Python and JavaScript in detail.",
            "What are the main advantages of using TypeScript over JavaScript?",
            "Describe the most important design patterns in software engineering.",
            "Explain how asynchronous programming works in Python with examples."
        ]

        prompts = (base_prompts * ((num_requests // len(base_prompts)) + 1))[:num_requests]

        async def single_request(prompt: str, request_id: str):
            tokens = 0
            try:
                async for output in engine.stream_generate(
                    prompt=prompt,
                    max_tokens=tokens_per_request,
                    temperature=0.0
                ):
                    if output.new_text:
                        tokens += 1
                        if tokens % 20 == 0:
                            print(f"  {request_id}: {tokens} tokens", end="\r")
                print(f"✅ {request_id}: {tokens} tokens completed")
                return tokens
            except Exception as e:
                print(f"❌ {request_id} failed: {e}")
                raise

        print("🚀 开始并发执行...")
        tasks = [
            single_request(prompts[i], f"R{i+1}")
            for i in range(num_requests)
        ]

        import time
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start

        success_count = 0
        total_tokens = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Request R{i+1} failed: {result}")
            else:
                success_count += 1
                total_tokens += result

        print(f"\n📊 结果:")
        print(f"  成功: {success_count}/{num_requests}")
        print(f"  总tokens: {total_tokens}")
        print(f"  总时间: {elapsed:.2f}s")
        print(f"  Processing TPS: {total_tokens / elapsed:.1f} tok/s")

        return success_count == num_requests

    except Exception as e:
        print(f"\n❌ 压力测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()


async def main():
    print("=" * 80)
    print("🔍 Metal 并发压力测试（限制版）")
    print("=" * 80)
    print("")

    # 只测试前两个场景
    test_configs = [
        (2, 64),    # 已知稳定
        (4, 64),    # 需要验证
    ]

    results = {}

    for num_requests, tokens_per_request in test_configs:
        test_name = f"{num_requests}x{tokens_per_request}"

        print(f"\n{'=' * 80}")
        print(f"测试配置: {test_name}")
        print(f"{'=' * 80}")

        success = await stress_test(num_requests, tokens_per_request)
        results[test_name] = success

        if not success:
            print(f"\n⚠️  测试 {test_name} 失败，停止后续测试")
            break

        await asyncio.sleep(2)

    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {test_name}: {status}")

    all_success = all(results.values())
    if all_success:
        print("\n✅ 所有测试通过！Metal 并发问题已修复。")
    else:
        failed = [name for name, success in results.items() if not success]
        print(f"\n⚠️  失败测试: {', '.join(failed)}")


if __name__ == "__main__":
    asyncio.run(main())
