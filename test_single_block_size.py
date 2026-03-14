"""单 block_size 性能测试

由于大模型 (35B) GPU 内存限制，每次只测试一个 block_size 配置。

用法:
    python test_single_block_size.py 1024  # 测试 block_size=1024
    python test_single_block_size.py 256   # 测试 block_size=256
    python test_single_block_size.py 64    # 测试 block_size=64
"""
import asyncio
import sys
import time
from pathlib import Path

from mlx_lm import load


async def test_with_block_size(block_size: int):
    """测试指定 block_size 的性能"""

    print("=" * 70)
    print(f"测试 block_size={block_size}")
    print("=" * 70)

    # 1. 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print(f"\n[加载模型]")
    print(f"  路径: {model_path}")

    if not model_path.exists():
        print(f"  ❌ 模型不存在")
        return None

    model, tokenizer = load(str(model_path))
    print(f"  ✅ 模型加载成功")

    # 2. 创建 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        paged_cache_block_size=block_size,
        disable_block_size_enlargement=True,  # 禁用自动提升
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / f"test_bs{block_size}"),
    )

    engine_config = EngineConfig(scheduler_config=scheduler_config)
    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

    print(f"\n配置:")
    print(f"  设置 block_size: {block_size}")
    print(f"  实际 block_size: {engine.scheduler.config.paged_cache_block_size}")

    await engine.start()

    # 3. 测试推理
    print(f"\n[开始测试推理]")

    base_prompt = "解释一下什么是"
    test_cases = [
        {"name": "第 1 次", "prompt": f"{base_prompt}人工智能"},
        {"name": "第 2 次（100% 重复）", "prompt": f"{base_prompt}人工智能"},
        {"name": "第 3 次（~80% 重复）", "prompt": f"{base_prompt}机器学习"},
        {"name": "第 4 次（新 prompt）", "prompt": "今天天气怎么样？"},
    ]

    sampling_params = SamplingParams(max_tokens=50)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"[{test_case['name']}] Prompt: \"{test_case['prompt']}\"")

        start = time.perf_counter()

        try:
            output = await engine.generate(
                prompt=test_case['prompt'],
                sampling_params=sampling_params,
            )

            inference_time = (time.perf_counter() - start) * 1000  # ms

            print(f"  推理时间: {inference_time:.2f} ms")
            print(f"  生成: {output.output_text[:40]}...")

            results.append({
                "name": test_case["name"],
                "time_ms": inference_time,
            })

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append({
                "name": test_case["name"],
                "time_ms": None,
            })

    # 4. 汇总
    print(f"\n{'=' * 70}")
    print(f"block_size={block_size} 性能汇总")
    print(f"{'=' * 70}")

    baseline = results[0]["time_ms"] if results[0]["time_ms"] else 0

    for i, result in enumerate(results):
        if result["time_ms"]:
            speedup = baseline / result["time_ms"] if result["time_ms"] > 0 else 0
            print(f"{i+1}. {result['name']}: {result['time_ms']:.2f} ms ({speedup:.2f}x)")

    # 计算重复场景平均时间
    repeat_times = [r["time_ms"] for r in results[1:3] if r["time_ms"]]
    if repeat_times:
        avg_time = sum(repeat_times) / len(repeat_times)
        print(f"\n重复场景（第 2-3 次）平均: {avg_time:.2f} ms")

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python test_single_block_size.py <block_size>")
        print("示例:")
        print("  python test_single_block_size.py 1024")
        print("  python test_single_block_size.py 256")
        print("  python test_single_block_size.py 64")
        sys.exit(1)

    try:
        block_size = int(sys.argv[1])
    except ValueError:
        print(f"❌ 错误: block_size 必须是整数")
        sys.exit(1)

    print("\n" + "🚀" * 35)
    print(f"ThunderOMLX - 单 block_size 测试 ({block_size})")
    print("🚀" * 35)

    try:
        results = asyncio.run(test_with_block_size(block_size))

        if results:
            print("\n\n" + "✅" * 35)
            print("测试完成")
            print("✅" * 35)

            print("\n提示: 运行其他 block_size 进行对比:")
            if block_size != 1024:
                print(f"  python test_single_block_size.py 1024")
            if block_size != 256:
                print(f"  python test_single_block_size.py 256")
            if block_size != 64:
                print(f"  python test_single_block_size.py 64")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
