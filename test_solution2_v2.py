"""方案 2 性能测试 v2 - 修正版

改进点：
1. 使用小 block_size（64）确保短 prompt 能创建 block
2. 使用小模型（如果可用）
3. 验证 Skip Logic 是否触发
"""
import asyncio
import time
from pathlib import Path

from mlx_lm import load


async def test_with_block_size(block_size: int, scenario_name: str):
    """测试指定 block_size 的性能"""

    print("=" * 70)
    print(f"{scenario_name}（block_size={block_size}）")
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
        disable_block_size_enlargement=True,  # 禁用自动提升，保持 block_size
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
    print(f"{scenario_name} 性能汇总")
    print(f"{'=' * 70}")

    baseline = results[0]["time_ms"] if results[0]["time_ms"] else 0

    for i, result in enumerate(results):
        if result["time_ms"]:
            speedup = baseline / result["time_ms"] if result["time_ms"] > 0 else 0
            print(f"{i+1}. {result['name']}: {result['time_ms']:.2f} ms ({speedup:.2f}x)")

    return results


async def main():
    """主测试"""

    print("\n" + "🚀" * 35)
    print("方案 2 性能测试 v2（修正版）")
    print("🚀" * 35)

    # 测试 3 个 block_size
    scenarios = [
        (1024, "场景 A: block_size=1024（默认）"),
        (256, "场景 B: block_size=256（方案 2 推荐）"),
        (64, "场景 C: block_size=64（更激进）"),
    ]

    all_results = {}

    for block_size, scenario_name in scenarios:
        print(f"\n📍 测试 {scenario_name}...")
        results = await test_with_block_size(block_size, scenario_name)
        all_results[block_size] = results

        # 短暂休息，避免内存问题
        await asyncio.sleep(2)

    # 对比分析
    print("\n\n" + "=" * 70)
    print("三场景对比分析")
    print("=" * 70)

    print(f"\n{'场景':<30} {'bs=1024':<15} {'bs=256':<15} {'bs=64':<15}")
    print("─" * 70)

    for i in range(4):
        name = all_results[1024][i]["name"] if all_results[1024] else f"测试 {i+1}"

        time_1024 = all_results[1024][i]["time_ms"] if all_results.get(1024) else None
        time_256 = all_results[256][i]["time_ms"] if all_results.get(256) else None
        time_64 = all_results[64][i]["time_ms"] if all_results.get(64) else None

        print(f"{name:<30}", end="")

        for t in [time_1024, time_256, time_64]:
            if t:
                print(f" {t:>10.2f} ms", end="")
            else:
                print(f" {'N/A':>10}", end="")

        print()

    # 分析重复场景
    print(f"\n{'=' * 70}")
    print("重复场景（第 2-3 次）平均性能")
    print(f"{'=' * 70}")

    for block_size in [1024, 256, 64]:
        if all_results.get(block_size):
            repeat_times = [
                all_results[block_size][1]["time_ms"],
                all_results[block_size][2]["time_ms"],
            ]

            if all(t is not None for t in repeat_times):
                avg_time = sum(repeat_times) / len(repeat_times)
                print(f"block_size={block_size:>4}: {avg_time:.2f} ms 平均")


if __name__ == "__main__":
    try:
        asyncio.run(main())

        print("\n\n" + "✅" * 35)
        print("测试完成")
        print("✅" * 35)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
