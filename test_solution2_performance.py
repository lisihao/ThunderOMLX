"""测试方案 2 的实际性能提升

对比场景：
- 场景 A：默认（block_size=1024，无 Skip Logic）
- 场景 B：方案 2（block_size=256，有 Skip Logic）

测试模型：小模型（避免 GPU OOM）
测试场景：短 prompt agent（< 256 tokens，高重复）
"""
import asyncio
import time
from pathlib import Path

from mlx_lm import load


async def test_scenario_a_baseline():
    """场景 A：默认（block_size=1024，无 Skip Logic）"""

    print("=" * 70)
    print("场景 A：默认配置（block_size 自动提升到 1024）")
    print("=" * 70)

    # 1. 加载小模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 如果大模型存在，提示用户使用小模型
    if model_path.exists():
        print(f"\n⚠️ 检测到大模型 {model_path.name}")
        print(f"⚠️ 为避免 GPU OOM，建议使用小模型（0.5B-3B）")
        print(f"\n尝试查找小模型...")

        # 查找小模型
        candidate_models = [
            "Qwen2.5-0.5B-Instruct-4bit",
            "Qwen2.5-1.5B-Instruct-4bit",
            "Llama-3.2-1B-Instruct-4bit",
            "Llama-3.2-3B-Instruct-4bit",
        ]

        small_model = None
        models_dir = Path.home() / "models"
        for model_name in candidate_models:
            model_dir = models_dir / model_name
            if model_dir.exists():
                small_model = str(model_dir)
                print(f"✅ 找到小模型: {model_name}")
                break

        if small_model:
            model_path = small_model
        else:
            print(f"\n⚠️ 未找到小模型，将使用大模型（可能 OOM）")
            print(f"⚠️ 建议下载小模型：mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    print(f"\n[加载模型]")
    print(f"  路径: {model_path}")

    model, tokenizer = load(str(model_path))
    print(f"  ✅ 模型加载成功")

    # 2. 创建 EngineCore（默认配置，block_size 会自动提升到 1024）
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        paged_cache_block_size=256,  # 初始值 256
        # arrays_cache_target_block_size=None,  # 默认，会自动提升到 1024
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_scenario_a"),
    )

    engine_config = EngineConfig(scheduler_config=scheduler_config)
    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

    print(f"\n配置:")
    print(f"  初始 block_size: 256")
    print(f"  预期最终值: 1024（自动提升）")
    print(f"  实际最终值: {engine.scheduler.config.paged_cache_block_size}")

    await engine.start()

    # 3. 测试推理（短 prompt，重复场景）
    print(f"\n[开始测试推理]")

    base_prompt = "解释一下什么是"
    test_cases = [
        {"name": "第 1 次", "prompt": f"{base_prompt}人工智能", "repeat": False},
        {"name": "第 2 次（100% 重复）", "prompt": f"{base_prompt}人工智能", "repeat": True},
        {"name": "第 3 次（~80% 重复）", "prompt": f"{base_prompt}机器学习", "repeat": True},
        {"name": "第 4 次（新 prompt）", "prompt": "今天天气怎么样？", "repeat": False},
    ]

    sampling_params = SamplingParams(max_tokens=50)  # 短生成（避免 OOM）

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"[{test_case['name']}]")
        print(f"  Prompt: \"{test_case['prompt']}\"")

        start = time.perf_counter()

        try:
            output = await engine.generate(
                prompt=test_case['prompt'],
                sampling_params=sampling_params,
            )

            inference_time = (time.perf_counter() - start) * 1000  # ms

            print(f"  生成结果: {output.output_text[:50]}...")
            print(f"  推理时间: {inference_time:.2f} ms")

            results.append({
                "name": test_case["name"],
                "time_ms": inference_time,
                "is_repeat": test_case["repeat"],
            })

        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            results.append({
                "name": test_case["name"],
                "time_ms": None,
                "is_repeat": test_case["repeat"],
            })

    # 4. 汇总结果
    print(f"\n{'=' * 70}")
    print(f"场景 A 性能汇总")
    print(f"{'=' * 70}")

    baseline_time = results[0]["time_ms"] if results[0]["time_ms"] else 0

    for i, result in enumerate(results):
        if result["time_ms"]:
            speedup = baseline_time / result["time_ms"] if result["time_ms"] > 0 else 0
            print(f"{i+1}. {result['name']}: {result['time_ms']:.2f} ms ({speedup:.2f}x)")
        else:
            print(f"{i+1}. {result['name']}: 失败")

    return results


async def test_scenario_b_solution2():
    """场景 B：方案 2（block_size=256，有 Skip Logic）"""

    print("\n\n" + "=" * 70)
    print("场景 B：方案 2（智能选择 block_size=256）")
    print("=" * 70)

    # 1. 加载小模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 查找小模型
    if model_path.exists():
        candidate_models = [
            "Qwen2.5-0.5B-Instruct-4bit",
            "Qwen2.5-1.5B-Instruct-4bit",
            "Llama-3.2-1B-Instruct-4bit",
            "Llama-3.2-3B-Instruct-4bit",
        ]

        small_model = None
        models_dir = Path.home() / "models"
        for model_name in candidate_models:
            model_dir = models_dir / model_name
            if model_dir.exists():
                small_model = str(model_dir)
                break

        if small_model:
            model_path = small_model

    print(f"\n[加载模型]")
    print(f"  路径: {model_path}")

    model, tokenizer = load(str(model_path))
    print(f"  ✅ 模型加载成功")

    # 2. 创建 EngineCore（方案 2 配置）
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        paged_cache_block_size=32,  # 小初始值
        arrays_cache_target_block_size=256,  # ⚡ 方案 2：显式指定 256
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_scenario_b"),
    )

    engine_config = EngineConfig(scheduler_config=scheduler_config)
    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

    print(f"\n配置:")
    print(f"  初始 block_size: 32")
    print(f"  方案 2 目标值: 256")
    print(f"  实际最终值: {engine.scheduler.config.paged_cache_block_size}")

    await engine.start()

    # 3. 测试推理（短 prompt，重复场景）
    print(f"\n[开始测试推理]")

    base_prompt = "解释一下什么是"
    test_cases = [
        {"name": "第 1 次", "prompt": f"{base_prompt}人工智能", "repeat": False},
        {"name": "第 2 次（100% 重复）", "prompt": f"{base_prompt}人工智能", "repeat": True},
        {"name": "第 3 次（~80% 重复）", "prompt": f"{base_prompt}机器学习", "repeat": True},
        {"name": "第 4 次（新 prompt）", "prompt": "今天天气怎么样？", "repeat": False},
    ]

    sampling_params = SamplingParams(max_tokens=50)  # 短生成（避免 OOM）

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"[{test_case['name']}]")
        print(f"  Prompt: \"{test_case['prompt']}\"")

        start = time.perf_counter()

        try:
            output = await engine.generate(
                prompt=test_case['prompt'],
                sampling_params=sampling_params,
            )

            inference_time = (time.perf_counter() - start) * 1000  # ms

            print(f"  生成结果: {output.output_text[:50]}...")
            print(f"  推理时间: {inference_time:.2f} ms")

            results.append({
                "name": test_case["name"],
                "time_ms": inference_time,
                "is_repeat": test_case["repeat"],
            })

        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
            results.append({
                "name": test_case["name"],
                "time_ms": None,
                "is_repeat": test_case["repeat"],
            })

    # 4. 汇总结果
    print(f"\n{'=' * 70}")
    print(f"场景 B 性能汇总")
    print(f"{'=' * 70}")

    baseline_time = results[0]["time_ms"] if results[0]["time_ms"] else 0

    for i, result in enumerate(results):
        if result["time_ms"]:
            speedup = baseline_time / result["time_ms"] if result["time_ms"] > 0 else 0
            print(f"{i+1}. {result['name']}: {result['time_ms']:.2f} ms ({speedup:.2f}x)")
        else:
            print(f"{i+1}. {result['name']}: 失败")

    return results


async def main():
    """主测试流程"""

    print("\n" + "🚀" * 35)
    print("方案 2 性能提升测试")
    print("🚀" * 35)

    # 场景 A：基线（block_size=1024）
    print("\n📍 开始测试场景 A（基线）...")
    results_a = await test_scenario_a_baseline()

    # 场景 B：方案 2（block_size=256）
    print("\n📍 开始测试场景 B（方案 2）...")
    results_b = await test_scenario_b_solution2()

    # 对比分析
    print("\n\n" + "=" * 70)
    print("场景对比分析")
    print("=" * 70)

    print(f"\n{'场景':<30} {'场景 A':<15} {'场景 B':<15} {'提升':<10}")
    print("─" * 70)

    for i in range(len(results_a)):
        name = results_a[i]["name"]
        time_a = results_a[i]["time_ms"]
        time_b = results_b[i]["time_ms"]

        if time_a and time_b:
            improvement = ((time_a - time_b) / time_a) * 100
            speedup = time_a / time_b

            status = "✅" if improvement > 5 else "⚠️" if improvement > 0 else "❌"

            print(f"{name:<30} {time_a:>10.2f} ms {time_b:>10.2f} ms {status} {improvement:>6.1f}% ({speedup:.2f}x)")
        else:
            print(f"{name:<30} {'失败':<15} {'失败':<15} {'N/A':<10}")

    # 重复场景汇总
    print(f"\n{'=' * 70}")
    print("重复场景性能提升汇总")
    print(f"{'=' * 70}")

    repeat_improvements = []
    for i in range(len(results_a)):
        if results_a[i]["is_repeat"] and results_a[i]["time_ms"] and results_b[i]["time_ms"]:
            improvement = ((results_a[i]["time_ms"] - results_b[i]["time_ms"]) / results_a[i]["time_ms"]) * 100
            repeat_improvements.append(improvement)

    if repeat_improvements:
        avg_improvement = sum(repeat_improvements) / len(repeat_improvements)
        print(f"\n平均性能提升: {avg_improvement:.1f}%")
        print(f"重复场景数量: {len(repeat_improvements)}")

        if avg_improvement > 20:
            print(f"\n✅ 方案 2 在重复场景中有显著性能提升！")
        elif avg_improvement > 10:
            print(f"\n✅ 方案 2 在重复场景中有明显性能提升")
        elif avg_improvement > 0:
            print(f"\n⚠️ 方案 2 性能提升较小")
        else:
            print(f"\n❌ 方案 2 性能反而下降（可能是测试误差）")
    else:
        print(f"\n⚠️ 无法计算性能提升（测试失败）")


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
