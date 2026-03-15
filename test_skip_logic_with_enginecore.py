#!/usr/bin/env python3
"""真正的 Skip Logic 测试 - 使用 ThunderOMLX EngineCore

测试目标:
验证 ThunderOMLX 的 Skip Logic（P0-1 Full Skip 和 P0-2 Approximate Skip）
在真实推理场景下的效果。

关键区别:
- ✅ 使用 EngineCore（会触发 Skip Logic）
- ❌ 不使用 mlx_lm.generate()（只会触发 MLX 系统预热）

测试流程:
1. 加载模型（Qwen 3.5 35B）
2. 创建 EngineCore（集成 BlockAwarePrefixCache）
3. 启动引擎（异步循环）
4. 执行 4 次推理测试：
   - 第一次：完整 prefill（缓存未命中）
   - 第二次：100% 前缀命中（应触发 Full Skip）
   - 第三次：~80% 前缀命中（应触发 Approximate Skip）
   - 第四次：无前缀匹配（完整 prefill）
5. 分析缓存命中日志和性能数据
6. 关闭引擎

预期结果（如果 Skip Logic 生效）:
- 第二次推理：~100ms（跳过 prefill，8x 快）
- 第三次推理：~300ms（部分跳过，2.7x 快）
- 第四次推理：~800ms（完整 prefill，和第一次类似）
"""

import asyncio
import logging
import time
from pathlib import Path

import mlx.core as mx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable详细日志 to debug Skip Logic
logging.getLogger('omlx.scheduler').setLevel(logging.DEBUG)  # 启用 DEBUG
logging.getLogger('omlx.engine_core').setLevel(logging.INFO)
logging.getLogger('omlx.cache.prefix_cache').setLevel(logging.DEBUG)  # 启用 DEBUG


async def test_skip_logic_with_enginecore():
    """使用 EngineCore 测试 Skip Logic 真实效果"""

    print("=" * 70)
    print("Skip Logic 真实测试（使用 EngineCore）")
    print("=" * 70)

    # 1. 使用小模型（从 HF Hub 加载）
    model_path = Path.home() / "models"

    # 先尝试本地模型，再尝试 HF Hub
    candidate_models = [
        ("qwen3.5-35b-mlx", True),  # 本地模型
        ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", False),  # HF Hub
        ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", False),  # HF Hub
    ]

    available_model = None
    is_local = False

    # 优先使用本地小模型（避免下载）
    for model_name, local in candidate_models:
        if local:
            model_dir = model_path / model_name
            if model_dir.exists():
                available_model = str(model_dir)
                is_local = True
                print(f"\n✅ 找到本地模型: {model_name}")
                break
        else:
            # HF Hub 模型（会自动下载）
            available_model = model_name
            is_local = False
            print(f"\n✅ 使用 HF Hub 模型: {model_name}（会自动下载）")
            break

    if not available_model:
        # 如果本地模型太大（如 35B），使用 HF Hub 小模型
        print("\n⚠️ 未找到本地小模型，使用 HF Hub 模型（需要下载）")
        available_model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        is_local = False

    # 2. 加载模型
    print(f"\n[加载模型]")
    print(f"  路径: {available_model}")

    start = time.perf_counter()

    # 使用 mlx_lm 加载模型（EngineCore 需要 mlx-lm 格式的模型）
    from mlx_lm import load
    model, tokenizer = load(available_model)

    load_time = time.perf_counter() - start
    print(f"  加载时间: {load_time:.2f}s")

    # 3. 创建 EngineCore（会自动初始化 BlockAwarePrefixCache）
    print(f"\n[创建 EngineCore]")

    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    # 配置 scheduler（启用 prefix caching）
    # ⚡ 使用优化后的 block_size=256（平衡缓存命中率和碎片化）
    # 256 tokens/block 可以有效缓存 140-1000 token 范围的 prompt
    scheduler_config = SchedulerConfig(
        max_num_seqs=2,  # 降低并发数
        paged_cache_block_size=256,  # ⚡ 优化后的值（修复碎片化问题）
        max_cache_blocks=1024,  # 限制 KV Cache 块数（避免大模型 OOM）
        initial_cache_blocks=128,  # 降低初始块数
        disable_block_size_enlargement=True,  # ⚡ 禁用自动提升，允许用 block_size=32 测试 Skip Logic
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_skip_logic"),
        model_name=available_model,
    )

    engine_config = EngineConfig(
        model_name=available_model,
        scheduler_config=scheduler_config,
    )

    start = time.perf_counter()
    engine = EngineCore(
        model=model,
        tokenizer=tokenizer,
        config=engine_config,
    )
    engine_init_time = (time.perf_counter() - start) * 1000
    print(f"  引擎初始化时间: {engine_init_time:.2f} ms")

    # 检查 BlockAwarePrefixCache 是否初始化
    if engine.scheduler.block_aware_cache is not None:
        print(f"  ✅ BlockAwarePrefixCache 已启用")
        print(f"  Block size: {engine.scheduler.config.paged_cache_block_size}")
        print(f"  Cache dir: {engine.scheduler.config.paged_ssd_cache_dir}")
    else:
        print(f"  ❌ BlockAwarePrefixCache 未启用（Skip Logic 不会生效）")
        engine.close()
        return False

    # 4. 启动引擎
    print(f"\n[启动引擎]")
    await engine.start()
    print(f"  ✅ 引擎已启动")

    # 5. 准备测试用例
    # ⚠️ 关键：使用长 prompt（> 1024 tokens）才能触发缓存
    # 因为 ArraysCache 模型会自动把 block_size 提升到 1024
    base_prompt = """请详细解释以下概念，包括定义、历史发展、核心技术、应用场景、优缺点和未来趋势。
请从以下几个方面进行阐述：
1. 基本定义和核心概念
2. 发展历史和重要里程碑
3. 核心技术和实现原理
4. 主要应用场景和实际案例
5. 当前的优势和局限性
6. 未来发展趋势和挑战
7. 与相关概念的区别和联系
8. 实际应用中的最佳实践
9. 常见问题和解决方案
10. 学习路径和推荐资源

请详细阐述："""

    test_cases = [
        {
            "name": "第一次推理（完整 prefill）",
            "prompt": f"{base_prompt}人工智能的定义、发展历史、核心技术、应用场景、优势和局限性、未来趋势等方面",
            "expected_skip": False,
            "expected_cache_hit_ratio": 0.0,
        },
        {
            "name": "第二次推理（100% 前缀命中）",
            "prompt": f"{base_prompt}人工智能的定义、发展历史、核心技术、应用场景、优势和局限性、未来趋势等方面",
            "expected_skip": "full",
            "expected_cache_hit_ratio": 1.0,
        },
        {
            "name": "第三次推理（~80% 前缀命中）",
            "prompt": f"{base_prompt}机器学习的定义、发展历史、核心技术、应用场景、优势和局限性、未来趋势等方面",
            "expected_skip": "approximate",
            "expected_cache_hit_ratio": 0.8,
        },
        {
            "name": "第四次推理（无前缀匹配）",
            "prompt": "今天天气怎么样？明天会下雨吗？周末适合出游吗？",
            "expected_skip": False,
            "expected_cache_hit_ratio": 0.0,
        },
    ]

    # 6. 执行推理测试
    results = []

    # ⚠️ 关键：max_tokens 需要足够大，让 prompt + output > 1024 tokens
    # 当前 prompt ~180 tokens，所以需要至少生成 850 tokens
    sampling_params = SamplingParams(
        max_tokens=900,  # 增加到 900，确保总 tokens > 1024
        temperature=0.0,  # 确定性输出，方便对比
    )

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"[测试 {i}/{len(test_cases)}] {test_case['name']}")
        print(f"{'=' * 70}")
        print(f"Prompt: \"{test_case['prompt']}\"")
        print(f"预期: {test_case['expected_skip'] or '完整 prefill'}")

        # 推理
        start = time.perf_counter()

        try:
            output = await engine.generate(
                prompt=test_case['prompt'],
                sampling_params=sampling_params,
            )

            inference_time = (time.perf_counter() - start) * 1000  # ms

            print(f"\n生成结果 ({len(output.output_text)} chars):")
            print(f"  {output.output_text[:100]}...")
            print(f"\n推理时间: {inference_time:.2f} ms")

            # 获取缓存统计
            if engine.scheduler.block_aware_cache:
                cache = engine.scheduler.block_aware_cache
                stats = cache.get_stats()

                print(f"\n缓存统计:")
                print(f"  总查询: {stats.get('total_queries', 0)}")
                print(f"  Full Skip: {stats.get('full_skip_count', 0)}")
                print(f"  Approximate Skip: {stats.get('approximate_skip_count', 0)}")
                print(f"  No Skip: {stats.get('no_skip_count', 0)}")

                # 计算本次查询的缓存命中情况
                # 注意：这里的统计是累计的，需要和上一次对比

            results.append({
                "name": test_case["name"],
                "prompt": test_case["prompt"],
                "time_ms": inference_time,
                "expected_skip": test_case["expected_skip"],
                "output_length": len(output.output_text),
            })

        except Exception as e:
            print(f"\n❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": test_case["name"],
                "prompt": test_case["prompt"],
                "time_ms": 0,
                "expected_skip": test_case["expected_skip"],
                "error": str(e),
            })

    # 7. 分析结果
    print(f"\n{'=' * 70}")
    print("性能分析")
    print(f"{'=' * 70}")

    print(f"\n推理时间对比:")
    baseline_time = results[0]["time_ms"]

    for i, result in enumerate(results):
        if result["time_ms"] == 0:
            print(f"\n{i+1}. {result['name']}")
            print(f"   ❌ 推理失败: {result.get('error', 'Unknown error')}")
            continue

        speedup = baseline_time / result["time_ms"] if result["time_ms"] > 0 else 0

        expected = result["expected_skip"]
        if expected == "full":
            skip_status = "✅ 应触发 Full Skip（100% 命中）"
        elif expected == "approximate":
            skip_status = "✅ 应触发 Approximate Skip（~80% 命中）"
        elif expected == False:
            skip_status = "⏭️ 完整 prefill（无缓存）"
        else:
            skip_status = f"预期: {expected}"

        print(f"\n{i+1}. {result['name']}")
        print(f"   时间: {result['time_ms']:.2f} ms")
        print(f"   加速: {speedup:.2f}x (vs baseline)")
        print(f"   预期: {skip_status}")

    # 8. Skip Logic 效果验证
    print(f"\n{'=' * 70}")
    print("Skip Logic 效果验证")
    print(f"{'=' * 70}")

    if len(results) >= 4 and all(r["time_ms"] > 0 for r in results):
        first_time = results[0]["time_ms"]
        second_time = results[1]["time_ms"]
        fourth_time = results[3]["time_ms"]

        full_skip_speedup = first_time / second_time if second_time > 0 else 0

        print(f"\n第一次推理（完整 prefill）: {first_time:.2f} ms")
        print(f"第二次推理（100% 命中）: {second_time:.2f} ms")
        print(f"第四次推理（无匹配）: {fourth_time:.2f} ms")
        print(f"\nFull Skip 加速比: {full_skip_speedup:.2f}x")

        # 关键判断：第二次和第四次的时间应该显著不同
        time_diff_ratio = abs(second_time - fourth_time) / fourth_time

        print(f"\n关键验证：")
        print(f"  第二次（100% 命中）vs 第四次（无匹配）时间差异: {time_diff_ratio * 100:.1f}%")

        if time_diff_ratio > 0.5:  # 时间差异 > 50%
            print(f"  ✅ Skip Logic 生效！（时间差异显著）")
            if full_skip_speedup > 5.0:
                print(f"  ✅ Full Skip 效果优秀！加速 {full_skip_speedup:.2f}x")
            elif full_skip_speedup > 2.0:
                print(f"  ⚠️ Full Skip 部分生效，加速 {full_skip_speedup:.2f}x")
            else:
                print(f"  ⚠️ Full Skip 加速不明显 ({full_skip_speedup:.2f}x)")
        else:
            print(f"  ❌ Skip Logic 可能未生效（时间差异不显著）")
            print(f"     这可能是因为：")
            print(f"     1. BlockAwarePrefixCache 未正确初始化")
            print(f"     2. 缓存命中逻辑未生效")
            print(f"     3. 测试 prompt 太短，Skip 效果不明显")

    # 9. 显示最终缓存统计
    if engine.scheduler.block_aware_cache:
        print(f"\n{'=' * 70}")
        print("最终缓存统计")
        print(f"{'=' * 70}")

        cache = engine.scheduler.block_aware_cache
        stats = cache.get_stats()

        print(f"\n总查询次数: {stats.get('total_queries', 0)}")
        print(f"Full Skip: {stats.get('full_skip_count', 0)} 次")
        print(f"Approximate Skip: {stats.get('approximate_skip_count', 0)} 次")
        print(f"No Skip: {stats.get('no_skip_count', 0)} 次")

        total_skips = stats.get('full_skip_count', 0) + stats.get('approximate_skip_count', 0)
        total_queries = stats.get('total_queries', 0)
        skip_rate = total_skips / total_queries * 100 if total_queries > 0 else 0

        print(f"\nSkip 率: {skip_rate:.1f}%")

    # 10. 关闭引擎
    print(f"\n[关闭引擎]")
    await engine.stop()
    engine.close()
    print(f"  ✅ 引擎已关闭")

    return True


async def main():
    """主函数"""
    print("\n" + "🚀" * 35)
    print("ThunderOMLX - Skip Logic 真实验证测试")
    print("🚀" * 35)

    try:
        success = await test_skip_logic_with_enginecore()

        if success:
            print("\n" + "✅" * 35)
            print("测试完成")
            print("✅" * 35)
        else:
            print("\n" + "⚠️" * 35)
            print("测试未完成")
            print("⚠️" * 35)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(main())
