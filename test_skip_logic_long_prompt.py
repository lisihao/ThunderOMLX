"""使用长 prompt 验证 Skip Logic

使用 116+ token 的长 prompt，验证 Skip Logic 是否能触发。
"""
import asyncio
import logging
import time
from pathlib import Path

from mlx_lm import load

# 设置日志级别为 INFO，确保能看到 Skip Logic 日志
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


async def test_skip_logic_with_long_prompt():
    """使用长 prompt 测试 Skip Logic"""

    print("=" * 70)
    print("Skip Logic 验证（长 prompt）")
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

    block_size = 256  # ⚡ 优化后的 block_size=256: 116/256 = 0.45 blocks，更少碎片化
    model_name_str = str(model_path)

    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        paged_cache_block_size=block_size,
        disable_block_size_enlargement=True,  # 禁用自动提升
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_skip_long"),
        model_name=model_name_str,  # ⚡ 添加 model_name
    )

    engine_config = EngineConfig(
        model_name=model_name_str,  # ⚡ 添加 model_name
        scheduler_config=scheduler_config,
    )
    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

    print(f"\n配置:")
    print(f"  block_size: {engine.scheduler.config.paged_cache_block_size}")

    # ⚠️ 检查 BlockAwarePrefixCache 是否初始化
    if engine.scheduler.block_aware_cache is not None:
        print(f"  ✅ BlockAwarePrefixCache 已启用")
    else:
        print(f"  ❌ BlockAwarePrefixCache 未启用（Skip Logic 不会生效）")
        print(f"  原因: 可能需要在 EngineConfig 中设置 model_name")
        return None

    await engine.start()

    # 3. 测试推理（使用长 prompt）
    print(f"\n[开始测试推理]")

    # 长 prompt (116 tokens)
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

    # 验证 token 数量
    tokens = tokenizer.encode(base_prompt + "人工智能")
    print(f"Prompt token 数量: {len(tokens)}")
    print(f"可创建 block 数: {len(tokens) // block_size}")

    test_cases = [
        {"name": "第 1 次", "prompt": f"{base_prompt}人工智能"},
        {"name": "第 2 次（100% 重复）", "prompt": f"{base_prompt}人工智能"},
        {"name": "第 3 次（95% 重复）", "prompt": f"{base_prompt}机器学习"},
        {"name": "第 4 次（新 prompt）", "prompt": "今天天气怎么样？"},
    ]

    sampling_params = SamplingParams(max_tokens=1)  # ⚡ 只生成 1 token，测试纯 FULL SKIP 性能

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"[{test_case['name']}]")

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
    print(f"性能汇总")
    print(f"{'=' * 70}")

    baseline = results[0]["time_ms"] if results[0]["time_ms"] else 0

    for i, result in enumerate(results):
        if result["time_ms"]:
            speedup = baseline / result["time_ms"] if result["time_ms"] > 0 else 0
            print(f"{i+1}. {result['name']}: {result['time_ms']:.2f} ms ({speedup:.2f}x)")

    # 检查 Skip Logic
    print(f"\n⚠️ 查看日志中是否有 'APPROXIMATE SKIP' 或 'FULL SKIP'")

    return results


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("Skip Logic 验证测试（长 prompt）")
    print("🚀" * 35)

    try:
        results = asyncio.run(test_skip_logic_with_long_prompt())

        if results:
            print("\n\n" + "✅" * 35)
            print("测试完成")
            print("✅" * 35)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
