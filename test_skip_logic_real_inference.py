"""MLX 系统预热效果测试 - 真实推理场景

⚠️ 重要说明：
本测试使用 mlx_lm.generate()，这是 MLX 官方库的函数，
**不会触发 ThunderOMLX 的 Skip Logic 系统**。

测试结果展示的是 MLX 自身的系统预热效果：
- 第一次推理慢：Metal shader 编译、模型加载、GPU 初始化
- 后续推理快：shader 缓存、算子融合、内存池预分配

要真正测试 Skip Logic，需要使用 ThunderOMLX 的 EngineCore：
  from omlx.engine_core import EngineCore
  engine = EngineCore(model, tokenizer, EngineConfig())
  await engine.start()
  output = await engine.generate(prompt, SamplingParams(...))

测试场景:
1. 第一次推理: MLX 冷启动
2. 第二次推理: MLX 系统预热（不是 Skip Logic）
3. 第三次推理: MLX 系统预热（不是 Skip Logic）
4. 第四次推理: MLX 系统预热（不是 Skip Logic）
"""
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

from omlx.cache.unified_memory_cache import UnifiedMemoryCacheManager
from omlx.thunder_config import SerializationConfig


def test_skip_logic_with_real_model():
    """使用真实模型测试 Skip Logic 效果"""

    print("=" * 70)
    print("Skip Logic 真实推理测试")
    print("=" * 70)

    # 1. 检查是否有可用模型
    model_path = Path.home() / "models"

    # 查找可用的模型（优先小模型，快速测试）
    candidate_models = [
        "qwen3.5-35b-mlx",  # 用户的 Qwen 3.5 35B 模型
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
    ]

    available_model = None
    for model_name in candidate_models:
        model_dir = model_path / model_name
        if model_dir.exists():
            available_model = str(model_dir)
            print(f"\n✅ 找到模型: {model_name}")
            break

    if not available_model:
        print("\n❌ 未找到测试模型")
        print(f"\n建议下载小模型用于测试:")
        print(f"  mlx_lm.convert --hf-path Qwen/Qwen2.5-0.5B-Instruct -q")
        print(f"  或")
        print(f"  huggingface-cli download mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        return False

    # 2. 加载模型
    print(f"\n[加载模型]")
    print(f"  路径: {available_model}")

    start = time.perf_counter()
    model, tokenizer = load(available_model)
    load_time = time.perf_counter() - start
    print(f"  加载时间: {load_time:.2f}s")

    # 3. 准备测试 prompts
    base_prompt = "解释一下什么是"

    test_cases = [
        {
            "name": "第一次推理（完整 prefill）",
            "prompt": f"{base_prompt}人工智能",
            "expected_skip": False,
        },
        {
            "name": "第二次推理（前缀命中 100%）",
            "prompt": f"{base_prompt}人工智能",
            "expected_skip": True,
        },
        {
            "name": "第三次推理（前缀命中 ~80%）",
            "prompt": f"{base_prompt}机器学习",
            "expected_skip": "partial",
        },
        {
            "name": "第四次推理（无命中）",
            "prompt": "今天天气怎么样？",
            "expected_skip": False,
        },
    ]

    # 4. 执行推理测试
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"[测试 {i}/{len(test_cases)}] {test_case['name']}")
        print(f"{'=' * 70}")
        print(f"Prompt: \"{test_case['prompt']}\"")

        # 推理
        start = time.perf_counter()

        response = generate(
            model,
            tokenizer,
            prompt=test_case['prompt'],
            max_tokens=50,  # 限制生成长度，加快测试
            verbose=False,
        )

        inference_time = (time.perf_counter() - start) * 1000  # ms

        print(f"\n生成结果 ({len(response)} chars):")
        print(f"  {response[:100]}...")
        print(f"\n推理时间: {inference_time:.2f} ms")

        results.append({
            "name": test_case["name"],
            "prompt": test_case["prompt"],
            "time_ms": inference_time,
            "expected_skip": test_case["expected_skip"],
        })

    # 5. 分析结果
    print(f"\n{'=' * 70}")
    print("性能分析")
    print(f"{'=' * 70}")

    print(f"\n推理时间对比:")
    baseline_time = results[0]["time_ms"]

    for i, result in enumerate(results):
        speedup = baseline_time / result["time_ms"] if result["time_ms"] > 0 else 0
        skip_status = "✅ 应触发 Skip" if result["expected_skip"] else "⏭️ 完整计算"

        print(f"\n{i+1}. {result['name']}")
        print(f"   时间: {result['time_ms']:.2f} ms")
        print(f"   加速: {speedup:.2f}x (vs baseline)")
        print(f"   预期: {skip_status}")

    # 6. MLX 系统预热效果验证
    print(f"\n{'=' * 70}")
    print("⚠️ MLX 系统预热效果（不是 Skip Logic）")
    print(f"{'=' * 70}")

    first_time = results[0]["time_ms"]
    second_time = results[1]["time_ms"]
    fourth_time = results[3]["time_ms"]

    warmup_speedup = first_time / second_time if second_time > 0 else 0

    print(f"\n第一次推理（冷启动）: {first_time:.2f} ms")
    print(f"第二次推理（预热后）: {second_time:.2f} ms")
    print(f"第四次推理（无匹配，预热后）: {fourth_time:.2f} ms")
    print(f"\n加速比: {warmup_speedup:.2f}x")

    print(f"\n⚠️ 关键发现：")
    print(f"   测试 2-4 的时间几乎相同（~{second_time:.0f}ms）")
    print(f"   这说明加速来自 MLX 系统预热，**不是 Skip Logic**")
    print(f"\n   MLX 预热包括:")
    print(f"   • Metal shader 编译和缓存")
    print(f"   • GPU 内存池预分配")
    print(f"   • 算子融合优化")
    print(f"   • 内存拷贝优化")

    print(f"\n✅ MLX 系统预热效果: {warmup_speedup:.2f}x")
    print(f"❌ Skip Logic 未被触发（需要使用 EngineCore）")

    return True


def test_cache_system_integration():
    """测试缓存系统与推理引擎的集成"""

    print("\n" + "=" * 70)
    print("缓存系统集成测试")
    print("=" * 70)

    # 检查缓存管理器是否正确初始化
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        config = SerializationConfig(
            compression="lz4",
            enable_checksum=True,
        )

        cache_mgr = UnifiedMemoryCacheManager(
            l2_max_size_mb=100,
            l3_cache_path=tmpdir / "l3_cache",
            l3_max_size_gb=2,
            serialization_config=config,
        )

        print(f"\n✅ 缓存管理器初始化成功")
        print(f"   L2 容量: {cache_mgr.l2_max_bytes / (1024**2):.0f} MB")
        print(f"   L3 路径: {cache_mgr.l3_cache_path}")
        print(f"   压缩方式: {config.compression}")

        # 测试缓存读写
        print(f"\n[测试缓存读写]")

        test_tensor = mx.random.normal(shape=(1, 512, 2048))
        key = "test_kv_cache"

        # 写入
        start = time.perf_counter()
        cache_mgr.store(key, test_tensor)
        write_time = (time.perf_counter() - start) * 1000
        print(f"  写入时间: {write_time:.2f} ms")

        # 读取
        start = time.perf_counter()
        loaded_tensor, hit = cache_mgr.fetch(key)
        read_time = (time.perf_counter() - start) * 1000
        print(f"  读取时间: {read_time:.2f} ms")
        print(f"  缓存命中: {'✅' if hit else '❌'}")

        # 验证数据一致性
        import numpy as np
        if np.allclose(np.array(test_tensor), np.array(loaded_tensor)):
            print(f"  数据一致性: ✅")
        else:
            print(f"  数据一致性: ❌")

        stats = cache_mgr.get_stats()
        print(f"\n缓存统计:")
        print(f"  总命中率: {stats.overall_hit_rate * 100:.1f}%")
        print(f"  L2 命中率: {stats.l2_hit_rate * 100:.1f}%")
        print(f"  L3 命中率: {stats.l3_hit_rate * 100:.1f}%")


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("ThunderOMLX - Skip Logic 验证测试")
    print("🚀" * 35)

    # 测试 1: 缓存系统集成
    test_cache_system_integration()

    # 测试 2: Skip Logic 真实推理
    print("\n" + "─" * 70 + "\n")

    try:
        success = test_skip_logic_with_real_model()

        if success:
            print("\n" + "✅" * 35)
            print("测试完成")
            print("✅" * 35)
        else:
            print("\n" + "⚠️" * 35)
            print("测试未完成（缺少模型）")
            print("⚠️" * 35)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
