"""测试方案 2: 动态 block_size 选择

验证智能 block_size 选择逻辑是否正确工作。
"""
import asyncio
from pathlib import Path

from mlx_lm import load


async def test_dynamic_block_size():
    """测试动态 block_size 选择"""

    print("=" * 70)
    print("方案 2: 动态 block_size 选择测试")
    print("=" * 70)

    # 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    if not model_path.exists():
        print(f"\n❌ 模型不存在: {model_path}")
        return False

    print(f"\n[加载模型]")
    print(f"  路径: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(str(model_path))
    print(f"  ✅ 模型加载成功")

    # 导入 EngineCore
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 智能选择（初始 block_size=32）",
            "config": SchedulerConfig(
                paged_cache_block_size=32,  # 小 block_size
                arrays_cache_target_block_size=None,  # 智能选择
                paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_dynamic_1"),
            ),
            "expected_target": 256,  # current < 128 → 256
        },
        {
            "name": "场景 2: 智能选择（初始 block_size=128）",
            "config": SchedulerConfig(
                paged_cache_block_size=128,  # 中小 block_size
                arrays_cache_target_block_size=None,  # 智能选择
                paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_dynamic_2"),
            ),
            "expected_target": 512,  # current 128-255 → 512
        },
        {
            "name": "场景 3: 智能选择（初始 block_size=256）",
            "config": SchedulerConfig(
                paged_cache_block_size=256,  # 中等 block_size
                arrays_cache_target_block_size=None,  # 智能选择
                paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_dynamic_3"),
            ),
            "expected_target": 1024,  # current >= 256 → 1024
        },
        {
            "name": "场景 4: 用户指定 block_size=64",
            "config": SchedulerConfig(
                paged_cache_block_size=32,  # 初始值（会被覆盖）
                arrays_cache_target_block_size=64,  # 用户指定
                paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_dynamic_4"),
            ),
            "expected_target": 64,  # 用户指定
        },
        {
            "name": "场景 5: 用户指定 block_size=1024",
            "config": SchedulerConfig(
                paged_cache_block_size=32,  # 初始值（会被覆盖）
                arrays_cache_target_block_size=1024,  # 用户指定
                paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_dynamic_5"),
            ),
            "expected_target": 1024,  # 用户指定
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"[测试 {i}/{len(test_cases)}] {test_case['name']}")
        print(f"{'=' * 70}")

        config = test_case['config']
        expected = test_case['expected_target']

        print(f"\n配置:")
        print(f"  初始 block_size: {config.paged_cache_block_size}")
        print(f"  目标设置: {config.arrays_cache_target_block_size}")
        print(f"  预期最终值: {expected}")

        # 创建 EngineCore（会触发 _enlarge_block_size_for_arrays_cache）
        engine_config = EngineConfig(scheduler_config=config)
        engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

        # 检查最终的 block_size
        actual = engine.scheduler.config.paged_cache_block_size

        print(f"\n结果:")
        print(f"  实际最终值: {actual}")

        if actual == expected:
            print(f"  ✅ 测试通过")
            results.append(True)
        else:
            print(f"  ❌ 测试失败（预期 {expected}，实际 {actual}）")
            results.append(False)

        # 清理（删除 engine 对象，释放资源）
        del engine

    # 总结
    print(f"\n{'=' * 70}")
    print(f"测试总结")
    print(f"{'=' * 70}")

    passed = sum(results)
    total = len(results)

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print(f"\n✅ 所有测试通过！")
        return True
    else:
        print(f"\n❌ 有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("ThunderOMLX - 动态 block_size 测试")
    print("🚀" * 35)

    try:
        success = asyncio.run(test_dynamic_block_size())

        if success:
            print("\n" + "✅" * 35)
            print("测试完成")
            print("✅" * 35)
        else:
            print("\n" + "⚠️" * 35)
            print("测试未完全通过")
            print("⚠️" * 35)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
