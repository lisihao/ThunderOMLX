#!/usr/bin/env python3
"""
Metal 并发问题调试 - 逐步测试各个 Phase

目标：找出导致 Metal 错误的具体 Phase
方法：创建多个配置，逐个启用 Phase
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_scenario(scenario_name: str, enable_phase1: bool, enable_phase2: bool, enable_phase4: bool):
    """测试特定配置"""
    from omlx.engine.batched import BatchedEngine

    print(f"\n{'=' * 80}")
    print(f"🧪 测试场景: {scenario_name}")
    print(f"{'=' * 80}")
    print(f"  Phase 1 (异步 tensor 提取): {'✅ 启用' if enable_phase1 else '❌ 禁用'}")
    print(f"  Phase 2 (异步 save_block):  {'✅ 启用' if enable_phase2 else '❌ 禁用'}")
    print(f"  Phase 4 (批量 Metal eval):  {'✅ 启用' if enable_phase4 else '❌ 禁用'}")
    print("")

    # TODO: 需要能够通过参数控制各个 Phase 的启用/禁用
    # 目前代码中这些 Phase 是硬编码的
    # 先运行测试，观察哪里出错

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        # 运行2个并发请求（简化测试）
        prompts = [
            "Explain the key differences between Python and JavaScript.",
            "What are the main advantages of using TypeScript?"
        ]

        async def single_request(prompt: str, request_id: str):
            tokens = 0
            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=64,  # 减少tokens以加快测试
                temperature=0.0
            ):
                if output.new_text:
                    tokens += 1
            return tokens

        # 并发执行
        tasks = [
            single_request(prompts[0], "R1"),
            single_request(prompts[1], "R2")
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 检查结果
        success = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Request R{i+1} failed: {result}")
                success = False
            else:
                print(f"✅ Request R{i+1} completed: {result} tokens")

        if success:
            print(f"\n🎉 {scenario_name} - 成功!")
            return True
        else:
            print(f"\n❌ {scenario_name} - 失败!")
            return False

    except Exception as e:
        print(f"\n❌ {scenario_name} - 异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()


async def main():
    print("=" * 80)
    print("🔍 Metal 并发问题调试")
    print("=" * 80)
    print("")
    print("目标：找出导致 Metal 错误的具体 Phase")
    print("方法：逐步启用各个 Phase，观察哪个导致错误")
    print("")

    # 测试场景列表
    scenarios = [
        ("Baseline (所有优化禁用)", False, False, False),
        ("仅 Phase 1", True, False, False),
        ("仅 Phase 2", False, True, False),
        ("仅 Phase 4", False, False, True),
        ("Phase 1+2", True, True, False),
        ("Phase 1+4", True, False, True),
        ("Phase 2+4", False, True, True),
        ("All Phases", True, True, True),
    ]

    results = {}

    # 注意：当前代码中 Phase 1-4 是硬编码启用的
    # 所以这个测试实际上会运行相同的配置
    # 但我们可以观察错误何时发生
    print("⚠️  注意：当前代码中 Phase 1-4 已硬编码启用")
    print("    本测试将观察错误何时发生，而非测试不同配置")
    print("")

    # 只运行当前配置（All Phases enabled）
    scenario_name = "当前配置 (Phase 1+2+4 启用)"
    success = await test_scenario(scenario_name, True, True, True)
    results[scenario_name] = success

    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)

    for scenario, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {scenario}: {status}")

    print("")

    if not results[scenario_name]:
        print("🔍 发现问题！当前配置会导致 Metal 错误")
        print("")
        print("📋 下一步调查方向：")
        print("  1. 检查 Phase 1: mx.synchronize() 后 arrays 的 Metal 生命周期")
        print("  2. 检查 Phase 2: CacheSaveExecutor 后台线程是否访问 Metal 对象")
        print("  3. 检查 Phase 4: 批量 mx.eval() 后的命令提交顺序")
        print("")
        print("建议：添加 Metal 相关的详细日志，追踪命令缓冲区的创建和提交")
    else:
        print("✅ 测试通过！并发场景稳定")


if __name__ == "__main__":
    asyncio.run(main())
