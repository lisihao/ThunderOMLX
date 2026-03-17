#!/usr/bin/env python3
"""
真实缓存性能基准测试

对比测试：
1. 缓存禁用（理论上限）
2. 缓存启用（测量真实overhead）
"""
import asyncio
import sys
import time
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_with_cache_config(cache_enabled: bool, model_path: Path):
    """测试指定缓存配置的性能"""
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    config_name = "缓存启用" if cache_enabled else "缓存禁用"
    print(f"\n{'='*80}")
    print(f"📊 测试：{config_name}")
    print(f"{'='*80}\n")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    # 生成 8K prompt
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
    ]
    target_tokens = 8192
    text = " ".join(base_words * (target_tokens // len(base_words) + 1))
    tokens = tokenizer.encode(text)[:target_tokens]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)
    print(f"✅ Prompt: {actual_tokens} tokens\n")

    # 配置缓存
    scheduler_config = SchedulerConfig()
    if not cache_enabled:
        # 禁用缓存：设置 paged_ssd_cache_dir 为 None
        scheduler_config.paged_ssd_cache_dir = None
        print("🔍 缓存配置: 禁用")
    else:
        # 启用缓存（使用默认配置）
        print(f"🔍 缓存配置: 启用")
        print(f"   - paged_ssd_cache_dir = {scheduler_config.paged_ssd_cache_dir}")
        print(f"   - paged_ssd_cache_max_size = {scheduler_config.paged_ssd_cache_max_size / (1024**3):.1f}GB")
        print(f"   - hot_cache_max_size = {scheduler_config.hot_cache_max_size / (1024**3):.1f}GB")
    print()

    # 初始化引擎
    print("初始化引擎...")
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    await engine.start()
    print("✅ 引擎启动\n")

    try:
        # 预热
        print("预热...")
        async for output in engine.stream_generate(
            prompt="Hello",
            max_tokens=1,
            temperature=0.0
        ):
            pass
        print("✅ 预热完成\n")

        # 测试 Prefill（3次取平均）
        print("测试 8K Prefill (3次)...\n")
        times = []

        for trial in range(3):
            start_time = time.perf_counter()
            first_token_time = None

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=1,
                temperature=0.0
            ):
                if output.new_text:
                    first_token_time = time.perf_counter()
                    break

            if first_token_time:
                prefill_time = first_token_time - start_time
                pp_tps = actual_tokens / prefill_time
                times.append((prefill_time, pp_tps))
                print(f"  Trial {trial + 1}: {prefill_time:.3f}s, {pp_tps:.1f} tok/s")

        if times:
            avg_time = sum(t[0] for t in times) / len(times)
            avg_tps = sum(t[1] for t in times) / len(times)
            return {
                'cache_enabled': cache_enabled,
                'avg_prefill_time': avg_time,
                'avg_pp_tps': avg_tps,
                'trials': times
            }

    finally:
        print("\n关闭引擎...")
        await engine.stop()
        print("✅ 引擎已关闭")

        # 等待资源释放
        await asyncio.sleep(2)

    return None


async def main():
    print("="*80)
    print("🧪 真实缓存性能基准测试")
    print("="*80)
    print()

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    cache_dir = Path.home() / ".cache" / "omlx" / "paged_ssd"

    # 清空缓存
    if cache_dir.exists():
        print("清空缓存目录...")
        shutil.rmtree(cache_dir)
        print("✅ 缓存已清空\n")

    results = []

    # Test 1: 缓存禁用（理论上限）
    try:
        result = await test_with_cache_config(False, model_path)
        if result:
            results.append(result)
    except Exception as e:
        print(f"❌ 缓存禁用测试失败: {e}\n")

    # 等待一段时间让系统稳定
    print("\n⏳ 等待 5 秒让系统稳定...\n")
    await asyncio.sleep(5)

    # Test 2: 缓存启用（真实 overhead）
    try:
        result = await test_with_cache_config(True, model_path)
        if result:
            results.append(result)
    except Exception as e:
        print(f"❌ 缓存启用测试失败: {e}\n")

    # 汇总结果
    print("\n" + "="*80)
    print("📊 基准测试结果对比")
    print("="*80)
    print()

    if len(results) < 2:
        print("❌ 测试不完整，无法对比")
        return

    # 找到对应的结果
    no_cache = next((r for r in results if not r['cache_enabled']), None)
    with_cache = next((r for r in results if r['cache_enabled']), None)

    if not no_cache or not with_cache:
        print("❌ 缺少对比数据")
        return

    print(f"{'配置':<15} {'平均时间':<12} {'平均 TPS':<15} {'vs 无缓存':<15}")
    print("-" * 80)

    # 无缓存
    print(f"{'缓存禁用':<15} {no_cache['avg_prefill_time']:.3f}s{'':<5} {no_cache['avg_pp_tps']:.1f} tok/s{'':<5} {'基准':<15}")

    # 有缓存
    overhead_pct = (with_cache['avg_prefill_time'] - no_cache['avg_prefill_time']) / no_cache['avg_prefill_time'] * 100
    tps_loss_pct = (no_cache['avg_pp_tps'] - with_cache['avg_pp_tps']) / no_cache['avg_pp_tps'] * 100

    overhead_str = f"+{overhead_pct:.1f}% 时间"
    tps_loss_str = f"-{tps_loss_pct:.1f}% TPS"

    print(f"{'缓存启用':<15} {with_cache['avg_prefill_time']:.3f}s{'':<5} {with_cache['avg_pp_tps']:.1f} tok/s{'':<5} {tps_loss_str:<15}")

    print()
    print("📈 性能分析:")
    print(f"   - 缓存写入 overhead: {overhead_pct:.1f}% 时间增加")
    print(f"   - Processing TPS 下降: {tps_loss_pct:.1f}%")
    print(f"   - 绝对性能损失: {no_cache['avg_pp_tps'] - with_cache['avg_pp_tps']:.1f} tok/s")
    print()

    # 检查缓存是否真的写入了
    print("🔍 验证缓存写入:")
    if cache_dir.exists():
        # 统计缓存文件
        cache_files = list(cache_dir.rglob("*.safetensors*"))
        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files)
            print(f"   ✅ 缓存文件数: {len(cache_files)}")
            print(f"   ✅ 总大小: {total_size / (1024**2):.1f} MB")
        else:
            print("   ⚠️  缓存目录存在但没有文件！")
    else:
        print("   ❌ 缓存目录不存在")

    print()
    print("💡 结论:")
    if tps_loss_pct >= 20:
        print(f"   - 缓存写入 overhead 很大（{tps_loss_pct:.1f}%），Phase 1-4 优化有价值")
    elif tps_loss_pct >= 10:
        print(f"   - 缓存写入 overhead 中等（{tps_loss_pct:.1f}%），优化有一定价值")
    else:
        print(f"   - 缓存写入 overhead 较小（{tps_loss_pct:.1f}%），优化收益有限")


if __name__ == "__main__":
    asyncio.run(main())
