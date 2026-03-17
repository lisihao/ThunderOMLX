#!/usr/bin/env python3
"""
Phase 1 全面验证测试：不同 context length 性能对比

测试场景：
1. 2K context: 快速验证
2. 4K context: 中等负载
3. 8K context: 高负载（主要测试点）
4. 16K context: 压力测试（如果内存允许）
"""
import asyncio
import sys
import time
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_context_length(context_length: int, model_path: Path):
    """测试指定 context length 的性能"""
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    print(f"\n{'='*80}")
    print(f"📏 测试 {context_length}K Context")
    print(f"{'='*80}\n")

    # 加载 tokenizer（复用，避免重复警告）
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )

    # 生成指定长度的 prompt
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
    ]
    target_tokens = context_length * 1024
    text = " ".join(base_words * (target_tokens // len(base_words) + 1))
    tokens = tokenizer.encode(text)[:target_tokens]
    prompt = tokenizer.decode(tokens)
    actual_tokens = len(tokens)
    print(f"✅ Prompt: {actual_tokens} tokens\n")

    # 初始化引擎（缓存启用）
    print("初始化引擎...")
    scheduler_config = SchedulerConfig()

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

        # 测试 Prefill
        print(f"测试 {context_length}K Prefill...\n")
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
            return {
                'context_length': context_length,
                'actual_tokens': actual_tokens,
                'prefill_time': prefill_time,
                'pp_tps': pp_tps
            }

    finally:
        print("\n关闭引擎...")
        await engine.stop()
        print("✅ 引擎已关闭")

        # 短暂等待让资源释放
        await asyncio.sleep(2)

    return None


async def main():
    print("="*80)
    print("🧪 Phase 1 全面验证测试")
    print("="*80)
    print()

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    # 清空缓存
    cache_dir = Path.home() / ".cache" / "omlx" / "paged_ssd"
    if cache_dir.exists():
        print("清空缓存目录...")
        shutil.rmtree(cache_dir)
        print("✅ 缓存已清空\n")

    # 加载 tokenizer（避免每次测试重复加载）
    print("加载 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
    print("✅ Tokenizer 加载完成\n")

    # 测试不同的 context lengths
    test_configs = [
        2,   # 2K - 快速验证
        4,   # 4K - 中等负载
        8,   # 8K - 主要测试点
        # 16,  # 16K - 压力测试（可选，可能 OOM）
    ]

    results = []
    for context_k in test_configs:
        try:
            result = await test_context_length(context_k, model_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ {context_k}K 测试失败: {e}\n")
            continue

    # 汇总结果
    print("\n" + "="*80)
    print("📊 Phase 1 全面测试结果")
    print("="*80)
    print()

    if not results:
        print("❌ 没有成功的测试结果")
        return

    # 打印表格
    print(f"{'Context':<10} {'Tokens':<10} {'Time (s)':<12} {'PP TPS':<12} {'vs 基准':<15}")
    print("-" * 80)

    for r in results:
        ctx = f"{r['context_length']}K"
        tokens = r['actual_tokens']
        time_s = f"{r['prefill_time']:.3f}"
        tps = f"{r['pp_tps']:.1f}"

        # 基准参考：缓存禁用 ~740-890 tok/s
        baseline_min = 740
        baseline_max = 890
        baseline_avg = (baseline_min + baseline_max) / 2

        vs_baseline = (r['pp_tps'] - baseline_avg) / baseline_avg * 100
        vs_str = f"{vs_baseline:+.1f}%"

        # 判断颜色（文本）
        if r['pp_tps'] >= baseline_min:
            status = "✅"
        elif r['pp_tps'] >= 600:
            status = "⚠️ "
        else:
            status = "❌"

        print(f"{ctx:<10} {tokens:<10} {time_s:<12} {tps:<12} {vs_str:<15} {status}")

    print()
    print("📍 参考基准:")
    print("   - 缓存禁用:      ~740-890 tok/s (理论上限)")
    print("   - 缓存启用(优化前): ~550-650 tok/s (-24-26%)")
    print("   - Phase 1 目标:  ~600-700 tok/s (+2.8%)")
    print()

    # 分析性能趋势
    print("📈 性能分析:")
    avg_tps = sum(r['pp_tps'] for r in results) / len(results)
    print(f"   - 平均 PP TPS: {avg_tps:.1f} tok/s")

    if all(r['pp_tps'] >= 740 for r in results):
        print("   - ✅ 所有测试达到无缓存性能水平！")
        print("   - ✅ Phase 1 完全消除了缓存写入开销")
    elif all(r['pp_tps'] >= 600 for r in results):
        print("   - ✅ 所有测试达到 Phase 1 目标")
        print(f"   - ✅ 相比优化前提升 {(avg_tps - 600) / 600 * 100:.1f}%")
    else:
        print("   - ⚠️  部分测试未达到目标，需要进一步优化")

    # 检查性能稳定性
    if len(results) >= 2:
        tps_values = [r['pp_tps'] for r in results]
        max_tps = max(tps_values)
        min_tps = min(tps_values)
        variation = (max_tps - min_tps) / avg_tps * 100
        print(f"   - 性能波动: ±{variation:.1f}%", end="")
        if variation < 5:
            print(" (稳定) ✅")
        elif variation < 10:
            print(" (较稳定) ⚠️")
        else:
            print(" (不稳定) ❌")

    print()


if __name__ == "__main__":
    asyncio.run(main())
