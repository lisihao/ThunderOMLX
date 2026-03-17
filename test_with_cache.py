#!/usr/bin/env python3
"""测试：缓存启用（真实 overhead）"""
import asyncio
import sys
import time
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from transformers import AutoTokenizer

    print("="*80)
    print("📊 基准测试：缓存启用（真实 overhead）")
    print("="*80)
    print()

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    cache_dir = Path.home() / ".cache" / "omlx" / "paged_ssd"

    # 清空缓存
    if cache_dir.exists():
        print("清空缓存目录...")
        shutil.rmtree(cache_dir)
        print("✅ 缓存已清空\n")

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

    # 配置：启用缓存
    scheduler_config = SchedulerConfig()
    # paged_ssd_cache_dir 默认已启用
    print("🔍 缓存配置: 启用")
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

        # 测试 Prefill（3次）
        print("测试 8K Prefill + 缓存写入 (3次)...\n")
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

            print()
            print("="*80)
            print("📊 结果（缓存启用）")
            print("="*80)
            print()
            print(f"平均 Prefill 时间: {avg_time:.3f}s")
            print(f"平均 PP TPS:       {avg_tps:.1f} tok/s")
            print()

            # 检查缓存写入
            print("🔍 验证缓存写入:")
            if cache_dir.exists():
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
            print("💡 与无缓存性能对比:")
            print("   - 参考：无缓存约 740-890 tok/s")
            print("   - 当前：{:.1f} tok/s".format(avg_tps))
            if avg_tps < 740:
                overhead_pct = (740 - avg_tps) / 740 * 100
                print(f"   - 估计 overhead: ~{overhead_pct:.1f}%")

    finally:
        print()
        print("关闭引擎...")
        await engine.stop()
        print("✅ 引擎已关闭")


if __name__ == "__main__":
    asyncio.run(main())
