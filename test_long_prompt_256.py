#!/usr/bin/env python3
"""测试 block_size=256 对不同长度 prompt 的缓存效果

测试目标：
验证 block_size=256 在处理中长 prompt（256-1024 tokens）时的优势：
1. 更少的碎片化
2. 更高的缓存命中率
3. 更好的 FULL SKIP 触发率
"""

import asyncio
import logging
import time
from pathlib import Path

from mlx_lm import load

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


async def test_long_prompts():
    """测试不同长度的 prompt"""

    print("=" * 80)
    print("Long Prompt 缓存性能测试 (block_size=256)")
    print("=" * 80)

    # 1. 加载模型
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"

    print(f"\n[加载模型]")
    print(f"  路径: {model_path}")

    if not model_path.exists():
        print(f"  ❌ 模型不存在")
        return

    model, tokenizer = load(str(model_path))
    print(f"  ✅ 模型加载成功")

    # 2. 创建 EngineCore（使用 block_size=256）
    from omlx.engine_core import EngineCore, EngineConfig
    from omlx.scheduler import SchedulerConfig
    from omlx.request import SamplingParams

    scheduler_config = SchedulerConfig(
        max_num_seqs=2,
        paged_cache_block_size=256,  # ⚡ 使用优化后的 256
        disable_block_size_enlargement=True,
        max_cache_blocks=512,
        initial_cache_blocks=64,
        paged_ssd_cache_dir=str(Path.home() / ".cache" / "omlx" / "test_long_prompt"),
        model_name=str(model_path),
    )

    engine_config = EngineConfig(
        model_name=str(model_path),
        scheduler_config=scheduler_config,
    )

    engine = EngineCore(model=model, tokenizer=tokenizer, config=engine_config)

    print(f"\n配置:")
    print(f"  block_size: {engine.scheduler.config.paged_cache_block_size}")
    print(f"  ✅ BlockAwarePrefixCache 已启用")

    await engine.start()

    # 3. 构建不同长度的 prompt
    # 基础文本（约 50 tokens）
    base_text = """人工智能（Artificial Intelligence, AI）是计算机科学的一个重要分支，
致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。"""

    # 扩展文本块（约 100 tokens）
    expansion = """
    AI 的研究领域包括机器学习、深度学习、自然语言处理、计算机视觉、专家系统、
    机器人技术等多个方向。近年来，随着大数据、云计算和算力的快速发展，
    AI 技术取得了突破性进展，在图像识别、语音识别、自然语言理解等领域达到了人类水平。"""

    # 生成不同长度的 prompt
    test_prompts = []

    # 256 tokens (1 个完整 block)
    prompt_256 = base_text + (expansion * 2)
    tokens_256 = tokenizer.encode(prompt_256)
    actual_len_256 = len(tokens_256)
    test_prompts.append({
        "name": f"256-token prompt (实际: {actual_len_256})",
        "text": prompt_256,
        "target": 256,
        "actual": actual_len_256,
    })

    # 512 tokens (2 个完整 block)
    prompt_512 = base_text + (expansion * 5)
    tokens_512 = tokenizer.encode(prompt_512)
    actual_len_512 = len(tokens_512)
    test_prompts.append({
        "name": f"512-token prompt (实际: {actual_len_512})",
        "text": prompt_512,
        "target": 512,
        "actual": actual_len_512,
    })

    # 768 tokens (3 个完整 block)
    prompt_768 = base_text + (expansion * 8)
    tokens_768 = tokenizer.encode(prompt_768)
    actual_len_768 = len(tokens_768)
    test_prompts.append({
        "name": f"768-token prompt (实际: {actual_len_768})",
        "text": prompt_768,
        "target": 768,
        "actual": actual_len_768,
    })

    # 1024 tokens (4 个完整 block)
    prompt_1024 = base_text + (expansion * 11)
    tokens_1024 = tokenizer.encode(prompt_1024)
    actual_len_1024 = len(tokens_1024)
    test_prompts.append({
        "name": f"1024-token prompt (实际: {actual_len_1024})",
        "text": prompt_1024,
        "target": 1024,
        "actual": actual_len_1024,
    })

    print(f"\n[测试用例]")
    for p in test_prompts:
        blocks = p["actual"] // 256
        partial = p["actual"] % 256
        print(f"  {p['name']}: {blocks} 个完整块 + {partial} 个部分 tokens")

    # 4. 运行测试
    sampling_params = SamplingParams(max_tokens=30)

    results = []

    for prompt_info in test_prompts:
        print(f"\n{'=' * 80}")
        print(f"测试: {prompt_info['name']}")
        print(f"{'=' * 80}")

        # 第一次运行（无缓存）
        print(f"\n[第 1 次运行 - 无缓存]")
        start = time.perf_counter()
        try:
            output1 = await engine.generate(
                prompt=prompt_info['text'],
                sampling_params=sampling_params,
            )
            time1 = (time.perf_counter() - start) * 1000
            print(f"  时间: {time1:.2f} ms")
            print(f"  输出: {output1.output_text[:50]}...")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            time1 = None

        # 第二次运行（100% 缓存命中）
        print(f"\n[第 2 次运行 - 100% 重复（期望 FULL SKIP）]")
        start = time.perf_counter()
        try:
            output2 = await engine.generate(
                prompt=prompt_info['text'],
                sampling_params=sampling_params,
            )
            time2 = (time.perf_counter() - start) * 1000
            print(f"  时间: {time2:.2f} ms")
            print(f"  输出: {output2.output_text[:50]}...")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            time2 = None

        # 计算加速比
        if time1 and time2:
            speedup = time1 / time2
            blocks = prompt_info['actual'] // 256
            fragmentation = (prompt_info['actual'] % 256) / 256 * 100

            results.append({
                "name": prompt_info['name'],
                "tokens": prompt_info['actual'],
                "blocks": blocks,
                "fragmentation": fragmentation,
                "time_no_cache": time1,
                "time_cached": time2,
                "speedup": speedup,
            })

            print(f"\n📊 性能分析:")
            print(f"  无缓存: {time1:.2f} ms")
            print(f"  有缓存: {time2:.2f} ms")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  完整块: {blocks} 个")
            print(f"  碎片化: {fragmentation:.1f}%")

    # 5. 汇总结果
    print(f"\n{'=' * 80}")
    print(f"测试总结 (block_size=256)")
    print(f"{'=' * 80}")
    print(f"\n{'Prompt Length':<20} {'Blocks':<10} {'碎片化':<12} {'加速比':<10} {'FULL SKIP'}")
    print(f"{'-' * 80}")

    for r in results:
        skip_indicator = "✅" if r['speedup'] >= 3.0 else "⚠️"
        print(f"{r['tokens']:<20} {r['blocks']:<10} {r['fragmentation']:.1f}%{' ' * 7} "
              f"{r['speedup']:.2f}x{' ' * 5} {skip_indicator}")

    print(f"\n预期：")
    print(f"  - 256+ tokens: 加速比 >= 3.0x (FULL SKIP 触发)")
    print(f"  - 完整块越多，碎片化越低")
    print(f"  - block_size=256 对中长 prompt 优化明显")

    await engine.close()

    print(f"\n✅ 测试完成！查看日志确认 FULL SKIP 触发情况")


if __name__ == "__main__":
    asyncio.run(test_long_prompts())
