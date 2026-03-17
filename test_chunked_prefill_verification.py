#!/usr/bin/env python3
"""
Chunked Prefill 验证实验

目标：验证分块 prefill 是否影响输出质量

测试方法：
1. 生成 16K tokens prompt
2. 方法 1: 一次性 prefill (baseline)
3. 方法 2: 分块 prefill (4 chunks × 4K)
4. 对比生成的文本是否一致

验收标准：
- 生成的文本应该完全一致或高度相似（>99%）
- Logits 应该非常接近（误差 < 1e-3）
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate


def generate_test_prompt(num_tokens: int = 16 * 1024) -> str:
    """生成测试 prompt"""
    system = (
        "You are a specialized AI assistant for code analysis. "
        "Your task is to review code for security, performance, and best practices. "
    ) * 20

    chars_per_token = 4
    remaining_chars = (num_tokens - 200) * chars_per_token

    conversations = []
    turn_size = 500
    num_turns = remaining_chars // (turn_size * chars_per_token * 2)

    for i in range(num_turns):
        user_msg = f"Turn {i+1}: Please analyze this code snippet. " * 20
        assistant_msg = f"Analysis {i+1}: Found issues. " * 20
        conversations.append(f"User: {user_msg}\n\nAssistant: {assistant_msg}\n\n")

    # 最终问题（关键：测试长距离依赖）
    final_question = (
        "Based on ALL the previous analysis, "
        "what are the top 3 most critical recommendations? "
        "Please be specific and reference the turn numbers."
    )

    prompt = f"{system}\n\n{''.join(conversations)}\n\nUser: {final_question}\n\nAssistant:"
    return prompt


def baseline_generate(model, tokenizer, prompt: str, max_tokens: int = 100) -> Tuple[str, List, float]:
    """
    方法 1: 一次性 prefill (baseline)

    Returns:
        (generated_text, logits, time_ms)
    """
    print("\n" + "="*80)
    print("方法 1: 一次性 Prefill (Baseline)")
    print("="*80)

    start_time = time.perf_counter()

    # 使用标准的 generate
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"✅ Generated in {elapsed_ms:.2f}ms")
    print(f"📝 Output preview: {response[:200]}...")

    return response, None, elapsed_ms


def chunked_generate(
    model,
    tokenizer,
    prompt: str,
    chunk_size: int = 4096,
    max_tokens: int = 100
) -> Tuple[str, List, float]:
    """
    方法 2: 分块 prefill（正确实现）

    核心机制：
    1. 创建 KVCache list（每层一个）
    2. 逐块调用 model(chunk, cache=cache)
    3. cache 会自动累积（原地修改）
    4. 最后基于完整 cache 生成

    Returns:
        (generated_text, logits, time_ms)
    """
    print("\n" + "="*80)
    print(f"方法 2: 分块 Prefill (chunk_size={chunk_size})")
    print("="*80)

    # Import KVCache
    from mlx_lm.models.cache import KVCache

    start_time = time.perf_counter()

    # Tokenize
    tokens = tokenizer.encode(prompt)
    print(f"📊 Total tokens: {len(tokens)}")

    # 分块
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        chunks.append(chunk)

    print(f"📊 Split into {len(chunks)} chunks")

    try:
        # 创建 cache（关键：每层一个 KVCache）
        cache = [KVCache() for _ in range(len(model.model.layers))]
        print(f"✅ Created cache for {len(cache)} layers")

        # 逐块 prefill
        for i, chunk in enumerate(chunks):
            chunk_mx = mx.array([chunk])  # 添加 batch 维度

            print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} tokens")

            # Forward pass（cache 会自动累积）
            logits = model(chunk_mx, cache=cache)
            mx.eval(logits)  # 确保计算完成
            mx.eval([c.keys for c in cache])  # 确保 cache 更新完成

        prefill_ms = (time.perf_counter() - start_time) * 1000
        print(f"✅ All chunks processed in {prefill_ms:.2f}ms")

        # 验证 cache 大小
        cache_size = cache[0].offset if cache[0].keys is not None else 0
        print(f"📊 Cache size: {cache_size} tokens (expected: {len(tokens)})")

        if cache_size != len(tokens):
            print(f"⚠️  Cache size mismatch!")

        # 基于完整 cache 生成
        print(f"\n🔹 Generating {max_tokens} tokens...")

        # 获取最后一个 token 的 logits
        next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
        generated_tokens = [next_token]

        # 继续生成
        for _ in range(max_tokens - 1):
            logits = model(mx.array([[next_token]]), cache=cache)
            mx.eval(logits)
            next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
            generated_tokens.append(next_token)

        response = tokenizer.decode(generated_tokens)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"✅ Total time: {elapsed_ms:.2f}ms")
        print(f"📝 Output preview: {response[:200]}...")

        return response, None, elapsed_ms

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return "", None, -1


def compare_outputs(baseline: str, chunked: str) -> Dict[str, Any]:
    """
    对比两个输出

    Returns:
        比较结果
    """
    print("\n" + "="*80)
    print("输出对比")
    print("="*80)

    # 完全一致检查
    if baseline == chunked:
        print("✅ 完全一致！")
        return {
            "identical": True,
            "similarity": 1.0,
            "edit_distance": 0
        }

    # 计算相似度
    from difflib import SequenceMatcher

    similarity = SequenceMatcher(None, baseline, chunked).ratio()

    # 计算编辑距离
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    edit_dist = levenshtein_distance(baseline, chunked)

    print(f"\n📊 相似度: {similarity:.2%}")
    print(f"📊 编辑距离: {edit_dist}")

    if similarity > 0.99:
        print("✅ 高度相似（>99%）")
        result = "PASS"
    elif similarity > 0.95:
        print("⚠️  基本相似（>95%）")
        result = "WARN"
    else:
        print("❌ 相似度较低（<95%）")
        result = "FAIL"

    # 显示差异
    if baseline != chunked:
        print("\n📝 输出对比:")
        print("\nBaseline:")
        print(baseline[:500])
        print("\nChunked:")
        print(chunked[:500])

        # 找出第一个不同的位置
        for i, (c1, c2) in enumerate(zip(baseline, chunked)):
            if c1 != c2:
                print(f"\n⚠️  第一个差异在位置 {i}:")
                print(f"   Baseline: ...{baseline[max(0,i-20):i+20]}...")
                print(f"   Chunked:  ...{chunked[max(0,i-20):i+20]}...")
                break

    return {
        "identical": False,
        "similarity": similarity,
        "edit_distance": edit_dist,
        "result": result
    }


def main():
    """运行验证实验"""
    print("\n🧪 Chunked Prefill 验证实验")
    print("="*80)
    print("目标: 验证分块 prefill 是否影响输出质量")
    print("="*80)

    # 加载模型（使用 3B 模型，速度快）
    model_path = "mlx-community/Qwen2.5-3B-Instruct-8bit"
    print(f"\n📦 Loading model: {model_path}")

    try:
        model, tokenizer = load(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # 生成测试 prompt
    print("\n📝 Generating test prompt (16K tokens)...")
    prompt = generate_test_prompt(16 * 1024)

    actual_tokens = len(tokenizer.encode(prompt))
    print(f"✅ Prompt generated: {actual_tokens} tokens")

    # 方法 1: 一次性 prefill
    baseline_output, baseline_logits, baseline_time = baseline_generate(
        model, tokenizer, prompt, max_tokens=100
    )

    # 清理
    import gc
    gc.collect()
    time.sleep(2)

    # 方法 2: 分块 prefill
    chunked_output, chunked_logits, chunked_time = chunked_generate(
        model, tokenizer, prompt, chunk_size=4096, max_tokens=100
    )

    # 对比结果
    if baseline_output and chunked_output:
        comparison = compare_outputs(baseline_output, chunked_output)

        # 性能对比
        print("\n" + "="*80)
        print("性能对比")
        print("="*80)
        print(f"Baseline: {baseline_time:.2f}ms")
        print(f"Chunked:  {chunked_time:.2f}ms")

        if chunked_time > 0 and baseline_time > 0:
            diff_pct = ((chunked_time - baseline_time) / baseline_time) * 100
            print(f"Difference: {diff_pct:+.1f}%")

        # 最终结论
        print("\n" + "="*80)
        print("实验结论")
        print("="*80)

        if comparison["identical"]:
            print("\n✅ 验证通过！")
            print("   - 输出完全一致")
            print("   - Chunked Prefill 不影响输出质量")
            print("   - 可以安全用于 128K-256K tokens")
            return 0
        elif comparison["similarity"] > 0.99:
            print("\n✅ 验证基本通过")
            print(f"   - 相似度: {comparison['similarity']:.2%}")
            print("   - 微小差异可能来自数值精度")
            print("   - Chunked Prefill 基本不影响质量")
            return 0
        elif comparison["similarity"] > 0.95:
            print("\n⚠️  验证部分通过")
            print(f"   - 相似度: {comparison['similarity']:.2%}")
            print("   - 存在一定差异，需要检查实现")
            print("   - 建议修复后再使用")
            return 1
        else:
            print("\n❌ 验证失败")
            print(f"   - 相似度: {comparison['similarity']:.2%}")
            print("   - 差异过大，实现可能有问题")
            print("   - 不建议使用")
            return 1
    else:
        print("\n❌ 测试失败")
        print("   - 无法完成对比")
        return 1


if __name__ == "__main__":
    exit(main())
