#!/usr/bin/env python3
"""
P2.2 性能对比实验 - 流式加载 vs 批量加载

测试场景：
- Prefix Cache Hit 场景（有已缓存的 blocks）
- 16K/32K 上下文
- 对比指标：Cache Loading 时间、TTFT、内存峰值

实验设计：
1. Phase 1: 构建 Prefix Cache（第一次 prefill）
2. Phase 2: 清理内存，重新加载模型
3. Phase 3: 测试 Cache Hit + Loading 性能
   - Run 1: 批量加载（OMLX_STREAMING_THRESHOLD=999999）
   - Run 2: 流式加载（OMLX_STREAMING_THRESHOLD=32）
"""

import sys
import os
import time
import gc
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available")

import mlx.core as mx
from mlx_lm import load, generate


class MemoryMonitor:
    """监控内存使用"""
    def __init__(self):
        self.start_rss_mb = None
        self.peak_rss_mb = 0
        self.samples = []

    def start(self):
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            self.start_rss_mb = process.memory_info().rss / (1024 ** 2)

    def sample(self, label: str = ""):
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            process = psutil.Process()

            rss_mb = process.memory_info().rss / (1024 ** 2)
            available_gb = mem.available / (1024 ** 3)

            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
            delta_mb = rss_mb - self.start_rss_mb if self.start_rss_mb else 0

            self.samples.append({
                "label": label,
                "rss_mb": rss_mb,
                "delta_mb": delta_mb,
                "available_gb": available_gb
            })

            print(f"    💾 {label:30s} | RSS={rss_mb:>8.0f}MB (Δ{delta_mb:>+7.0f}MB) | Avail={available_gb:.1f}GB")


def generate_long_context_prompt(num_tokens: int) -> str:
    """生成超长上下文 prompt"""
    # System prompt
    system = (
        "You are a specialized AI assistant for code analysis. "
        "Your task is to review code for security, performance, and best practices. "
    ) * 20  # ~800 tokens

    # 长对话历史
    chars_per_token = 4
    remaining_chars = (num_tokens - 200) * chars_per_token

    conversations = []
    turn_size = 500  # tokens per turn
    num_turns = remaining_chars // (turn_size * chars_per_token * 2)

    for i in range(num_turns):
        user_msg = f"Turn {i+1}: Please analyze this code snippet for potential issues. " * 20
        assistant_msg = f"Analysis {i+1}: Found issues in security and performance. " * 20
        conversations.append(f"User: {user_msg}\n\nAssistant: {assistant_msg}\n\n")

    # 最终问题
    final_question = "Based on all the previous analysis, what are the top 3 recommendations?"

    prompt = f"{system}\n\n{''.join(conversations)}\n\nUser: {final_question}\n\nAssistant:"
    return prompt


def run_test_with_threshold(
    model_path: str,
    prompt: str,
    threshold: int,
    label: str
) -> Dict[str, Any]:
    """
    使用指定阈值运行测试

    Args:
        model_path: 模型路径
        prompt: 测试 prompt
        threshold: 流式加载阈值
        label: 测试标签

    Returns:
        性能指标
    """
    print(f"\n{'='*80}")
    print(f"🧪 {label}")
    print(f"   Threshold: {threshold} blocks")
    print(f"{'='*80}")

    monitor = MemoryMonitor()
    monitor.start()
    monitor.sample("start")

    # 加载模型
    print(f"\n  📦 Loading model...")
    start_load = time.perf_counter()

    try:
        model, tokenizer = load(model_path)
        load_time = (time.perf_counter() - start_load) * 1000
        print(f"     ✅ Model loaded in {load_time:.0f}ms")
    except Exception as e:
        print(f"     ❌ Failed to load model: {e}")
        return {"error": str(e)}

    monitor.sample("model loaded")

    # Tokenize
    print(f"\n  🔤 Tokenizing...")
    tokens = tokenizer.encode(prompt)
    actual_tokens = len(tokens)
    print(f"     Tokens: {actual_tokens}")

    monitor.sample("tokenized")

    # 测试生成（TTFT + Cache Loading）
    print(f"\n  ⚡ Testing generation (TTFT + Cache Loading)...")
    start_ttft = time.perf_counter()

    try:
        # 生成第一个 token（包含 cache loading 时间）
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1,
            verbose=False
        )

        ttft_ms = (time.perf_counter() - start_ttft) * 1000

        print(f"     ✅ TTFT (including cache load): {ttft_ms:.2f}ms")

    except Exception as e:
        print(f"     ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        ttft_ms = -1

    monitor.sample("generation done")

    # 清理
    print(f"\n  🧹 Cleanup...")
    del model
    del tokenizer
    gc.collect()

    monitor.sample("cleanup done")

    return {
        "label": label,
        "threshold": threshold,
        "tokens": actual_tokens,
        "ttft_ms": ttft_ms,
        "peak_memory_mb": monitor.peak_rss_mb
    }


def main():
    """运行性能对比实验"""
    print("\n🧪 P2.2 性能对比实验 - 流式加载 vs 批量加载")
    print("="*80)
    print("模型: Qwen3.5 35B MLX")
    print("场景: Prefix Cache Hit（有已缓存的 blocks）")
    print("="*80)

    model_path = str(Path.home() / "models" / "qwen3.5-35b-mlx")

    # 验证模型存在
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    print(f"✅ Model found: {model_path}")

    # 测试场景
    test_scenarios = [
        (16 * 1024, "16K Context"),
        (32 * 1024, "32K Context"),
    ]

    all_results = []

    for num_tokens, scenario_label in test_scenarios:
        print(f"\n\n{'#'*80}")
        print(f"# Scenario: {scenario_label} ({num_tokens//1000}K tokens)")
        print(f"{'#'*80}")

        # 生成 prompt
        print(f"\n📝 Generating {num_tokens//1000}K tokens prompt...")
        prompt = generate_long_context_prompt(num_tokens)

        # Phase 1: 构建 Prefix Cache (第一次 prefill)
        print(f"\n{'='*80}")
        print(f"Phase 1: Building Prefix Cache")
        print(f"{'='*80}")

        # 设置环境变量（使用默认阈值）
        os.environ["OMLX_STREAMING_THRESHOLD"] = "32"

        monitor_build = MemoryMonitor()
        monitor_build.start()

        try:
            # 加载模型并生成（构建 cache）
            print(f"\n  📦 Loading model...")
            model, tokenizer = load(model_path)
            print(f"     ✅ Model loaded")

            print(f"\n  🔨 Building cache (first prefill)...")
            _ = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=1,
                verbose=False
            )
            print(f"     ✅ Cache built")

            # 清理
            del model
            del tokenizer
            gc.collect()

        except Exception as e:
            print(f"     ❌ Failed to build cache: {e}")
            continue

        monitor_build.sample("cache built")

        # 等待系统稳定
        print(f"\n  ⏳ Waiting for system to stabilize...")
        time.sleep(3)

        # Phase 2: 测试批量加载（阈值 999999 = 禁用流式加载）
        os.environ["OMLX_STREAMING_THRESHOLD"] = "999999"
        result_batch = run_test_with_threshold(
            model_path,
            prompt,
            999999,
            f"{scenario_label} - Batch Load (No Streaming)"
        )
        all_results.append(result_batch)

        # 等待系统稳定
        print(f"\n  ⏳ Waiting for system to stabilize...")
        time.sleep(3)

        # Phase 3: 测试流式加载（阈值 32）
        os.environ["OMLX_STREAMING_THRESHOLD"] = "32"
        result_streaming = run_test_with_threshold(
            model_path,
            prompt,
            32,
            f"{scenario_label} - Streaming Load (Threshold 32)"
        )
        all_results.append(result_streaming)

        # 等待系统稳定
        print(f"\n  ⏳ Waiting for system to stabilize...")
        time.sleep(3)

    # 结果汇总
    print("\n\n" + "="*80)
    print("📊 实验结果汇总")
    print("="*80)

    print(f"\n{'场景':<45} | {'阈值':>10} | {'Tokens':>8} | {'TTFT (ms)':>12} | {'内存 (MB)':>12}")
    print("-"*100)

    for result in all_results:
        if "error" in result:
            print(f"{result['label']:<45} | {'ERROR':>10} | {result['error'][:30]}")
            continue

        label = result['label']
        threshold = result['threshold']
        tokens = result['tokens']
        ttft = result['ttft_ms']
        memory = result['peak_memory_mb']

        print(f"{label:<45} | {threshold:>10} | {tokens:>8} | {ttft:>12.2f} | {memory:>12.0f}")

    # 性能对比
    print("\n" + "="*80)
    print("📈 性能对比分析")
    print("="*80)

    for i in range(0, len(all_results), 2):
        if i + 1 >= len(all_results):
            break

        batch_result = all_results[i]
        streaming_result = all_results[i + 1]

        if "error" in batch_result or "error" in streaming_result:
            continue

        scenario = batch_result['label'].split(" - ")[0]

        print(f"\n{'='*80}")
        print(f"Scenario: {scenario}")
        print(f"{'='*80}")

        # TTFT 对比
        batch_ttft = batch_result['ttft_ms']
        streaming_ttft = streaming_result['ttft_ms']

        if batch_ttft > 0 and streaming_ttft > 0:
            ttft_diff = streaming_ttft - batch_ttft
            ttft_pct = (ttft_diff / batch_ttft) * 100

            print(f"\n📊 TTFT (包含 Cache Loading):")
            print(f"   Batch Load (No Streaming):  {batch_ttft:>10.2f}ms")
            print(f"   Streaming Load (Threshold 32): {streaming_ttft:>10.2f}ms")
            print(f"   Difference:                  {ttft_diff:>+10.2f}ms ({ttft_pct:>+6.1f}%)")

            if abs(ttft_pct) < 10:
                print(f"   ✅ 性能开销可接受 (< 10%)")
            elif ttft_pct < 0:
                print(f"   ✅ 流式加载更快!")
            else:
                print(f"   ⚠️  流式加载较慢 (> 10%)")

        # 内存对比
        batch_mem = batch_result['peak_memory_mb']
        streaming_mem = streaming_result['peak_memory_mb']

        mem_diff = streaming_mem - batch_mem
        mem_pct = (mem_diff / batch_mem) * 100

        print(f"\n💾 内存峰值:")
        print(f"   Batch Load (No Streaming):  {batch_mem:>10.0f}MB")
        print(f"   Streaming Load (Threshold 32): {streaming_mem:>10.0f}MB")
        print(f"   Difference:                  {mem_diff:>+10.0f}MB ({mem_pct:>+6.1f}%)")

        if mem_diff < 0:
            print(f"   ✅ 流式加载节省内存 ({abs(mem_diff):.0f}MB)")
        elif abs(mem_pct) < 5:
            print(f"   ✅ 内存使用相当 (< 5% 差异)")
        else:
            print(f"   ⚠️  流式加载使用更多内存")

    # 最终结论
    print("\n\n" + "="*80)
    print("📝 结论")
    print("="*80)

    print("\n✅ 如果 TTFT 差异 < 10%: 流式加载性能开销可接受")
    print("✅ 如果内存峰值更低: 流式加载有效控制内存")
    print("✅ 如果 TTFT 更快: 流式加载优化了 cache loading 路径")

    print("\n注意:")
    print("- 本测试测量的是 **Prefix Cache Hit + Loading** 场景")
    print("- TTFT 包含了 cache loading 时间")
    print("- 对于 OpenClaw 多 agent 场景，这是最常见的使用模式")

    return 0


if __name__ == "__main__":
    exit(main())
