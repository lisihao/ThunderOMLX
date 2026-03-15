"""
分析 llama-server 日志，提取真实的 KV Cache 统计
"""

import re
import sys
from collections import defaultdict


def analyze_llama_log(log_file: str):
    """分析 llama-server 日志"""
    print(f"分析日志: {log_file}")
    print("=" * 70)

    # 统计数据
    stats = {
        "lmcache_restored": 0,  # LMCache 恢复次数
        "requests": 0,  # 请求总数
        "prefill_times": [],  # Prefill 时间列表
        "eval_times": [],  # Eval 时间列表
        "total_times": [],  # 总时间列表
        "tokens_prefill": [],  # Prefill tokens
        "tokens_eval": [],  # Eval tokens
    }

    with open(log_file, "r") as f:
        for line in f:
            # LMCache 恢复
            if "LMCache RESTORED" in line:
                stats["lmcache_restored"] += 1

            # Timing 信息
            if "prompt eval time" in line:
                # prompt eval time =      40.41 ms /     1 tokens
                match = re.search(r"prompt eval time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens", line)
                if match:
                    stats["prefill_times"].append(float(match.group(1)))
                    stats["tokens_prefill"].append(int(match.group(2)))

            if "eval time" in line and "prompt eval time" not in line:
                # eval time =     853.10 ms /    50 tokens
                match = re.search(r"eval time\s+=\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens", line)
                if match:
                    stats["eval_times"].append(float(match.group(1)))
                    stats["tokens_eval"].append(int(match.group(2)))

            if "total time" in line:
                # total time =     893.51 ms /    51 tokens
                match = re.search(r"total time\s+=\s+([\d.]+)\s+ms", line)
                if match:
                    stats["total_times"].append(float(match.group(1)))
                    stats["requests"] += 1

    # 计算统计
    print(f"\n📊 请求统计:")
    print(f"   总请求数: {stats['requests']}")
    print(f"   LMCache 恢复次数: {stats['lmcache_restored']}")

    if stats["lmcache_restored"] > 0:
        # 每个请求可能有多层恢复（48 层）
        requests_with_cache = stats["lmcache_restored"] // 48 if stats["lmcache_restored"] >= 48 else stats["lmcache_restored"]
        cache_hit_rate = (requests_with_cache / stats["requests"] * 100) if stats["requests"] > 0 else 0
        print(f"   KV Cache 命中率: {cache_hit_rate:.1f}% ({requests_with_cache}/{stats['requests']})")

    if stats["prefill_times"]:
        avg_prefill = sum(stats["prefill_times"]) / len(stats["prefill_times"])
        avg_tokens_prefill = sum(stats["tokens_prefill"]) / len(stats["tokens_prefill"])
        print(f"\n📊 Prefill 统计:")
        print(f"   平均时间: {avg_prefill:.2f}ms")
        print(f"   平均 tokens: {avg_tokens_prefill:.0f}")
        print(f"   速度: {avg_tokens_prefill / (avg_prefill / 1000):.1f} tokens/s")

    if stats["eval_times"]:
        avg_eval = sum(stats["eval_times"]) / len(stats["eval_times"])
        avg_tokens_eval = sum(stats["tokens_eval"]) / len(stats["tokens_eval"])
        print(f"\n📊 Eval (Decode) 统计:")
        print(f"   平均时间: {avg_eval:.2f}ms")
        print(f"   平均 tokens: {avg_tokens_eval:.0f}")
        print(f"   速度: {avg_tokens_eval / (avg_eval / 1000):.1f} tokens/s")

    if stats["total_times"]:
        avg_total = sum(stats["total_times"]) / len(stats["total_times"])
        print(f"\n📊 总体统计:")
        print(f"   平均推理时间: {avg_total:.2f}ms")

    # 分析 Cache 效果
    if stats["lmcache_restored"] > 0 and stats["prefill_times"]:
        print(f"\n💡 KV Cache 分析:")
        print(f"   LMCache 在运行中被复用 {stats['lmcache_restored']} 次")
        print(f"   这说明 ThunderLLAMA 的 KV Cache 机制在工作！")
        print(f"   每次恢复可以节省 prefill 计算时间")

        if avg_tokens_prefill > 100:
            estimated_saved_ms = avg_tokens_prefill * 0.5  # 假设 0.5ms/token
            print(f"   估计每次 Cache 命中节省: {estimated_saved_ms:.1f}ms")

    return stats


def main():
    log_file = "/tmp/llama-server-30b.log"
    try:
        analyze_llama_log(log_file)
    except FileNotFoundError:
        print(f"❌ 日志文件不存在: {log_file}")
        print("   llama-server 可能没有运行过")
        return 1
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
