"""
简化的缓存匹配测试 - 只运行 2 个请求，用于调试 block_hash 匹配问题

目标：验证第 2 个请求是否能匹配第 1 个请求创建的 cache blocks
"""

import requests
import time

SERVER_URL = "http://localhost:8000"

# 使用足够长的 System Prompt 以确保 prompt 本身 >= 64 tokens (block_size=64)
# 这样可以在 prefill 前就匹配到缓存
SYSTEM_PROMPT = """You are a highly knowledgeable AI assistant specialized in providing detailed technical explanations.
Your expertise covers a wide range of topics including software engineering, computer science fundamentals,
programming languages, algorithms, data structures, system design, and modern technology stacks.
You excel at breaking down complex concepts into clear, understandable explanations suitable for learners
at all levels. When answering questions, you prioritize accuracy, clarity, and practical examples.
You also stay current with industry best practices and emerging technologies to provide the most
relevant and useful information possible."""

def send_request(query: str, request_num: int):
    """发送单个请求"""
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {query}\n\nAssistant:"

    payload = {
        "model": "Qwen3.5-35B-A3B-6bit",
        "prompt": full_prompt,
        "max_tokens": 20,  # 减少生成 tokens，降低内存压力
        "temperature": 0.0,  # 确定性生成
        "stream": False
    }

    print(f"\n{'='*60}")
    print(f"请求 {request_num}: '{query}'")
    print(f"{'='*60}")

    try:
        start_time = time.time()
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        # 提取 cached_tokens
        cached_tokens = result.get("usage", {}).get("cached_tokens", 0)
        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)

        print(f"✅ 成功 - 耗时: {elapsed:.2f}s")
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Cached tokens: {cached_tokens}")
        print(f"   Cache hit rate: {100.0 * cached_tokens / prompt_tokens if prompt_tokens > 0 else 0:.1f}%")

        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        return False

def main():
    print("="*60)
    print("ThunderOMLX 缓存匹配调试测试")
    print("="*60)

    # 检查服务器
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务器未运行")
            return
    except:
        print("❌ 无法连接到服务器")
        return

    print("✅ 服务器正在运行\n")

    # 发送 2 个相同的请求
    queries = [
        "What is AI?",
        "What is AI?"  # 完全相同，应该 100% 命中缓存
    ]

    for i, query in enumerate(queries, 1):
        success = send_request(query, i)
        if not success:
            print(f"\n⚠️  请求 {i} 失败，停止测试")
            break

        # 等待一下，让缓存写入完成
        if i < len(queries):
            time.sleep(2)

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n💡 提示：查看服务器日志中的 block_hash 信息：")
    print("   tail -100 /Users/lisihao/ThunderOMLX/omlx_server_debug.log | grep -E '🔑 Computed|✅ Registered|🔍 Cache MISS|✅ Cache HIT'")

if __name__ == "__main__":
    main()
