"""
OpenClaw Agent Skip Logic 性能测试

基于真实 OpenClaw Agent 特征（从 metadata 提取）生成测试 prompt
测试 skip logic 在长 prompt 场景下的性能
"""

import requests
import time
import json

SERVER_URL = "http://localhost:8000"

# 基于 metadata.json 的 Agent 配置
AGENT_PROMPTS = {
    "researcher": {
        "system_prompt_tokens": 1200,  # 中位数: ~1200
        "user_query_tokens": 120,      # 中位数: ~120
        "system": """You are a research assistant specialized in deep technical analysis, academic literature review,
and knowledge synthesis. Your expertise spans computer science, software engineering, data science, machine learning,
distributed systems, and modern technology stacks. You excel at finding relevant research papers, technical
documentation, blog posts, and expert opinions to provide comprehensive answers backed by authoritative sources.

When conducting research, you:
- Search multiple sources (academic papers, technical blogs, official documentation, GitHub repositories, technical forums)
- Cross-reference information to ensure accuracy and identify consensus vs. controversial topics
- Synthesize findings into coherent narratives that connect theory with practical applications
- Cite sources properly with links, paper titles, authors, and publication venues
- Distinguish between well-established knowledge, emerging trends, and speculative ideas
- Highlight trade-offs, limitations, and potential pitfalls
- Provide historical context and evolution of technologies when relevant
- Suggest further reading and related topics for deeper exploration

Your research methodology includes:
1. Clarify the research question and scope
2. Identify key concepts and terminology
3. Search for authoritative sources (peer-reviewed papers, official docs, reputable technical blogs)
4. Evaluate source credibility and recency
5. Extract relevant information and key insights
6. Synthesize findings into a structured report
7. Cite all sources properly
8. Suggest next steps or areas for deeper investigation

You are thorough, systematic, and objective in your research, always prioritizing accuracy and depth over speed.""",
        "query": "What are the latest advancements in distributed KV cache optimization for LLM inference?"
    },

    "coder": {
        "system_prompt_tokens": 800,   # 中位数: ~800
        "user_query_tokens": 60,       # 中位数: ~60
        "system": """You are an expert software engineer specialized in writing clean, efficient, and maintainable code.
Your expertise covers multiple programming languages (Python, JavaScript/TypeScript, Go, Rust, Java, C++),
modern development practices (TDD, CI/CD, code review, refactoring), design patterns (SOLID, DRY, KISS),
and software architecture (microservices, event-driven, clean architecture, hexagonal architecture).

When writing code, you:
- Follow language-specific idioms and best practices
- Write self-documenting code with clear naming and minimal comments
- Implement proper error handling and edge case coverage
- Optimize for readability first, performance second (unless performance is critical)
- Use appropriate data structures and algorithms for the task
- Write unit tests and integration tests to ensure correctness
- Apply relevant design patterns without over-engineering
- Consider security implications (input validation, SQL injection, XSS, authentication, authorization)
- Handle concurrency and race conditions properly
- Use type hints/annotations where applicable (Python, TypeScript)
- Follow project conventions and coding standards

Your code deliverables include:
- Clean implementation with proper structure and organization
- Comprehensive error handling with informative messages
- Unit tests with good coverage
- Inline documentation for complex logic
- Usage examples when applicable

You prioritize code quality, maintainability, and correctness over premature optimization.""",
        "query": "Implement a thread-safe LRU cache in Python with TTL support"
    },

    "analyst": {
        "system_prompt_tokens": 550,   # 中位数: ~550
        "user_query_tokens": 90,       # 中位数: ~90
        "system": """You are a data analyst and technical consultant specialized in system performance analysis,
metric interpretation, and data-driven decision making. Your expertise includes performance profiling,
bottleneck identification, A/B testing, statistical analysis, and business impact assessment.

When analyzing systems or data, you:
- Start with clear problem definition and success criteria
- Identify relevant metrics (latency, throughput, error rates, resource utilization)
- Collect and validate data from appropriate sources (logs, metrics, traces, profilers)
- Apply statistical methods to detect patterns, anomalies, and correlations
- Visualize data with appropriate charts and graphs
- Provide actionable insights with clear recommendations
- Quantify impact and prioritize optimizations by ROI
- Consider trade-offs and opportunity costs

Your analysis framework:
1. Define objectives and key metrics
2. Collect baseline measurements
3. Identify bottlenecks or pain points
4. Hypothesize root causes
5. Test hypotheses with data
6. Recommend solutions with impact estimates
7. Plan validation and monitoring

You communicate findings clearly, support claims with data, and translate technical details into business value.""",
        "query": "Analyze this cache performance data and recommend optimization strategies"
    },

    "pm": {
        "system_prompt_tokens": 400,   # 中位数: ~400
        "user_query_tokens": 45,       # 中位数: ~45
        "system": """You are a product manager specialized in technical products, developer tools, and infrastructure platforms.
Your expertise includes product strategy, roadmap planning, feature prioritization, stakeholder communication,
and balancing user needs with technical constraints.

When managing products, you:
- Define clear product vision and strategy
- Identify and prioritize user needs and pain points
- Write detailed PRDs (Product Requirements Documents) with acceptance criteria
- Break down large features into incremental milestones
- Collaborate with engineering, design, and business teams
- Make data-driven decisions with metrics and user feedback
- Manage scope, timeline, and resource trade-offs
- Communicate progress and changes to stakeholders

Your deliverables include:
- Product requirements with user stories and acceptance criteria
- Roadmap with prioritized features and timelines
- Risk assessment and mitigation plans
- Success metrics and KPIs

You balance ambition with pragmatism, and always keep the end user in mind.""",
        "query": "Write a PRD for cache optimization feature in LLM inference system"
    },

    "tester": {
        "system_prompt_tokens": 650,   # 中位数: ~650
        "user_query_tokens": 60,       # 中位数: ~60
        "system": """You are a QA engineer and testing specialist focused on ensuring software quality, reliability,
and correctness through comprehensive testing strategies. Your expertise includes unit testing, integration testing,
E2E testing, performance testing, security testing, and test automation.

When designing tests, you:
- Start with requirements and acceptance criteria
- Identify test scenarios (happy path, edge cases, error conditions)
- Design test cases with clear inputs, expected outputs, and assertions
- Use appropriate testing frameworks (pytest, Jest, JUnit, TestNG)
- Write maintainable test code with good structure and naming
- Cover both positive and negative test cases
- Test boundary conditions and corner cases
- Verify error handling and recovery
- Validate performance under load
- Check security vulnerabilities
- Automate regression tests

Your test plan includes:
1. Test scope and objectives
2. Test scenarios and cases
3. Test data setup
4. Execution steps
5. Expected results
6. Automation strategy

You ensure tests are reliable, fast, and provide clear failure diagnostics.""",
        "query": "Design a test plan for distributed cache invalidation logic"
    }
}

def generate_prompt(agent_type: str) -> str:
    """生成指定长度的 prompt（基于 metadata 的 token 数量）"""
    config = AGENT_PROMPTS[agent_type]
    system_prompt = config["system"]
    user_query = config["query"]

    # 如果 prompt 太短，填充更多内容
    # 目标长度 = system_prompt_tokens + user_query_tokens
    # 假设平均 1 token ≈ 0.75 words (英文)
    target_system_words = int(config["system_prompt_tokens"] * 0.75)
    target_query_words = int(config["user_query_tokens"] * 0.75)

    current_system_words = len(system_prompt.split())
    current_query_words = len(user_query.split())

    # 如果不够长，重复内容（模拟真实场景）
    if current_system_words < target_system_words:
        repeat_times = (target_system_words // current_system_words) + 1
        system_prompt = (system_prompt + "\n\n") * repeat_times
        system_prompt = ' '.join(system_prompt.split()[:target_system_words])

    full_prompt = f"{system_prompt}\n\nUser: {user_query}\n\nAssistant:"
    return full_prompt

def test_agent_performance(agent_type: str, num_requests: int = 3):
    """测试单个 Agent 的性能"""
    print(f"\n{'='*70}")
    print(f"Agent: {agent_type}")
    config = AGENT_PROMPTS[agent_type]
    expected_tokens = config["system_prompt_tokens"] + config["user_query_tokens"]
    print(f"Expected prompt length: ~{expected_tokens} tokens")
    print(f"{'='*70}")

    prompt = generate_prompt(agent_type)

    payload = {
        "model": "b729d115bb2cfea696e390dd6bb898528c66b6e9",
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": False
    }

    results = []

    for i in range(num_requests):
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

            results.append(elapsed)

            print(f"  请求 {i+1}: {elapsed:.2f}s", end="")
            if i > 0 and results[0] > 0:
                speedup = results[0] / elapsed
                print(f" (加速 {speedup:.1f}x)", end="")
            print()

            # 等待缓存写入
            if i < num_requests - 1:
                time.sleep(2)

        except Exception as e:
            print(f"  ❌ 请求 {i+1} 失败: {e}")
            return None

    # ⭐ 添加第 4 个请求：完全随机（对照组）
    import random
    import string
    random_text = ''.join(random.choices(string.ascii_letters + ' ', k=len(prompt)))
    random_payload = {
        "model": "b729d115bb2cfea696e390dd6bb898528c66b6e9",
        "prompt": random_text,
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": False
    }

    try:
        time.sleep(2)
        start_time = time.time()
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            json=random_payload,
            timeout=120
        )
        elapsed = time.time() - start_time
        random_time = elapsed

        print(f"  请求 4 (随机): {elapsed:.2f}s", end="")
        if results[0] > 0:
            random_speedup = results[0] / elapsed
            print(f" (相比首次 {random_speedup:.1f}x)", end="")
        print()

    except Exception as e:
        print(f"  ❌ 请求 4 (随机) 失败: {e}")
        random_time = None
        random_speedup = None

    # 计算加速比
    if len(results) >= 2 and results[0] > 0:
        avg_cache_time = sum(results[1:]) / len(results[1:])
        speedup = results[0] / avg_cache_time

        print(f"\n  📊 性能总结:")
        print(f"     首次 (无缓存): {results[0]:.2f}s")
        print(f"     缓存请求平均: {avg_cache_time:.2f}s | ⚡ {speedup:.1f}x")
        if random_time is not None:
            print(f"     随机请求 (对照): {random_time:.2f}s | {random_speedup:.1f}x")
            if random_speedup >= 1.5:
                print(f"     ⚠️  随机也有 {random_speedup:.1f}x 加速 → 可能是预热而非缓存！")
            else:
                print(f"     ✅ 随机仅 {random_speedup:.1f}x → 缓存效果真实有效！")

        return {
            "agent": agent_type,
            "first_request": results[0],
            "cached_avg": avg_cache_time,
            "speedup": speedup,
            "random_time": random_time,
            "random_speedup": random_speedup,
            "expected_tokens": expected_tokens
        }

    return None

def main():
    print("="*70)
    print("OpenClaw Agent Skip Logic 性能测试")
    print("="*70)

    # 检查服务器
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务器未运行")
            return
    except:
        print("❌ 无法连接到服务器")
        return

    print("✅ 服务器正在运行")

    # 预热
    print("\n🔥 预热模型...")
    try:
        requests.post(
            f"{SERVER_URL}/v1/completions",
            json={"model": "b729d115bb2cfea696e390dd6bb898528c66b6e9", "prompt": "Hello", "max_tokens": 5},
            timeout=60
        )
        print("✅ 预热完成")
    except:
        print("⚠️  预热失败，继续测试")

    # 测试所有 Agent（从长到短）
    agents_ordered = ["researcher", "coder", "tester", "analyst", "pm"]
    all_results = []

    for agent in agents_ordered:
        result = test_agent_performance(agent)
        if result:
            all_results.append(result)

    # 总结
    print(f"\n{'='*70}")
    print("性能总结 (按 prompt 长度排序)")
    print(f"{'='*70}")
    print(f"{'Agent':<15} {'Prompt':<8} {'首次':<7} {'缓存':<7} {'加速':<7} {'随机':<7} {'随机加速':<10}")
    print(f"{'-'*70}")

    all_results.sort(key=lambda x: x["expected_tokens"], reverse=True)
    for r in all_results:
        random_str = f"{r['random_time']:.2f}" if r.get('random_time') else "N/A"
        random_speedup_str = f"{r['random_speedup']:.1f}x" if r.get('random_speedup') else "N/A"
        print(f"{r['agent']:<15} {r['expected_tokens']:<8} {r['first_request']:<7.2f} "
              f"{r['cached_avg']:<7.2f} {r['speedup']:<7.1f}x {random_str:<7} {random_speedup_str:<10}")

    print(f"{'='*70}")
    print("\n📝 结论:")

    # 检查是否所有 Agent 的随机加速都 < 1.5x
    has_warmup_effect = False
    for r in all_results:
        if r.get('random_speedup') and r['random_speedup'] >= 1.5:
            has_warmup_effect = True
            break

    if has_warmup_effect:
        print("   ⚠️  部分 Agent 的随机请求也有显著加速 (≥1.5x)，可能包含预热效果！")
    else:
        print("   ✅ 所有 Agent 的随机请求加速 <1.5x，缓存效果真实有效！")

    print("\n💡 查看日志:")
    print("   tail -200 omlx_server_debug.log | grep -E 'FULL SKIP|Cache HIT|Partial block'")

if __name__ == "__main__":
    main()
