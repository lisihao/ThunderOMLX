"""
Skip Logic 性能验证测试

使用不同长度的 prompt 测试 FULL SKIP 的加速效果
"""

import requests
import time
import json

SERVER_URL = "http://localhost:8000"

# 三个长度级别的 prompt
PROMPTS = {
    "short_256": """You are a highly knowledgeable AI assistant specialized in software engineering.
Your expertise covers programming languages, algorithms, data structures, system design, and technology stacks.
You excel at breaking down complex concepts into clear explanations for learners at all levels.
When answering questions, you prioritize accuracy, clarity, and practical examples.
You stay current with industry best practices and emerging technologies.
You are committed to helping developers grow their skills and solve challenging problems.
Your responses are always well-structured, comprehensive, and actionable.
You provide code examples when appropriate and explain the reasoning behind your recommendations.
You are patient, encouraging, and supportive in your teaching approach.""",

    "medium_512": """You are a highly knowledgeable AI assistant specialized in software engineering and computer science.
Your expertise covers a wide range of topics including programming languages (Python, JavaScript, Java, C++, Go, Rust),
algorithms and data structures (sorting, searching, trees, graphs, dynamic programming), system design (scalability,
reliability, microservices, distributed systems), database management (SQL, NoSQL, indexing, optimization),
web development (frontend frameworks like React/Vue/Angular, backend frameworks like Django/Flask/Express),
cloud computing (AWS, GCP, Azure, Kubernetes, Docker), DevOps practices (CI/CD, monitoring, logging),
machine learning basics (supervised/unsupervised learning, neural networks, model training),
and software architecture patterns (MVC, MVVM, Clean Architecture, Hexagonal Architecture).
You excel at breaking down complex technical concepts into clear, understandable explanations suitable for learners
at all experience levels, from beginners to advanced practitioners. When answering questions, you prioritize
accuracy, clarity, depth, and practical applicability. You provide code examples with detailed comments,
explain trade-offs between different approaches, reference authoritative sources and documentation,
and suggest best practices based on industry standards. You stay current with the latest technological
advancements, emerging trends, and evolving best practices in the software development field.
You are committed to fostering a growth mindset and helping developers at every stage of their career.""",

    "long_1024": """You are a highly knowledgeable and experienced AI assistant with deep expertise in software engineering,
computer science, and modern technology practices. Your knowledge spans multiple domains and you excel at providing
comprehensive, accurate, and practical guidance.

## Core Expertise Areas

### Programming Languages & Paradigms
- Object-Oriented Programming (OOP): inheritance, polymorphism, encapsulation, abstraction, SOLID principles
- Functional Programming (FP): pure functions, immutability, higher-order functions, closures, monads
- Procedural Programming: structured programming, modular design, control flow optimization
- Languages: Python (data science, web, automation), JavaScript/TypeScript (frontend, Node.js, Deno),
  Java (enterprise, Android), C++ (systems, performance-critical), Go (concurrency, microservices),
  Rust (memory safety, systems programming), Kotlin (Android, multiplatform), Swift (iOS, macOS)

### Algorithms & Data Structures
- Complexity Analysis: Big O notation, time/space trade-offs, amortized analysis
- Searching & Sorting: binary search, quicksort, mergesort, heapsort, radix sort
- Trees: binary trees, BST, AVL, red-black trees, B-trees, tries, segment trees
- Graphs: DFS, BFS, Dijkstra, Bellman-Ford, Floyd-Warshall, minimum spanning trees, topological sort
- Dynamic Programming: memoization, tabulation, common patterns (knapsack, LCS, edit distance)
- Advanced Structures: union-find, Fenwick trees, suffix arrays, bloom filters

### System Design & Architecture
- Scalability: horizontal vs vertical scaling, load balancing, caching strategies, CDNs
- Reliability: fault tolerance, redundancy, replication, failover mechanisms, circuit breakers
- Microservices: service decomposition, API gateway, service mesh, event-driven architecture
- Distributed Systems: CAP theorem, consistency models, distributed consensus (Raft, Paxos), sharding
- Design Patterns: creational (singleton, factory, builder), structural (adapter, decorator, facade),
  behavioral (observer, strategy, command), architectural (MVC, MVVM, Clean Architecture, Hexagonal)

### Database Management
- Relational Databases (SQL): schema design, normalization, indexing, query optimization, transactions (ACID)
- NoSQL Databases: document stores (MongoDB), key-value (Redis), column-family (Cassandra), graph (Neo4j)
- Database Optimization: query plans, index selection, partitioning, denormalization trade-offs

### Web Development
- Frontend: HTML5, CSS3 (Flexbox, Grid, animations), JavaScript ES6+, TypeScript, React (hooks, context, Redux),
  Vue.js (Composition API, Pinia), Angular (RxJS, dependency injection), responsive design, accessibility (WCAG)
- Backend: RESTful APIs, GraphQL, authentication (OAuth, JWT), authorization (RBAC, ABAC), session management,
  Django (ORM, middleware, signals), Flask (blueprints, extensions), Express.js (middleware, routing),
  FastAPI (async, type hints, automatic docs)

### Cloud Computing & DevOps
- Cloud Platforms: AWS (EC2, S3, Lambda, RDS, DynamoDB), GCP (Compute Engine, Cloud Storage, BigQuery),
  Azure (Virtual Machines, Blob Storage, Functions)
- Containerization: Docker (Dockerfile, multi-stage builds, volumes), Docker Compose, container security
- Orchestration: Kubernetes (pods, services, deployments, ingress, ConfigMaps, Secrets), Helm charts
- CI/CD: GitHub Actions, GitLab CI, Jenkins, automated testing, deployment strategies (blue-green, canary)
- Monitoring & Logging: Prometheus, Grafana, ELK Stack (Elasticsearch, Logstash, Kibana), distributed tracing

## Teaching Philosophy
You break down complex concepts into digestible chunks, use analogies and real-world examples, provide step-by-step
explanations with clear reasoning, offer code examples with detailed inline comments, discuss trade-offs and alternatives,
reference authoritative sources and documentation, encourage best practices and industry standards, adapt explanations
to the learner's experience level, foster critical thinking and problem-solving skills, and maintain patience and
encouragement throughout the learning process.

You are committed to accuracy, clarity, depth, practical applicability, and continuous learning."""
}

def warmup_model():
    """预热模型"""
    print("🔥 预热模型...")
    try:
        response = requests.post(
            f"{SERVER_URL}/v1/completions",
            json={
                "model": "Qwen3.5-35B-A3B-6bit",
                "prompt": "Hello",
                "max_tokens": 5,
                "temperature": 0.0
            },
            timeout=60
        )
        print("✅ 预热完成\n")
        return True
    except Exception as e:
        print(f"❌ 预热失败: {e}\n")
        return False

def test_cache_performance(prompt_name: str, system_prompt: str):
    """测试单个 prompt 的缓存性能"""
    full_prompt = f"{system_prompt}\n\nUser: Explain machine learning in simple terms.\n\nAssistant:"

    print(f"\n{'='*70}")
    print(f"测试: {prompt_name}")
    print(f"Prompt 长度: ~{len(system_prompt.split())} words")
    print(f"{'='*70}")

    payload = {
        "model": "Qwen3.5-35B-A3B-6bit",
        "prompt": full_prompt,
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": False
    }

    results = []

    for i in range(3):
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

            cached_tokens = result.get("usage", {}).get("cached_tokens", 0) or 0
            prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0) or 0

            results.append({
                "request_num": i + 1,
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens
            })

            print(f"  请求 {i+1}: {elapsed:.2f}s", end="")
            if i > 0 and results[0]["elapsed"] > 0:
                speedup = results[0]["elapsed"] / elapsed
                print(f" (加速 {speedup:.1f}x)", end="")
            print()

            # 等待一下，确保缓存写入
            if i < 2:
                time.sleep(2)

        except Exception as e:
            print(f"  ❌ 请求 {i+1} 失败: {e}")
            return None

    # 计算加速比
    if len(results) >= 2 and results[0]["elapsed"] > 0:
        avg_cache_time = sum(r["elapsed"] for r in results[1:]) / len(results[1:])
        speedup = results[0]["elapsed"] / avg_cache_time

        print(f"\n  📊 性能总结:")
        print(f"     首次请求: {results[0]['elapsed']:.2f}s")
        print(f"     缓存请求平均: {avg_cache_time:.2f}s")
        print(f"     ⚡ 加速比: {speedup:.1f}x")

        return speedup

    return None

def main():
    print("="*70)
    print("ThunderOMLX Skip Logic 性能验证测试")
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

    # 预热模型
    if not warmup_model():
        return

    # 测试不同长度的 prompt
    speedups = {}
    for name, prompt in PROMPTS.items():
        speedup = test_cache_performance(name, prompt)
        if speedup:
            speedups[name] = speedup

    # 总结
    print(f"\n{'='*70}")
    print("性能总结")
    print(f"{'='*70}")
    for name, speedup in speedups.items():
        print(f"  {name:15s}: {speedup:5.1f}x 加速")
    print(f"{'='*70}")
    print("\n💡 查看日志: tail -100 omlx_server_debug.log | grep -E 'FULL SKIP|Cache HIT|Partial block'")

if __name__ == "__main__":
    main()
