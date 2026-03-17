#!/usr/bin/env python3
"""
P2.4 智能分块系统测试

目标：验证智能分块系统的质量和性能
- 语义边界识别
- 内容类型检测
- 动态 chunk size
- 质量指标验证

对比：
- 固定 4K 分块 vs 智能分块
- 质量指标对比
- 性能开销对比
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from mlx_lm import load

from omlx.chunking import (
    intelligent_chunked_prefill,
    ContentType,
    ChunkStats,
    ChunkQuality
)


def generate_dialogue_prompt(num_tokens: int) -> str:
    """生成对话格式的 prompt"""
    system = "You are a helpful AI assistant for Python development.\n\n"

    conversations = []
    target_tokens = num_tokens - 100
    current_tokens = 0

    questions = [
        "How do I implement a binary search tree in Python?",
        "What's the best way to handle exceptions in async code?",
        "Can you explain decorators and how they work?",
        "How do I optimize memory usage in data processing?",
        "What's the difference between list and generator?",
    ]

    answers = [
        "To implement a binary search tree, you'll need a Node class and a BST class. " * 10,
        "In async code, use try-except blocks within async functions and handle CancelledError. " * 10,
        "Decorators are functions that modify the behavior of other functions using the @ syntax. " * 10,
        "Use generators, avoid copying data, and release references when done. " * 10,
        "Lists store all elements in memory while generators produce items on-demand. " * 10,
    ]

    turn = 0
    while current_tokens < target_tokens:
        q = questions[turn % len(questions)]
        a = answers[turn % len(answers)]
        conversations.append(f"User: {q}\n\nAssistant: {a}\n\n")
        current_tokens += len(q.split()) + len(a.split())
        turn += 1

    final_question = "User: Can you summarize the key points from above?\n\nAssistant:"
    prompt = f"{system}{''.join(conversations)}{final_question}"
    return prompt


def generate_document_prompt(num_tokens: int) -> str:
    """生成文档格式的 prompt"""
    doc = """
# Python Best Practices Guide

## Introduction

Python is a powerful programming language that emphasizes code readability and simplicity.
Following best practices helps maintain clean, efficient, and maintainable code.

## Code Style

### PEP 8 Compliance

Always follow PEP 8 guidelines for code formatting. Use 4 spaces for indentation,
not tabs. Limit lines to 79 characters for code and 72 for docstrings.

### Naming Conventions

Use snake_case for function and variable names, PascalCase for class names,
and UPPER_CASE for constants. Be descriptive but concise in naming.

## Error Handling

### Proper Exception Handling

Always catch specific exceptions rather than using bare except clauses.
This helps identify and fix bugs more easily. Use context managers with
the 'with' statement for resource management.

### Custom Exceptions

Create custom exception classes for domain-specific errors. This makes
error handling more semantic and easier to debug.

## Performance Optimization

### Use Built-in Functions

Python's built-in functions are implemented in C and are highly optimized.
Use them whenever possible instead of reimplementing functionality.

### Generator Expressions

For large datasets, use generator expressions instead of list comprehensions
to save memory. Generators produce items on-demand rather than storing
everything in memory.

## Testing

### Unit Tests

Write unit tests for all critical functionality. Use pytest or unittest
framework. Aim for high code coverage but focus on meaningful tests.

### Integration Tests

Test how components work together. Mock external dependencies to keep
tests fast and reliable.

"""

    # 重复文档内容直到达到目标 tokens
    chars_per_token = 4
    target_chars = num_tokens * chars_per_token

    repeated_doc = ""
    while len(repeated_doc) < target_chars:
        repeated_doc += doc

    prompt = repeated_doc[:target_chars] + "\n\nSummarize the key points:"
    return prompt


def generate_code_prompt(num_tokens: int) -> str:
    """生成代码格式的 prompt"""
    code_blocks = [
        """
```python
def binary_search(arr: List[int], target: int) -> int:
    \"\"\"Binary search implementation\"\"\"
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```
""",
        """
```python
class TreeNode:
    \"\"\"Binary tree node\"\"\"
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder_traversal(root: TreeNode) -> List[int]:
    \"\"\"Inorder traversal of binary tree\"\"\"
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)

    traverse(root)
    return result
```
""",
        """
```python
async def fetch_data(url: str) -> dict:
    \"\"\"Async HTTP request\"\"\"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def process_urls(urls: List[str]) -> List[dict]:
    \"\"\"Process multiple URLs concurrently\"\"\"
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)
```
"""
    ]

    # 重复代码块直到达到目标 tokens
    chars_per_token = 4
    target_chars = num_tokens * chars_per_token

    repeated_code = ""
    idx = 0
    while len(repeated_code) < target_chars:
        repeated_code += code_blocks[idx % len(code_blocks)]
        idx += 1

    prompt = repeated_code[:target_chars] + "\n\nExplain the code above:"
    return prompt


def fixed_chunked_generate(
    model,
    tokenizer,
    prompt: str,
    chunk_size: int = 4096,
    max_tokens: int = 10
) -> Tuple[str, ChunkStats]:
    """
    固定分块（对照组）
    """
    print("\n" + "="*80)
    print(f"固定分块 (chunk_size={chunk_size})")
    print("="*80)

    from omlx.chunking.types import ChunkStats
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

    print(f"📊 Split into {len(chunks)} chunks (fixed size)")

    # 创建 cache
    cache = [KVCache() for _ in range(len(model.model.layers))]
    stats = ChunkStats()

    # 逐块 prefill
    for i, chunk in enumerate(chunks):
        chunk_start_time = time.perf_counter()

        chunk_mx = mx.array([chunk])
        logits = model(chunk_mx, cache=cache)
        mx.eval(logits)
        mx.eval([c.keys for c in cache])

        chunk_time = time.perf_counter() - chunk_start_time
        stats.add_chunk(len(chunk), chunk_time, None)

        print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} tokens, "
              f"{chunk_time:.2f}s, {len(chunk)/chunk_time:.1f} tok/s")

    # 生成
    print(f"\n生成 {max_tokens} tokens...")
    generated_tokens = []
    next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
    generated_tokens.append(next_token)

    for _ in range(max_tokens - 1):
        logits = model(mx.array([[next_token]]), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[0, -1, :], axis=-1).item()
        generated_tokens.append(next_token)

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ 总耗时: {elapsed:.2f}s")
    print(f"📊 统计: {stats}")

    response = tokenizer.decode(generated_tokens)
    return response, stats


def compare_quality(
    fixed_stats: ChunkStats,
    intelligent_quality: ChunkQuality,
    intelligent_stats: ChunkStats
):
    """对比质量指标"""
    print("\n" + "="*80)
    print("质量对比")
    print("="*80)

    # 固定分块质量（模拟）
    print("\n固定分块:")
    print(f"  边界完整性: N/A (无语义边界)")
    print(f"  Size 均匀性: 1.00 (固定大小)")
    print(f"  跨边界率: 未知")
    print(f"  综合得分: N/A")

    # 智能分块质量
    print("\n智能分块:")
    print(f"  边界完整性: {intelligent_quality.boundary_integrity:.2%}")
    print(f"  Size 均匀性: {intelligent_quality.size_uniformity:.2%}")
    print(f"  跨边界率: {intelligent_quality.cross_boundary_rate:.2%}")
    print(f"  综合得分: {intelligent_quality.overall_score:.2%}")

    if intelligent_quality.is_high_quality:
        print("\n✅ 智能分块达到高质量标准（>80%）")
    else:
        print(f"\n⚠️  智能分块质量需改进（{intelligent_quality.overall_score:.2%}）")

    # 性能对比
    print("\n" + "="*80)
    print("性能对比")
    print("="*80)

    print(f"\n固定分块:")
    print(f"  总 tokens: {fixed_stats.total_tokens}")
    print(f"  总耗时: {fixed_stats.total_time:.2f}s")
    print(f"  吞吐量: {fixed_stats.tokens_per_second:.1f} tok/s")
    print(f"  平均 chunk 大小: {fixed_stats.avg_chunk_size:.0f} tokens")

    print(f"\n智能分块:")
    print(f"  总 tokens: {intelligent_stats.total_tokens}")
    print(f"  总耗时: {intelligent_stats.total_time:.2f}s")
    print(f"  吞吐量: {intelligent_stats.tokens_per_second:.1f} tok/s")
    print(f"  平均 chunk 大小: {intelligent_stats.avg_chunk_size:.0f} tokens")

    if fixed_stats.total_time > 0:
        overhead = ((intelligent_stats.total_time - fixed_stats.total_time) /
                   fixed_stats.total_time) * 100
        print(f"\n性能开销: {overhead:+.1f}%")

        if abs(overhead) < 10:
            print("✅ 性能接近（<10% 差异）")
        elif overhead > 0 and overhead < 20:
            print("⚠️  轻微性能开销（<20%）")
        else:
            print(f"⚠️  性能差异较大（{overhead:+.1f}%）")


def test_content_type(
    model,
    tokenizer,
    content_type: str,
    prompt: str
):
    """测试特定内容类型"""
    print("\n\n" + "#"*80)
    print(f"# 测试: {content_type}")
    print("#"*80)

    actual_tokens = len(tokenizer.encode(prompt))
    print(f"\n📝 Prompt: {actual_tokens} tokens")
    print(f"📝 Preview: {prompt[:200]}...")

    # 固定分块
    fixed_output, fixed_stats = fixed_chunked_generate(
        model, tokenizer, prompt, chunk_size=4096, max_tokens=10
    )

    # 智能分块
    print("\n" + "="*80)
    print(f"智能分块")
    print("="*80)

    intelligent_output, intelligent_stats, intelligent_quality = intelligent_chunked_prefill(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_size=4096,
        flexibility=0.125,
        max_tokens=10,
        verbose=True
    )

    # 对比
    compare_quality(fixed_stats, intelligent_quality, intelligent_stats)

    # 输出对比
    print("\n" + "="*80)
    print("输出对比")
    print("="*80)
    print(f"固定分块输出: {fixed_output[:100]}...")
    print(f"智能分块输出: {intelligent_output[:100]}...")


def main():
    """运行智能分块系统测试"""
    print("\n🧪 P2.4 智能分块系统测试")
    print("="*80)
    print("目标: 验证智能分块系统的质量和性能")
    print("="*80)

    # 使用 3B 模型（速度快）
    model_path = "mlx-community/Qwen2.5-3B-Instruct-8bit"
    print(f"\n📦 Loading model: {model_path}")

    try:
        model, tokenizer = load(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # 测试 1: 对话格式
    prompt_dialogue = generate_dialogue_prompt(16 * 1024)
    test_content_type(model, tokenizer, "对话格式 (16K tokens)", prompt_dialogue)

    # 测试 2: 文档格式
    prompt_document = generate_document_prompt(16 * 1024)
    test_content_type(model, tokenizer, "文档格式 (16K tokens)", prompt_document)

    # 测试 3: 代码格式
    prompt_code = generate_code_prompt(16 * 1024)
    test_content_type(model, tokenizer, "代码格式 (16K tokens)", prompt_code)

    print("\n\n" + "#"*80)
    print("# 总结")
    print("#"*80)
    print("\n智能分块系统的优势:")
    print("  1. 语义边界识别（对话/段落/代码块）")
    print("  2. 内容类型自适应（dialogue/document/code）")
    print("  3. 动态 chunk size（512-6K，目标 4K ±12.5%）")
    print("  4. 质量指标验证（boundary_integrity >95%）")
    print("  5. 自动回退机制（质量不达标时回退到固定分块）")

    return 0


if __name__ == "__main__":
    exit(main())
