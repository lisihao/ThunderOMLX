# 🧠 Intelligent Chunk System - 设计文档

> **目标**: 设计业界最先进的智能分块系统，吊打所有竞品
> **核心理念**: 上下文感知 + 自适应 + 零性能损失

---

## 🎯 设计目标

### 1. 吊打竞品的核心差异

**当前业界方案（LangChain/LlamaIndex）**：
- ❌ 固定 chunk size（512/1024/2048）
- ❌ 简单滑动窗口 + overlap
- ❌ 不感知内容类型
- ❌ 切断语义边界
- ❌ 无法自适应

**我们的方案**：
- ✅ 动态 chunk size（512-8192）
- ✅ 语义边界感知（段落/句子/函数）
- ✅ 内容类型自适应（对话/文档/代码）
- ✅ 零拷贝 cache 累积
- ✅ 自适应性能优化

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Intelligent Chunk System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Prompt (text/tokens)                                │
│     ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 1: Content Type Detection                   │    │
│  │  - 对话检测 (User/Assistant 模式)                  │    │
│  │  - 文档检测 (连续段落模式)                         │    │
│  │  - 代码检测 (```code``` / function/class)         │    │
│  │  - 混合检测 (多类型切换)                           │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   ↓                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 2: Semantic Boundary Extraction             │    │
│  │  - 段落边界: \n\n (强边界)                         │    │
│  │  - 句子边界: . ! ? (中边界)                        │    │
│  │  - 对话边界: User:/Assistant: (强边界)             │    │
│  │  - 代码边界: function/class/} (强边界)             │    │
│  │  - Token offset mapping                            │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   ↓                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 3: Dynamic Chunking Strategy                │    │
│  │                                                     │    │
│  │  输入: boundaries + content_type + constraints     │    │
│  │  输出: chunks (variable size)                      │    │
│  │                                                     │    │
│  │  算法: Greedy Boundary-Aware Packing               │    │
│  │  - 目标 chunk size: 4096 tokens                    │    │
│  │  - 弹性范围: [3584, 4608] (±12.5%)                │    │
│  │  - 边界权重: 强边界优先，弱边界次之                │    │
│  │  - 跨边界惩罚: 避免切断语义单元                    │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   ↓                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 4: Chunk Quality Validation                 │    │
│  │  - 边界完整性检查                                  │    │
│  │  - Size 分布均匀性                                 │    │
│  │  - 跨边界率 < 5%                                   │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   ↓                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Phase 5: Prefill with Cache Accumulation          │    │
│  │  - 逐块 model(chunk, cache=cache)                  │    │
│  │  - 零拷贝 KVCache 累积                             │    │
│  │  - 进度监控 + 错误恢复                             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 Phase 1: Content Type Detection

### 检测算法

**对话检测**:
```python
def is_dialogue(text: str) -> bool:
    """检测是否为对话格式"""
    patterns = [
        r"User:\s*",
        r"Assistant:\s*",
        r"Human:\s*",
        r"AI:\s*",
        r"<\|im_start\|>user",
        r"<\|im_start\|>assistant",
    ]

    matches = sum(bool(re.search(p, text, re.IGNORECASE)) for p in patterns)

    # 至少出现 2 次对话标记 → 对话模式
    return matches >= 2
```

**文档检测**:
```python
def is_document(text: str) -> bool:
    """检测是否为文档格式"""
    # 连续段落模式
    paragraphs = text.split("\n\n")

    # 段落数量 > 3，且平均长度 > 100 字符
    if len(paragraphs) >= 3:
        avg_len = sum(len(p) for p in paragraphs) / len(paragraphs)
        return avg_len > 100

    return False
```

**代码检测**:
```python
def is_code(text: str) -> bool:
    """检测是否为代码格式"""
    code_indicators = [
        r"```",  # Markdown code block
        r"def\s+\w+\s*\(",  # Python function
        r"function\s+\w+\s*\(",  # JavaScript function
        r"class\s+\w+",  # Class definition
        r"import\s+\w+",  # Import statement
        r"#include\s*<",  # C/C++ include
    ]

    matches = sum(bool(re.search(p, text)) for p in code_indicators)

    # 至少匹配 2 个代码特征 → 代码模式
    return matches >= 2
```

**混合检测**:
```python
def detect_content_type(text: str) -> ContentType:
    """综合检测内容类型"""
    scores = {
        "dialogue": is_dialogue(text),
        "document": is_document(text),
        "code": is_code(text),
    }

    # 多类型混合
    true_count = sum(scores.values())

    if true_count >= 2:
        return ContentType.MIXED
    elif scores["dialogue"]:
        return ContentType.DIALOGUE
    elif scores["code"]:
        return ContentType.CODE
    elif scores["document"]:
        return ContentType.DOCUMENT
    else:
        return ContentType.GENERIC
```

---

## 📍 Phase 2: Semantic Boundary Extraction

### 边界类型

| 边界类型 | 强度 | 示例 | 优先级 |
|----------|------|------|--------|
| **对话边界** | 强 | `User:` / `Assistant:` | 1 (最高) |
| **代码块边界** | 强 | `\`\`\`python` / `}` (函数结束) | 1 |
| **段落边界** | 强 | `\n\n` (双换行) | 1 |
| **句子边界** | 中 | `. ` / `! ` / `? ` | 2 |
| **逗号边界** | 弱 | `, ` | 3 (最低) |

### Token Offset Mapping

**关键**: 需要将 text offset 映射到 token offset

```python
def extract_boundaries_with_offsets(
    text: str,
    tokens: List[int],
    tokenizer: Tokenizer
) -> List[Boundary]:
    """提取语义边界 + token offset"""

    # 1. Text offset → Token offset mapping
    # 使用 tokenizer 的 offset_mapping (如果支持)
    # 或者使用 binary search

    # 2. 提取边界
    boundaries = []

    # 强边界: 段落
    for match in re.finditer(r"\n\n", text):
        text_offset = match.end()
        token_offset = text_to_token_offset(text_offset, tokens, tokenizer)
        boundaries.append(Boundary(
            text_offset=text_offset,
            token_offset=token_offset,
            type=BoundaryType.PARAGRAPH,
            strength=1.0
        ))

    # 强边界: 对话
    for pattern in [r"User:\s*", r"Assistant:\s*"]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            text_offset = match.end()
            token_offset = text_to_token_offset(text_offset, tokens, tokenizer)
            boundaries.append(Boundary(
                text_offset=text_offset,
                token_offset=token_offset,
                type=BoundaryType.DIALOGUE,
                strength=1.0
            ))

    # 强边界: 代码块
    for match in re.finditer(r"```[\w]*\n", text):
        text_offset = match.end()
        token_offset = text_to_token_offset(text_offset, tokens, tokenizer)
        boundaries.append(Boundary(
            text_offset=text_offset,
            token_offset=token_offset,
            type=BoundaryType.CODE_BLOCK,
            strength=1.0
        ))

    # 中边界: 句子
    for match in re.finditer(r"[.!?]\s+", text):
        text_offset = match.end()
        token_offset = text_to_token_offset(text_offset, tokens, tokenizer)
        boundaries.append(Boundary(
            text_offset=text_offset,
            token_offset=token_offset,
            type=BoundaryType.SENTENCE,
            strength=0.5
        ))

    # 排序 + 去重
    boundaries.sort(key=lambda b: b.token_offset)
    boundaries = deduplicate_boundaries(boundaries)

    return boundaries
```

---

## 🧩 Phase 3: Dynamic Chunking Strategy

### 核心算法: Greedy Boundary-Aware Packing

**目标**: 在保持语义完整性的前提下，最大化 chunk size 利用率

**算法**:
```python
def dynamic_chunk(
    tokens: List[int],
    boundaries: List[Boundary],
    target_size: int = 4096,
    min_size: int = 3584,  # -12.5%
    max_size: int = 4608,  # +12.5%
) -> List[Chunk]:
    """动态分块算法"""

    chunks = []
    current_start = 0

    while current_start < len(tokens):
        # 1. 理想切分点: current_start + target_size
        ideal_end = current_start + target_size

        # 2. 如果理想点在弹性范围内，直接切分
        if ideal_end <= len(tokens) and ideal_end <= max_size:
            # 查找 [ideal_end - flex, ideal_end + flex] 范围内的最强边界
            flex = target_size - min_size  # 512 tokens

            best_boundary = find_best_boundary(
                boundaries,
                ideal_end - flex,
                ideal_end + flex,
                current_start
            )

            if best_boundary:
                chunk_end = best_boundary.token_offset
            else:
                # 无边界：使用理想点
                chunk_end = ideal_end
        else:
            # 剩余不足一个 chunk：全部打包
            chunk_end = len(tokens)

        # 3. 创建 chunk
        chunk = Chunk(
            tokens=tokens[current_start:chunk_end],
            start=current_start,
            end=chunk_end,
            boundary=best_boundary if best_boundary else None
        )
        chunks.append(chunk)

        current_start = chunk_end

    return chunks


def find_best_boundary(
    boundaries: List[Boundary],
    search_start: int,
    search_end: int,
    chunk_start: int
) -> Optional[Boundary]:
    """在范围内查找最强边界"""

    candidates = [
        b for b in boundaries
        if search_start <= b.token_offset <= search_end
        and b.token_offset > chunk_start
    ]

    if not candidates:
        return None

    # 按边界强度 + 距离理想点的距离排序
    # 优先选择强边界，其次选择距离理想点近的
    ideal = (search_start + search_end) // 2

    def score(b: Boundary) -> float:
        strength_score = b.strength * 1000  # 强度权重
        distance_penalty = abs(b.token_offset - ideal)  # 距离惩罚
        return strength_score - distance_penalty

    candidates.sort(key=score, reverse=True)

    return candidates[0]
```

### 场景自适应策略

**对话模式**:
- 强制在 `User:` / `Assistant:` 边界切分
- 确保每个 chunk 包含完整的对话轮次
- 最小 chunk size 放宽到 512 tokens（允许短对话）

**文档模式**:
- 优先在段落边界切分
- 次选句子边界
- 最大 chunk size 可扩展到 6144 tokens（长段落）

**代码模式**:
- 强制在函数/类边界切分
- 保持完整的代码块
- 最小 chunk size 可低至 256 tokens（短函数）

**混合模式**:
- 动态识别内容类型切换点
- 在切换点强制切分
- 每个 chunk 保持单一内容类型

---

## ✅ Phase 4: Chunk Quality Validation

### 验证指标

**1. 边界完整性**:
```python
def validate_boundary_integrity(chunks: List[Chunk]) -> float:
    """验证边界完整性"""

    boundary_cuts = sum(1 for c in chunks if c.boundary is not None)
    total_cuts = len(chunks) - 1

    # 边界切分率
    integrity_rate = boundary_cuts / total_cuts if total_cuts > 0 else 1.0

    # 目标: > 95%
    return integrity_rate
```

**2. Size 分布均匀性**:
```python
def validate_size_distribution(chunks: List[Chunk], target_size: int) -> float:
    """验证 size 分布均匀性"""

    sizes = [len(c.tokens) for c in chunks]
    avg_size = sum(sizes) / len(sizes)

    # 标准差
    variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
    std_dev = variance ** 0.5

    # 变异系数 (CV)
    cv = std_dev / avg_size

    # 目标: CV < 0.2 (20%)
    return 1.0 - min(cv, 1.0)
```

**3. 跨边界率**:
```python
def validate_cross_boundary_rate(chunks: List[Chunk]) -> float:
    """验证跨边界率"""

    # 切分点不在边界上的比例
    non_boundary_cuts = sum(1 for c in chunks if c.boundary is None)
    total_cuts = len(chunks) - 1

    cross_rate = non_boundary_cuts / total_cuts if total_cuts > 0 else 0.0

    # 目标: < 5%
    return 1.0 - cross_rate
```

**综合质量分数**:
```python
def calculate_quality_score(chunks: List[Chunk], target_size: int) -> float:
    """计算综合质量分数"""

    integrity = validate_boundary_integrity(chunks)
    uniformity = validate_size_distribution(chunks, target_size)
    boundary_rate = validate_cross_boundary_rate(chunks)

    # 加权平均
    score = (
        integrity * 0.5 +      # 边界完整性权重 50%
        uniformity * 0.3 +     # 均匀性权重 30%
        boundary_rate * 0.2    # 跨边界率权重 20%
    )

    return score
```

---

## 🚀 Phase 5: Prefill with Cache Accumulation

### 优化后的 Prefill

```python
def intelligent_chunked_prefill(
    model,
    tokenizer,
    prompt: str,
    target_chunk_size: int = 4096,
) -> Tuple[KVCache, ChunkStats]:
    """智能分块 prefill"""

    # 1. Tokenize
    tokens = tokenizer.encode(prompt)

    # 2. Content Type Detection
    content_type = detect_content_type(prompt)

    # 3. Semantic Boundary Extraction
    boundaries = extract_boundaries_with_offsets(prompt, tokens, tokenizer)

    # 4. Dynamic Chunking
    chunks = dynamic_chunk(
        tokens,
        boundaries,
        target_size=target_chunk_size,
        content_type=content_type
    )

    # 5. Quality Validation
    quality_score = calculate_quality_score(chunks, target_chunk_size)

    if quality_score < 0.8:
        # 质量不达标：回退到固定切分
        chunks = fixed_chunk(tokens, target_chunk_size)

    # 6. Create KVCache
    cache = [KVCache() for _ in range(len(model.model.layers))]

    # 7. Prefill
    stats = ChunkStats()

    for i, chunk in enumerate(chunks):
        chunk_mx = mx.array([chunk.tokens])

        start_time = time.perf_counter()
        logits = model(chunk_mx, cache=cache)
        mx.eval(logits)
        mx.eval([c.keys for c in cache])
        elapsed = time.perf_counter() - start_time

        stats.add_chunk(
            size=len(chunk.tokens),
            time=elapsed,
            boundary_type=chunk.boundary.type if chunk.boundary else None
        )

    return cache, stats
```

---

## 📊 性能优化

### 1. Tokenizer Offset Mapping 缓存

**问题**: 每次查找 text → token offset 都需要重新 tokenize

**解决**: 预计算 offset mapping，缓存到内存

```python
def build_offset_mapping(
    text: str,
    tokens: List[int],
    tokenizer: Tokenizer
) -> List[Tuple[int, int]]:
    """构建 offset mapping"""

    # 如果 tokenizer 支持 return_offsets_mapping
    if hasattr(tokenizer, "encode_with_offsets"):
        _, offsets = tokenizer.encode_with_offsets(text)
        return offsets

    # 否则，手动构建
    # (token_idx) -> (start_char, end_char)
    offsets = []
    current_pos = 0

    for token_id in tokens:
        token_text = tokenizer.decode([token_id])
        start = text.find(token_text, current_pos)

        if start == -1:
            # 特殊 token（如 <s>, </s>）
            offsets.append((current_pos, current_pos))
        else:
            end = start + len(token_text)
            offsets.append((start, end))
            current_pos = end

    return offsets
```

### 2. Boundary Extraction 并行化

**对于长文本（> 100K tokens）**：
- 将文本分成多个段
- 并行提取每段的边界
- 合并结果

```python
def extract_boundaries_parallel(
    text: str,
    tokens: List[int],
    tokenizer: Tokenizer,
    num_workers: int = 4
) -> List[Boundary]:
    """并行提取边界"""

    from concurrent.futures import ThreadPoolExecutor

    # 分段
    segment_size = len(text) // num_workers
    segments = []

    for i in range(num_workers):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_workers - 1 else len(text)
        segments.append((start, end, text[start:end]))

    # 并行提取
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(extract_boundaries_segment, seg, tokens, tokenizer)
            for seg in segments
        ]

        results = [f.result() for f in futures]

    # 合并
    all_boundaries = []
    for boundaries in results:
        all_boundaries.extend(boundaries)

    # 去重 + 排序
    all_boundaries.sort(key=lambda b: b.token_offset)
    all_boundaries = deduplicate_boundaries(all_boundaries)

    return all_boundaries
```

---

## 🏆 与竞品对比

| 特性 | LangChain | LlamaIndex | Ours |
|------|-----------|------------|------|
| **固定 chunk size** | ✅ | ✅ | ❌ |
| **动态 chunk size** | ❌ | ❌ | ✅ (512-8192) |
| **语义边界感知** | ❌ | ⚠️ (简单) | ✅ (多层次) |
| **内容类型检测** | ❌ | ❌ | ✅ (对话/文档/代码) |
| **场景自适应** | ❌ | ❌ | ✅ |
| **质量验证** | ❌ | ❌ | ✅ (三维指标) |
| **性能优化** | ❌ | ❌ | ✅ (缓存 + 并行) |
| **零拷贝 cache** | N/A | N/A | ✅ (MLX 原地修改) |

---

## 🎯 实现路线图

### Phase 1: 基础实现 (1-2 天)
- [ ] Content Type Detection
- [ ] Semantic Boundary Extraction (基础)
- [ ] Dynamic Chunking (Greedy 算法)
- [ ] 集成到 test_p2.3_chunked_prefill.py

### Phase 2: 场景自适应 (1 天)
- [ ] 对话模式优化
- [ ] 文档模式优化
- [ ] 代码模式优化
- [ ] 混合模式处理

### Phase 3: 质量保证 (1 天)
- [ ] Chunk Quality Validation
- [ ] 自动回退机制（质量不达标 → 固定切分）
- [ ] 统计和监控

### Phase 4: 性能优化 (1 天)
- [ ] Offset Mapping 缓存
- [ ] Boundary Extraction 并行化
- [ ] Benchmark 对比（vs 固定切分）

### Phase 5: 文档和测试 (1 天)
- [ ] 完整的单元测试
- [ ] 端到端测试（128K-256K）
- [ ] 技术文档
- [ ] 性能报告

---

## 📈 预期效果

### 定量指标

| 指标 | 固定切分 | 智能切分 | 改进 |
|------|----------|----------|------|
| 边界完整性 | ~50% | **>95%** | +90% |
| 跨边界率 | ~50% | **<5%** | -90% |
| 输出质量 | 99.88% | **>99.95%** | +0.07% |
| Size 均匀性 (CV) | ~0.3 | **<0.15** | -50% |

### 定性优势

1. **更好的语义连贯性**
   - 对话轮次完整
   - 段落不被切断
   - 代码函数完整

2. **更高的 cache 利用率**
   - 边界对齐 → 更高的 cache 命中率
   - 减少冗余计算

3. **更强的可扩展性**
   - 支持新的内容类型
   - 支持自定义边界规则

---

## 🚀 核心竞争力

### 1. 多层次边界检测

**业界唯一**: 同时支持强/中/弱三层边界

### 2. 场景自适应

**业界唯一**: 自动识别内容类型并调整策略

### 3. 质量驱动

**业界唯一**: 质量不达标自动回退

### 4. 零性能损失

**业界领先**: 缓存 + 并行，overhead < 5%

---

## 💡 未来展望

### 1. 机器学习优化

**训练一个轻量级边界分类器**:
- 输入: 前后 N tokens
- 输出: 边界强度 (0-1)
- 基于历史数据训练

### 2. 动态调整

**根据运行时反馈调整策略**:
- cache 命中率低 → 调整边界权重
- 生成质量下降 → 增大 chunk size

### 3. 多模态支持

**扩展到图像/音频**:
- 图像: 按场景切分
- 音频: 按静音段切分

---

*Intelligent Chunk System v1.0 Design*
*设计于: 2026-03-17*
*目标: 吊打所有竞品 🚀*
