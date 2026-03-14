# P1-7: Adaptive Chunk Prefill 实现设计

> **目标**: 自适应分块 prefill，减少内存碎片化
> **预计工期**: 1 天

---

## 📊 问题分析

### 当前状态：固定批次 Prefill

oMLX 当前实现：
- 将整个 prompt 一次性送入 prefill
- 不考虑 prompt 长度
- 可能导致大批次内存峰值

**问题**：
1. **内存峰值**：长 prompt (4096+ tokens) 一次性 prefill 导致内存峰值过高
2. **缓存碎片化**：保存时整块写入，难以复用部分缓存
3. **无法渐进式缓存**：必须等全部 prefill 完成才能保存

### ThunderLLAMA 策略：固定 Chunk 分块

```cpp
// thunder-context.cpp
const int THUNDER_CHUNK_SIZE = 64;  // 固定 64 tokens per chunk

// Prefill in chunks
for (size_t chunk_idx = 0; chunk_idx < max_chunks; chunk_idx++) {
    int chunk_start = chunk_idx * THUNDER_CHUNK_SIZE;
    int chunk_end = min(chunk_start + THUNDER_CHUNK_SIZE, n_tokens);

    // Prefill this chunk
    prefill_tokens(chunk_start, chunk_end);

    // Save this chunk to cache
    save_chunk_to_cache(chunk_idx);
}
```

**优势**：
- ✅ 内存峰值可控（每次只处理 64 tokens）
- ✅ 渐进式缓存（每个 chunk 独立保存）
- ✅ 部分复用（可以复用部分 chunks）

**不足**：
- ❌ 固定 chunk size 不适应不同场景
- ❌ 边界不对齐时有碎片
- ❌ 小 prompt 分块反而增加开销

---

## 🏗️ oMLX 自适应策略

### 核心思想：动态 Chunk Size

根据以下因素动态调整 chunk size：

| 因素 | 影响 | 策略 |
|------|------|------|
| **Prompt 长度** | 短 prompt 分块开销大 | < 128 tokens: 不分块<br>128-1024: chunk_size=128<br>> 1024: chunk_size=256 |
| **可用内存** | 内存紧张时减小 chunk | 动态调整避免 OOM |
| **缓存边界对齐** | 对齐块边界减少碎片 | chunk_size 是 page_size 的倍数 |

### 自适应算法

```python
def compute_adaptive_chunk_size(
    prompt_length: int,
    available_memory: int,
    cache_block_size: int = 64  # Default page size
) -> int:
    """
    计算自适应 chunk size。

    Args:
        prompt_length: Prompt token 数量
        available_memory: 可用内存（bytes）
        cache_block_size: 缓存块大小（tokens）

    Returns:
        Optimal chunk size (tokens)
    """
    # Strategy 1: 短 prompt 不分块
    if prompt_length < 128:
        return prompt_length  # 不分块

    # Strategy 2: 基于 prompt 长度的基准 chunk size
    if prompt_length < 1024:
        base_chunk_size = 128
    elif prompt_length < 4096:
        base_chunk_size = 256
    else:
        base_chunk_size = 512

    # Strategy 3: 对齐缓存块边界（减少碎片）
    # chunk_size 应该是 cache_block_size 的整数倍
    aligned_chunk_size = (base_chunk_size // cache_block_size) * cache_block_size
    if aligned_chunk_size == 0:
        aligned_chunk_size = cache_block_size

    # Strategy 4: 内存约束检查
    # 假设每个 token 的 KV cache 占用 ~4KB (float16, 32 layers, 128 emb_dim)
    bytes_per_token = 4096
    chunk_memory = aligned_chunk_size * bytes_per_token

    # 如果 chunk 超过可用内存的 50%，减半
    while chunk_memory > available_memory * 0.5 and aligned_chunk_size > cache_block_size:
        aligned_chunk_size //= 2
        chunk_memory = aligned_chunk_size * bytes_per_token

    return max(aligned_chunk_size, cache_block_size)
```

---

## 📋 实现清单

### 1. 新增文件：chunk_adapter.py

**文件**: `src/omlx/cache/chunk_adapter.py` (新建)

```python
"""
自适应分块 prefill 适配器。

P1-7: Adaptive Chunk Prefill
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class AdaptiveChunkCalculator:
    """
    自适应 chunk size 计算器。

    根据 prompt 长度、可用内存和缓存块大小动态调整 chunk size。
    """

    # 默认策略参数
    SHORT_PROMPT_THRESHOLD = 128      # 短 prompt 不分块
    MEDIUM_CHUNK_SIZE = 128           # 中等 prompt 的 chunk size
    LARGE_CHUNK_SIZE = 256            # 长 prompt 的 chunk size
    VERY_LARGE_CHUNK_SIZE = 512       # 超长 prompt 的 chunk size

    MEMORY_SAFETY_RATIO = 0.5         # chunk 内存不超过可用内存的 50%
    BYTES_PER_TOKEN_ESTIMATE = 4096   # 每个 token 的 KV cache 估算（bytes）

    def __init__(
        self,
        cache_block_size: int = 64,
        enable_alignment: bool = True,
        enable_memory_check: bool = True,
    ):
        """
        初始化自适应 chunk 计算器。

        Args:
            cache_block_size: 缓存块大小（tokens），chunk size 会对齐到此值的倍数
            enable_alignment: 启用缓存块对齐
            enable_memory_check: 启用内存检查
        """
        self.cache_block_size = cache_block_size
        self.enable_alignment = enable_alignment
        self.enable_memory_check = enable_memory_check

        logger.info(
            f"AdaptiveChunkCalculator initialized: "
            f"cache_block_size={cache_block_size}, "
            f"alignment={enable_alignment}, "
            f"memory_check={enable_memory_check}"
        )

    def compute_chunk_size(
        self,
        prompt_length: int,
        available_memory: Optional[int] = None,
    ) -> int:
        """
        计算自适应 chunk size。

        Args:
            prompt_length: Prompt token 数量
            available_memory: 可用内存（bytes），None 则自动检测

        Returns:
            Optimal chunk size (tokens)
        """
        # 短 prompt 不分块
        if prompt_length < self.SHORT_PROMPT_THRESHOLD:
            logger.debug(
                f"Short prompt ({prompt_length} tokens), no chunking"
            )
            return prompt_length

        # 基于 prompt 长度选择基准 chunk size
        if prompt_length < 1024:
            base_chunk_size = self.MEDIUM_CHUNK_SIZE
        elif prompt_length < 4096:
            base_chunk_size = self.LARGE_CHUNK_SIZE
        else:
            base_chunk_size = self.VERY_LARGE_CHUNK_SIZE

        # 对齐到缓存块边界
        if self.enable_alignment:
            aligned_chunk_size = self._align_to_cache_block(base_chunk_size)
        else:
            aligned_chunk_size = base_chunk_size

        # 内存约束检查
        if self.enable_memory_check:
            if available_memory is None:
                available_memory = self._get_available_memory()

            aligned_chunk_size = self._enforce_memory_limit(
                aligned_chunk_size,
                available_memory
            )

        logger.debug(
            f"Adaptive chunk size: {aligned_chunk_size} tokens "
            f"(prompt_length={prompt_length}, base={base_chunk_size})"
        )

        return aligned_chunk_size

    def split_into_chunks(
        self,
        prompt_length: int,
        chunk_size: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        将 prompt 分割为 chunks。

        Args:
            prompt_length: Prompt token 数量
            chunk_size: Chunk size，None 则自动计算

        Returns:
            List of (start, end) tuples
        """
        if chunk_size is None:
            chunk_size = self.compute_chunk_size(prompt_length)

        chunks = []
        for start in range(0, prompt_length, chunk_size):
            end = min(start + chunk_size, prompt_length)
            chunks.append((start, end))

        logger.info(
            f"Split prompt into {len(chunks)} chunks "
            f"(prompt_length={prompt_length}, chunk_size={chunk_size})"
        )

        return chunks

    def _align_to_cache_block(self, chunk_size: int) -> int:
        """对齐到缓存块边界。"""
        aligned = (chunk_size // self.cache_block_size) * self.cache_block_size
        if aligned == 0:
            aligned = self.cache_block_size
        return aligned

    def _get_available_memory(self) -> int:
        """获取可用内存（bytes）。"""
        try:
            # macOS / Linux
            if hasattr(os, 'sysconf'):
                page_size = os.sysconf('SC_PAGE_SIZE')
                avail_pages = os.sysconf('SC_AVPHYS_PAGES')
                return page_size * avail_pages
        except (ValueError, OSError):
            pass

        # Fallback: 假设 4GB 可用
        return 4 * 1024**3

    def _enforce_memory_limit(
        self,
        chunk_size: int,
        available_memory: int,
    ) -> int:
        """根据可用内存限制 chunk size。"""
        chunk_memory = chunk_size * self.BYTES_PER_TOKEN_ESTIMATE
        max_allowed_memory = available_memory * self.MEMORY_SAFETY_RATIO

        # 如果超限，减半直到满足
        while chunk_memory > max_allowed_memory and chunk_size > self.cache_block_size:
            chunk_size //= 2
            chunk_memory = chunk_size * self.BYTES_PER_TOKEN_ESTIMATE

        return max(chunk_size, self.cache_block_size)
```

---

### 2. 集成到现有系统

#### 修改 1: prefix_cache.py

在 `PromptPrefixCache.generate()` 中添加分块 prefill 支持：

```python
# src/omlx/cache/prefix_cache.py

from .chunk_adapter import AdaptiveChunkCalculator

class PromptPrefixCache:
    def __init__(
        self,
        ...
        enable_adaptive_chunking: bool = True,  # P1-7
        cache_block_size: int = 64,             # P1-7
    ):
        ...

        # P1-7: Adaptive chunking
        self.enable_adaptive_chunking = enable_adaptive_chunking
        if enable_adaptive_chunking:
            self._chunk_calculator = AdaptiveChunkCalculator(
                cache_block_size=cache_block_size
            )
            logger.info("✅ P1-7 Adaptive Chunk Prefill enabled")
        else:
            self._chunk_calculator = None
            logger.info("Adaptive chunking disabled")

    def generate(self, prompt, ...):
        ...

        # P1-7: Adaptive chunking for long prompts
        if self.enable_adaptive_chunking and len(prompt) > 128:
            return self._generate_chunked(prompt, ...)
        else:
            return self._generate_full(prompt, ...)

    def _generate_chunked(self, prompt, ...):
        """分块 prefill 生成。"""
        chunks = self._chunk_calculator.split_into_chunks(len(prompt))

        # 逐 chunk prefill
        for chunk_idx, (start, end) in enumerate(chunks):
            chunk_prompt = prompt[start:end]

            # Prefill this chunk
            chunk_cache = self._prefill_chunk(chunk_prompt, ...)

            # Save chunk to cache manager
            if self.manager:
                chunk_hash = self._compute_chunk_hash(prompt, start, end)
                self.manager.save_block(
                    chunk_hash,
                    chunk_cache,
                    token_count=end - start,
                )

        # Generate from last chunk
        return self._generate_from_cache(...)
```

---

## 🧪 测试验证

### 功能测试

**文件**: `tests/test_adaptive_chunk.py`

```python
def test_short_prompt_no_chunking():
    """短 prompt 不分块"""
    calculator = AdaptiveChunkCalculator()

    chunk_size = calculator.compute_chunk_size(prompt_length=100)
    assert chunk_size == 100  # 不分块

    chunks = calculator.split_into_chunks(prompt_length=100)
    assert len(chunks) == 1
    assert chunks[0] == (0, 100)


def test_medium_prompt_chunking():
    """中等 prompt 分块"""
    calculator = AdaptiveChunkCalculator()

    chunk_size = calculator.compute_chunk_size(prompt_length=512)
    assert chunk_size == 128  # 基准 chunk size

    chunks = calculator.split_into_chunks(prompt_length=512)
    assert len(chunks) == 4  # 512 / 128 = 4
    assert chunks[0] == (0, 128)
    assert chunks[3] == (384, 512)


def test_alignment():
    """缓存块对齐"""
    calculator = AdaptiveChunkCalculator(cache_block_size=64)

    # 128 已经是 64 的倍数
    chunk_size = calculator.compute_chunk_size(prompt_length=512)
    assert chunk_size % 64 == 0


def test_memory_limit():
    """内存限制"""
    calculator = AdaptiveChunkCalculator()

    # 模拟低内存场景
    small_memory = 100 * 1024**2  # 100MB
    chunk_size = calculator.compute_chunk_size(
        prompt_length=4096,
        available_memory=small_memory
    )

    # Chunk size 应该被限制
    chunk_memory = chunk_size * 4096  # bytes per token
    assert chunk_memory <= small_memory * 0.5
```

---

## 📈 预期效果

### 内存优化

| Prompt 长度 | 无分块内存峰值 | 分块内存峰值 | 优化 |
|-------------|---------------|-------------|------|
| 128 tokens  | 512KB | 512KB | 无变化（不分块）|
| 512 tokens  | 2MB | 512KB | **4x 减少** |
| 2048 tokens | 8MB | 1MB | **8x 减少** |
| 4096 tokens | 16MB | 2MB | **8x 减少** |

### 碎片化减少

| 场景 | 无对齐 | 对齐到 64 | 改善 |
|------|--------|----------|------|
| 缓存块利用率 | ~70% | **~95%** | +25% |
| 内存碎片 | High | **Low** | 显著改善 |

---

## 🎯 成功标准

### 功能标准

- [ ] 短 prompt (< 128) 不分块
- [ ] 中长 prompt 自适应分块
- [ ] 缓存块边界对齐
- [ ] 内存约束检查生效

### 性能标准

- [ ] 长 prompt 内存峰值减少 > 4x
- [ ] 无性能回退（小 prompt）
- [ ] 缓存块利用率 > 90%

### 质量标准

- [ ] 测试覆盖率 > 80%
- [ ] 所有测试通过
- [ ] 代码审查通过

---

**设计版本**: v1.0
**创建日期**: 2026-03-13
**预计工期**: 1 天
