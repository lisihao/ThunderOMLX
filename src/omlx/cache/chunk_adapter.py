# SPDX-License-Identifier: Apache-2.0
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

    def get_stats(self) -> dict:
        """
        获取配置统计信息。

        Returns:
            配置字典
        """
        return {
            'cache_block_size': self.cache_block_size,
            'enable_alignment': self.enable_alignment,
            'enable_memory_check': self.enable_memory_check,
            'short_prompt_threshold': self.SHORT_PROMPT_THRESHOLD,
            'medium_chunk_size': self.MEDIUM_CHUNK_SIZE,
            'large_chunk_size': self.LARGE_CHUNK_SIZE,
            'very_large_chunk_size': self.VERY_LARGE_CHUNK_SIZE,
        }
