"""
智能分块系统 - 动态分块算法

核心算法：Greedy Boundary-Aware Packing
在保持语义完整性的前提下，最大化 chunk size 利用率
"""

from typing import List, Optional
from .types import Boundary, Chunk, ContentType


class DynamicChunker:
    """动态分块器"""

    def __init__(
        self,
        target_size: int = 4096,
        flexibility: float = 0.125,  # ±12.5%
    ):
        """
        初始化动态分块器

        Args:
            target_size: 目标 chunk size
            flexibility: 弹性范围（相对于 target_size 的比例）
        """
        self.target_size = target_size
        self.flexibility = flexibility

        # 计算弹性范围
        flex_tokens = int(target_size * flexibility)
        self.min_size = target_size - flex_tokens
        self.max_size = target_size + flex_tokens

    def chunk(
        self,
        tokens: List[int],
        boundaries: List[Boundary],
        content_type: ContentType
    ) -> List[Chunk]:
        """
        动态分块

        Args:
            tokens: Token 列表
            boundaries: 边界列表
            content_type: 内容类型

        Returns:
            List[Chunk]: 分块列表
        """
        # 根据内容类型调整参数
        target_size, min_size, max_size = self._adjust_params_for_content_type(content_type)

        chunks = []
        current_start = 0

        while current_start < len(tokens):
            # 1. 理想切分点
            ideal_end = current_start + target_size

            # 2. 如果剩余不足一个 chunk，全部打包
            if ideal_end >= len(tokens):
                chunk = Chunk(
                    tokens=tokens[current_start:],
                    start=current_start,
                    end=len(tokens),
                    boundary=None
                )
                chunks.append(chunk)
                break

            # 3. 在弹性范围内查找最强边界
            search_start = max(current_start + min_size, current_start)
            search_end = min(ideal_end + (max_size - target_size), len(tokens))

            best_boundary = self._find_best_boundary(
                boundaries,
                search_start,
                search_end,
                ideal_end,
                current_start
            )

            if best_boundary:
                chunk_end = best_boundary.token_offset
            else:
                # 无边界：使用理想点
                chunk_end = ideal_end

            # 4. 创建 chunk
            chunk = Chunk(
                tokens=tokens[current_start:chunk_end],
                start=current_start,
                end=chunk_end,
                boundary=best_boundary
            )
            chunks.append(chunk)

            current_start = chunk_end

        return chunks

    def _adjust_params_for_content_type(
        self,
        content_type: ContentType
    ) -> tuple[int, int, int]:
        """
        根据内容类型调整参数

        Args:
            content_type: 内容类型

        Returns:
            (target_size, min_size, max_size)
        """
        if content_type == ContentType.DIALOGUE:
            # 对话模式：最小 chunk size 可以更小（允许短对话）
            return self.target_size, max(512, self.min_size), self.max_size

        elif content_type == ContentType.CODE:
            # 代码模式：最小 chunk size 可以更小（允许短函数）
            return self.target_size, max(256, self.min_size // 2), self.max_size

        elif content_type == ContentType.DOCUMENT:
            # 文档模式：最大 chunk size 可以更大（允许长段落）
            return self.target_size, self.min_size, min(6144, int(self.max_size * 1.5))

        else:
            # 通用/混合模式：使用默认参数
            return self.target_size, self.min_size, self.max_size

    def _find_best_boundary(
        self,
        boundaries: List[Boundary],
        search_start: int,
        search_end: int,
        ideal: int,
        chunk_start: int
    ) -> Optional[Boundary]:
        """
        在范围内查找最强边界

        Args:
            boundaries: 边界列表
            search_start: 搜索起始位置
            search_end: 搜索结束位置
            ideal: 理想切分点
            chunk_start: 当前 chunk 起始位置

        Returns:
            Optional[Boundary]: 最佳边界（如果有）
        """
        # 筛选候选边界
        candidates = [
            b for b in boundaries
            if search_start <= b.token_offset <= search_end
            and b.token_offset > chunk_start
        ]

        if not candidates:
            return None

        # 计算每个候选边界的得分
        # 得分 = 边界强度 * 1000 - 距离理想点的距离
        def score(b: Boundary) -> float:
            strength_score = b.strength * 1000  # 强度权重
            distance_penalty = abs(b.token_offset - ideal)  # 距离惩罚
            return strength_score - distance_penalty

        # 选择得分最高的边界
        candidates.sort(key=score, reverse=True)

        return candidates[0]

    def chunk_fixed(
        self,
        tokens: List[int],
        chunk_size: int
    ) -> List[Chunk]:
        """
        固定大小分块（回退方案）

        Args:
            tokens: Token 列表
            chunk_size: Chunk 大小

        Returns:
            List[Chunk]: 分块列表
        """
        chunks = []
        current_start = 0

        while current_start < len(tokens):
            chunk_end = min(current_start + chunk_size, len(tokens))

            chunk = Chunk(
                tokens=tokens[current_start:chunk_end],
                start=current_start,
                end=chunk_end,
                boundary=None  # 固定切分没有边界
            )
            chunks.append(chunk)

            current_start = chunk_end

        return chunks
