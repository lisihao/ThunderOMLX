"""
智能分块系统 - 质量验证

验证分块质量，确保 boundary_integrity >95%, size uniformity CV <0.15, cross_boundary_rate <5%
"""

from typing import List
import math
from .types import Chunk, ChunkQuality, Boundary, BoundaryType


class QualityValidator:
    """分块质量验证器"""

    def __init__(
        self,
        min_boundary_integrity: float = 0.95,
        max_size_cv: float = 0.15,
        max_cross_boundary_rate: float = 0.05
    ):
        """
        初始化质量验证器

        Args:
            min_boundary_integrity: 最小边界完整性（默认 0.95）
            max_size_cv: 最大 size 变异系数（默认 0.15）
            max_cross_boundary_rate: 最大跨边界率（默认 0.05）
        """
        self.min_boundary_integrity = min_boundary_integrity
        self.max_size_cv = max_size_cv
        self.max_cross_boundary_rate = max_cross_boundary_rate

    def validate(
        self,
        chunks: List[Chunk],
        boundaries: List[Boundary],
        total_tokens: int
    ) -> ChunkQuality:
        """
        验证分块质量

        Args:
            chunks: 分块列表
            boundaries: 边界列表
            total_tokens: 总 token 数

        Returns:
            ChunkQuality: 质量指标
        """
        boundary_integrity = self._calculate_boundary_integrity(chunks, boundaries)
        size_uniformity = self._calculate_size_uniformity(chunks)
        cross_boundary_rate = self._calculate_cross_boundary_rate(chunks, boundaries)

        return ChunkQuality(
            boundary_integrity=boundary_integrity,
            size_uniformity=size_uniformity,
            cross_boundary_rate=cross_boundary_rate
        )

    def _calculate_boundary_integrity(
        self,
        chunks: List[Chunk],
        boundaries: List[Boundary]
    ) -> float:
        """
        计算边界完整性

        定义：在强边界（对话/段落/代码块）处切分的比例

        Args:
            chunks: 分块列表
            boundaries: 边界列表

        Returns:
            float: 边界完整性 (0.0-1.0)
        """
        if not chunks or len(chunks) == 1:
            return 1.0  # 只有一个 chunk，无需切分

        # 统计在强边界处切分的次数
        strong_boundary_types = {
            BoundaryType.DIALOGUE,
            BoundaryType.PARAGRAPH,
            BoundaryType.CODE_BLOCK
        }

        strong_cuts = 0
        total_cuts = len(chunks) - 1  # 切分次数 = chunk 数量 - 1

        for chunk in chunks[:-1]:  # 排除最后一个 chunk（无后续切分）
            if chunk.boundary and chunk.boundary.type in strong_boundary_types:
                strong_cuts += 1

        return strong_cuts / total_cuts if total_cuts > 0 else 1.0

    def _calculate_size_uniformity(self, chunks: List[Chunk]) -> float:
        """
        计算 size 均匀性

        定义：1.0 - size 的变异系数 (CV)
        CV = std / mean

        Args:
            chunks: 分块列表

        Returns:
            float: size 均匀性 (0.0-1.0)，越接近 1.0 越均匀
        """
        if not chunks:
            return 1.0

        sizes = [chunk.size for chunk in chunks]

        # 计算均值和标准差
        mean = sum(sizes) / len(sizes)

        if mean == 0:
            return 1.0

        variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        std = math.sqrt(variance)

        cv = std / mean  # 变异系数

        # 转换为均匀性得分（CV 越小，均匀性越高）
        uniformity = max(0.0, 1.0 - cv)

        return uniformity

    def _calculate_cross_boundary_rate(
        self,
        chunks: List[Chunk],
        boundaries: List[Boundary]
    ) -> float:
        """
        计算跨边界率

        定义：在边界附近（±50 tokens）但不在边界处切分的比例

        Args:
            chunks: 分块列表
            boundaries: 边界列表

        Returns:
            float: 跨边界率 (0.0-1.0)，越低越好
        """
        if not chunks or len(chunks) == 1:
            return 0.0  # 只有一个 chunk，无跨边界

        # 构建边界位置集合
        boundary_offsets = {b.token_offset for b in boundaries}

        cross_boundary_count = 0
        total_cuts = len(chunks) - 1

        for chunk in chunks[:-1]:
            cut_position = chunk.end

            # 检查是否在边界处切分
            if cut_position in boundary_offsets:
                continue  # 在边界处，不算跨边界

            # 检查是否在边界附近（±50 tokens）
            near_boundary = any(
                abs(cut_position - b_offset) <= 50
                for b_offset in boundary_offsets
            )

            if near_boundary:
                cross_boundary_count += 1

        return cross_boundary_count / total_cuts if total_cuts > 0 else 0.0

    def is_high_quality(self, quality: ChunkQuality) -> bool:
        """
        判断是否高质量

        Args:
            quality: 质量指标

        Returns:
            bool: 是否高质量
        """
        return (
            quality.boundary_integrity >= self.min_boundary_integrity and
            quality.size_uniformity >= (1.0 - self.max_size_cv) and
            quality.cross_boundary_rate <= self.max_cross_boundary_rate
        )
