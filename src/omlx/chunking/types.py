"""
智能分块系统 - 数据类型定义
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class ContentType(Enum):
    """内容类型"""
    DIALOGUE = "dialogue"  # 对话格式
    DOCUMENT = "document"  # 文档格式
    CODE = "code"          # 代码格式
    MIXED = "mixed"        # 混合格式
    GENERIC = "generic"    # 通用格式（默认）


class BoundaryType(Enum):
    """边界类型"""
    DIALOGUE = "dialogue"      # 对话边界 (User:/Assistant:)
    PARAGRAPH = "paragraph"    # 段落边界 (\n\n)
    CODE_BLOCK = "code_block"  # 代码块边界
    SENTENCE = "sentence"      # 句子边界 (. ! ?)
    COMMA = "comma"            # 逗号边界


@dataclass
class Boundary:
    """语义边界"""
    token_offset: int       # Token 偏移量
    text_offset: int        # 文本偏移量
    type: BoundaryType      # 边界类型
    strength: float         # 边界强度 (0.0-1.0)

    def __repr__(self) -> str:
        return f"Boundary(token={self.token_offset}, type={self.type.value}, strength={self.strength:.2f})"


@dataclass
class Chunk:
    """分块"""
    tokens: List[int]                 # Token 列表
    start: int                        # 起始 token offset
    end: int                          # 结束 token offset
    boundary: Optional[Boundary]      # 切分边界（如果有）

    @property
    def size(self) -> int:
        """Chunk 大小（token 数量）"""
        return len(self.tokens)

    def __repr__(self) -> str:
        boundary_str = f", boundary={self.boundary.type.value}" if self.boundary else ""
        return f"Chunk(start={self.start}, end={self.end}, size={self.size}{boundary_str})"


@dataclass
class ChunkStats:
    """分块统计"""
    total_chunks: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    chunk_sizes: List[int] = None
    chunk_times: List[float] = None
    boundary_types: List[Optional[BoundaryType]] = None

    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = []
        if self.chunk_times is None:
            self.chunk_times = []
        if self.boundary_types is None:
            self.boundary_types = []

    def add_chunk(self, size: int, time: float, boundary_type: Optional[BoundaryType] = None):
        """添加一个 chunk 的统计"""
        self.total_chunks += 1
        self.total_tokens += size
        self.total_time += time

        self.chunk_sizes.append(size)
        self.chunk_times.append(time)
        self.boundary_types.append(boundary_type)

    @property
    def avg_chunk_size(self) -> float:
        """平均 chunk 大小"""
        return self.total_tokens / self.total_chunks if self.total_chunks > 0 else 0.0

    @property
    def avg_chunk_time(self) -> float:
        """平均 chunk 处理时间"""
        return self.total_time / self.total_chunks if self.total_chunks > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        """吞吐量（tokens/s）"""
        return self.total_tokens / self.total_time if self.total_time > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"ChunkStats(chunks={self.total_chunks}, "
            f"tokens={self.total_tokens}, "
            f"avg_size={self.avg_chunk_size:.0f}, "
            f"speed={self.tokens_per_second:.0f} tok/s)"
        )


@dataclass
class ChunkQuality:
    """分块质量指标"""
    boundary_integrity: float    # 边界完整性 (0.0-1.0)
    size_uniformity: float       # Size 均匀性 (0.0-1.0)
    cross_boundary_rate: float   # 跨边界率 (0.0-1.0, 越低越好)

    @property
    def overall_score(self) -> float:
        """综合质量分数"""
        return (
            self.boundary_integrity * 0.5 +
            self.size_uniformity * 0.3 +
            (1.0 - self.cross_boundary_rate) * 0.2
        )

    @property
    def is_high_quality(self) -> bool:
        """是否高质量（>= 0.8）"""
        return self.overall_score >= 0.8

    def __repr__(self) -> str:
        return (
            f"ChunkQuality(score={self.overall_score:.2f}, "
            f"integrity={self.boundary_integrity:.2f}, "
            f"uniformity={self.size_uniformity:.2f}, "
            f"cross_rate={self.cross_boundary_rate:.2f})"
        )
