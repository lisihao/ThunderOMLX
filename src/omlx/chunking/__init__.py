"""
智能分块系统

核心组件：
- ContentDetector: 内容类型检测（对话/文档/代码/混合）
- BoundaryExtractor: 语义边界提取（段落/句子/对话/代码块）
- DynamicChunker: 动态分块算法（Greedy Boundary-Aware Packing）
- QualityValidator: 质量验证（boundary_integrity, size_uniformity, cross_boundary_rate）
- IntelligentChunker: 主编排器（集成所有模块 + 自动回退）

主要 API：
    intelligent_chunked_prefill(model, tokenizer, prompt, ...)
"""

from .types import (
    ContentType,
    BoundaryType,
    Boundary,
    Chunk,
    ChunkStats,
    ChunkQuality
)

from .content_detector import ContentDetector
from .boundary_extractor import BoundaryExtractor
from .dynamic_chunker import DynamicChunker
from .quality_validator import QualityValidator
from .intelligent_chunker import (
    IntelligentChunker,
    intelligent_chunked_prefill
)

__all__ = [
    # Types
    "ContentType",
    "BoundaryType",
    "Boundary",
    "Chunk",
    "ChunkStats",
    "ChunkQuality",
    # Modules
    "ContentDetector",
    "BoundaryExtractor",
    "DynamicChunker",
    "QualityValidator",
    "IntelligentChunker",
    # Main API
    "intelligent_chunked_prefill",
]
