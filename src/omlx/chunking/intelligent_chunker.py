"""
智能分块系统 - 主编排器

集成内容检测、边界提取、动态分块、质量验证
提供统一的 API: intelligent_chunked_prefill()
"""

from typing import List, Tuple, Optional
import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.cache import KVCache

from .types import Chunk, ChunkQuality, ContentType, ChunkStats
from .content_detector import ContentDetector
from .boundary_extractor import BoundaryExtractor
from .dynamic_chunker import DynamicChunker
from .quality_validator import QualityValidator


class IntelligentChunker:
    """智能分块编排器"""

    def __init__(
        self,
        target_size: int = 4096,
        flexibility: float = 0.125,
        min_quality_score: float = 0.8
    ):
        """
        初始化智能分块器

        Args:
            target_size: 目标 chunk size
            flexibility: 弹性范围（相对于 target_size 的比例）
            min_quality_score: 最低质量分数（低于此值自动回退到固定分块）
        """
        self.target_size = target_size
        self.flexibility = flexibility
        self.min_quality_score = min_quality_score

        # 初始化各模块
        self.content_detector = ContentDetector()
        self.boundary_extractor = BoundaryExtractor()
        self.dynamic_chunker = DynamicChunker(
            target_size=target_size,
            flexibility=flexibility
        )
        self.quality_validator = QualityValidator()

    def chunk(
        self,
        text: str,
        tokens: List[int]
    ) -> Tuple[List[Chunk], ChunkQuality, ContentType]:
        """
        执行智能分块

        Args:
            text: 输入文本
            tokens: Token 列表

        Returns:
            (chunks, quality, content_type):
                chunks: 分块列表
                quality: 质量指标
                content_type: 检测到的内容类型
        """
        # 1. 检测内容类型
        content_type = self.content_detector.detect(text)

        # 2. 提取语义边界
        boundaries = self.boundary_extractor.extract(text, tokens, content_type)

        # 3. 动态分块
        chunks = self.dynamic_chunker.chunk(tokens, boundaries, content_type)

        # 4. 质量验证
        quality = self.quality_validator.validate(chunks, boundaries, len(tokens))

        # 5. 如果质量不达标，回退到固定分块
        if quality.overall_score < self.min_quality_score:
            print(f"⚠️ 质量不达标 (score={quality.overall_score:.2f}), 回退到固定分块")
            chunks = self.dynamic_chunker.chunk_fixed(tokens, self.target_size)
            quality = self.quality_validator.validate(chunks, [], len(tokens))

        return chunks, quality, content_type

    def intelligent_chunked_prefill(
        self,
        model,
        tokenizer: TokenizerWrapper,
        prompt: str,
        max_tokens: int = 100,
        verbose: bool = True
    ) -> Tuple[str, ChunkStats, ChunkQuality]:
        """
        执行智能分块预填充

        Args:
            model: MLX 模型
            tokenizer: Tokenizer
            prompt: 输入提示
            max_tokens: 最大生成 tokens
            verbose: 是否打印详细信息

        Returns:
            (output_text, stats, quality):
                output_text: 生成的文本
                stats: 分块统计
                quality: 质量指标
        """
        # 1. Tokenize
        tokens = tokenizer.encode(prompt)

        if verbose:
            print(f"\n{'='*60}")
            print(f"智能分块预填充")
            print(f"{'='*60}")
            print(f"输入 tokens: {len(tokens)}")

        # 2. 智能分块
        chunks, quality, content_type = self.chunk(prompt, tokens)

        if verbose:
            print(f"内容类型: {content_type.value}")
            print(f"分块数量: {len(chunks)}")
            print(f"质量指标: {quality}")
            print(f"{'='*60}\n")

        # 3. 执行分块预填充
        cache = [KVCache() for _ in range(len(model.model.layers))]
        stats = ChunkStats()

        import time

        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()

            # 转换为 MLX array
            chunk_mx = mx.array([chunk.tokens])

            # 模型推理（累积 cache）
            logits = model(chunk_mx, cache=cache)

            # 确保计算完成
            mx.eval(logits)
            mx.eval([c.keys for c in cache])

            chunk_time = time.time() - chunk_start_time

            # 记录统计
            boundary_type = chunk.boundary.type if chunk.boundary else None
            stats.add_chunk(chunk.size, chunk_time, boundary_type)

            if verbose:
                print(f"Chunk {i+1}/{len(chunks)}: "
                      f"{chunk.size} tokens, "
                      f"{chunk_time:.2f}s, "
                      f"{chunk.size/chunk_time:.1f} tok/s"
                      f"{f', boundary={boundary_type.value}' if boundary_type else ''}")

        # 4. 生成阶段
        if verbose:
            print(f"\n生成阶段:")

        generated_tokens = []
        token_start_time = time.time()

        for _ in range(max_tokens):
            # 使用最后一个 token 生成
            next_token_mx = mx.array([[tokens[-1] if not generated_tokens else generated_tokens[-1]]])

            logits = model(next_token_mx, cache=cache)
            mx.eval(logits)
            mx.eval([c.keys for c in cache])

            # 采样下一个 token
            next_token = mx.argmax(logits[0, -1, :], keepdims=True).item()
            generated_tokens.append(next_token)

            # 如果是 EOS，停止
            if next_token == tokenizer.eos_token_id:
                break

        token_time = time.time() - token_start_time

        if verbose:
            print(f"生成 {len(generated_tokens)} tokens in {token_time:.2f}s "
                  f"({len(generated_tokens)/token_time:.1f} tok/s)")
            print(f"\n总耗时: {stats.total_time + token_time:.2f}s")
            print(f"{'='*60}\n")

        # 5. 解码输出
        output_text = tokenizer.decode(generated_tokens)

        return output_text, stats, quality


def intelligent_chunked_prefill(
    model,
    tokenizer: TokenizerWrapper,
    prompt: str,
    target_size: int = 4096,
    flexibility: float = 0.125,
    max_tokens: int = 100,
    verbose: bool = True
) -> Tuple[str, ChunkStats, ChunkQuality]:
    """
    智能分块预填充（便捷函数）

    Args:
        model: MLX 模型
        tokenizer: Tokenizer
        prompt: 输入提示
        target_size: 目标 chunk size（默认 4096）
        flexibility: 弹性范围（默认 ±12.5%）
        max_tokens: 最大生成 tokens（默认 100）
        verbose: 是否打印详细信息（默认 True）

    Returns:
        (output_text, stats, quality):
            output_text: 生成的文本
            stats: 分块统计
            quality: 质量指标
    """
    chunker = IntelligentChunker(
        target_size=target_size,
        flexibility=flexibility
    )

    return chunker.intelligent_chunked_prefill(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=verbose
    )
