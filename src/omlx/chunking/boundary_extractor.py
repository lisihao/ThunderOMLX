"""
智能分块系统 - 语义边界提取

提取文本中的语义边界（段落/句子/对话/代码块）
"""

import re
from typing import List, Tuple
from .types import Boundary, BoundaryType, ContentType


class BoundaryExtractor:
    """语义边界提取器"""

    def __init__(self):
        # 对话边界模式
        self.dialogue_patterns = [
            (r"User:\s*", BoundaryType.DIALOGUE, 1.0),
            (r"Assistant:\s*", BoundaryType.DIALOGUE, 1.0),
            (r"Human:\s*", BoundaryType.DIALOGUE, 1.0),
            (r"AI:\s*", BoundaryType.DIALOGUE, 1.0),
            (r"<\|im_start\|>user", BoundaryType.DIALOGUE, 1.0),
            (r"<\|im_start\|>assistant", BoundaryType.DIALOGUE, 1.0),
        ]

        # 代码块边界模式
        self.code_patterns = [
            (r"```[\w]*\n", BoundaryType.CODE_BLOCK, 1.0),  # Markdown code block start
            (r"\n```", BoundaryType.CODE_BLOCK, 1.0),        # Markdown code block end
        ]

        # 段落边界
        self.paragraph_pattern = (r"\n\n", BoundaryType.PARAGRAPH, 1.0)

        # 句子边界
        self.sentence_pattern = (r"[.!?]\s+", BoundaryType.SENTENCE, 0.5)

    def extract(
        self,
        text: str,
        tokens: List[int],
        content_type: ContentType
    ) -> List[Boundary]:
        """
        提取语义边界

        Args:
            text: 输入文本
            tokens: Token 列表
            content_type: 内容类型

        Returns:
            List[Boundary]: 边界列表（按 token_offset 排序）
        """
        boundaries = []

        # 根据内容类型选择边界提取策略
        if content_type == ContentType.DIALOGUE:
            boundaries.extend(self._extract_dialogue_boundaries(text, tokens))
            boundaries.extend(self._extract_paragraph_boundaries(text, tokens))
        elif content_type == ContentType.CODE:
            boundaries.extend(self._extract_code_boundaries(text, tokens))
            boundaries.extend(self._extract_paragraph_boundaries(text, tokens))
        elif content_type == ContentType.DOCUMENT:
            boundaries.extend(self._extract_paragraph_boundaries(text, tokens))
            boundaries.extend(self._extract_sentence_boundaries(text, tokens))
        elif content_type == ContentType.MIXED:
            # 混合模式：提取所有类型的边界
            boundaries.extend(self._extract_dialogue_boundaries(text, tokens))
            boundaries.extend(self._extract_code_boundaries(text, tokens))
            boundaries.extend(self._extract_paragraph_boundaries(text, tokens))
            boundaries.extend(self._extract_sentence_boundaries(text, tokens))
        else:  # GENERIC
            boundaries.extend(self._extract_paragraph_boundaries(text, tokens))
            boundaries.extend(self._extract_sentence_boundaries(text, tokens))

        # 排序 + 去重
        boundaries.sort(key=lambda b: b.token_offset)
        boundaries = self._deduplicate_boundaries(boundaries)

        return boundaries

    def _extract_dialogue_boundaries(self, text: str, tokens: List[int]) -> List[Boundary]:
        """提取对话边界"""
        boundaries = []

        for pattern, boundary_type, strength in self.dialogue_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                text_offset = match.end()
                token_offset = self._text_to_token_offset(text_offset, text, tokens)

                if token_offset > 0:  # 忽略起始位置
                    boundaries.append(Boundary(
                        text_offset=text_offset,
                        token_offset=token_offset,
                        type=boundary_type,
                        strength=strength
                    ))

        return boundaries

    def _extract_code_boundaries(self, text: str, tokens: List[int]) -> List[Boundary]:
        """提取代码块边界"""
        boundaries = []

        for pattern, boundary_type, strength in self.code_patterns:
            for match in re.finditer(pattern, text):
                text_offset = match.end()
                token_offset = self._text_to_token_offset(text_offset, text, tokens)

                if token_offset > 0:
                    boundaries.append(Boundary(
                        text_offset=text_offset,
                        token_offset=token_offset,
                        type=boundary_type,
                        strength=strength
                    ))

        return boundaries

    def _extract_paragraph_boundaries(self, text: str, tokens: List[int]) -> List[Boundary]:
        """提取段落边界"""
        boundaries = []

        pattern, boundary_type, strength = self.paragraph_pattern

        for match in re.finditer(pattern, text):
            text_offset = match.end()
            token_offset = self._text_to_token_offset(text_offset, text, tokens)

            if token_offset > 0:
                boundaries.append(Boundary(
                    text_offset=text_offset,
                    token_offset=token_offset,
                    type=boundary_type,
                    strength=strength
                ))

        return boundaries

    def _extract_sentence_boundaries(self, text: str, tokens: List[int]) -> List[Boundary]:
        """提取句子边界"""
        boundaries = []

        pattern, boundary_type, strength = self.sentence_pattern

        for match in re.finditer(pattern, text):
            text_offset = match.end()
            token_offset = self._text_to_token_offset(text_offset, text, tokens)

            if token_offset > 0:
                boundaries.append(Boundary(
                    text_offset=text_offset,
                    token_offset=token_offset,
                    type=boundary_type,
                    strength=strength
                ))

        return boundaries

    def _text_to_token_offset(self, text_offset: int, text: str, tokens: List[int]) -> int:
        """
        将文本 offset 转换为 token offset

        简化版本：使用近似估计
        假设平均 1 token ≈ 4 characters

        Args:
            text_offset: 文本偏移量
            text: 原始文本
            tokens: Token 列表

        Returns:
            int: Token 偏移量（近似）
        """
        # 简化版本：字符数 / 4
        # TODO: 更精确的实现需要 tokenizer 的 offset_mapping
        chars_per_token = len(text) / max(len(tokens), 1)
        token_offset = int(text_offset / chars_per_token)

        # 确保在有效范围内
        return min(token_offset, len(tokens))

    def _deduplicate_boundaries(self, boundaries: List[Boundary]) -> List[Boundary]:
        """
        去重边界（同一位置只保留强度最高的）

        Args:
            boundaries: 边界列表

        Returns:
            List[Boundary]: 去重后的边界列表
        """
        if not boundaries:
            return []

        # 按 token_offset 分组
        grouped = {}
        for b in boundaries:
            if b.token_offset not in grouped:
                grouped[b.token_offset] = b
            else:
                # 保留强度更高的
                if b.strength > grouped[b.token_offset].strength:
                    grouped[b.token_offset] = b

        # 转回列表并排序
        result = list(grouped.values())
        result.sort(key=lambda b: b.token_offset)

        return result
