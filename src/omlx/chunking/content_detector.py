"""
智能分块系统 - 内容类型检测

检测文本的内容类型：对话/文档/代码/混合/通用
"""

import re
from typing import Dict
from .types import ContentType


class ContentDetector:
    """内容类型检测器"""

    def __init__(self):
        # 对话模式的正则表达式
        self.dialogue_patterns = [
            r"User:\s*",
            r"Assistant:\s*",
            r"Human:\s*",
            r"AI:\s*",
            r"<\|im_start\|>user",
            r"<\|im_start\|>assistant",
            r"###\s*Human:",
            r"###\s*Assistant:",
        ]

        # 代码模式的正则表达式
        self.code_patterns = [
            r"```",  # Markdown code block
            r"def\s+\w+\s*\(",  # Python function
            r"function\s+\w+\s*\(",  # JavaScript function
            r"class\s+\w+",  # Class definition
            r"import\s+\w+",  # Import statement
            r"#include\s*<",  # C/C++ include
            r"fn\s+\w+\s*\(",  # Rust function
            r"func\s+\w+\s*\(",  # Go function
        ]

    def detect(self, text: str) -> ContentType:
        """
        检测文本的内容类型

        Args:
            text: 输入文本

        Returns:
            ContentType: 检测到的内容类型
        """
        # 计算各类型的得分
        scores = {
            "dialogue": self._is_dialogue(text),
            "document": self._is_document(text),
            "code": self._is_code(text),
        }

        # 统计为 True 的数量
        true_count = sum(scores.values())

        # 多类型混合
        if true_count >= 2:
            return ContentType.MIXED

        # 单一类型
        if scores["dialogue"]:
            return ContentType.DIALOGUE
        elif scores["code"]:
            return ContentType.CODE
        elif scores["document"]:
            return ContentType.DOCUMENT
        else:
            return ContentType.GENERIC

    def _is_dialogue(self, text: str) -> bool:
        """检测是否为对话格式"""
        matches = sum(
            bool(re.search(pattern, text, re.IGNORECASE))
            for pattern in self.dialogue_patterns
        )

        # 至少出现 2 次对话标记 → 对话模式
        return matches >= 2

    def _is_document(self, text: str) -> bool:
        """检测是否为文档格式"""
        # 连续段落模式
        paragraphs = text.split("\n\n")

        # 段落数量 > 3，且平均长度 > 100 字符
        if len(paragraphs) >= 3:
            avg_len = sum(len(p) for p in paragraphs if p.strip()) / max(len(paragraphs), 1)
            return avg_len > 100

        return False

    def _is_code(self, text: str) -> bool:
        """检测是否为代码格式"""
        matches = sum(
            bool(re.search(pattern, text))
            for pattern in self.code_patterns
        )

        # 至少匹配 2 个代码特征 → 代码模式
        return matches >= 2

    def get_detailed_scores(self, text: str) -> Dict[str, bool]:
        """
        获取详细的类型得分

        Args:
            text: 输入文本

        Returns:
            Dict[str, bool]: 各类型的得分
        """
        return {
            "dialogue": self._is_dialogue(text),
            "document": self._is_document(text),
            "code": self._is_code(text),
        }
