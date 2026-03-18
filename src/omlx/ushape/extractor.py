# SPDX-License-Identifier: Apache-2.0
"""
Extractive summarizer for U-Shape Reinforcement.

Selects the most query-relevant sentences from scored chunks
using BM25 re-scoring at the sentence level.
"""

import logging
import re

from .bm25_scorer import BM25Scorer
from .types import ScoredChunk

logger = logging.getLogger(__name__)

# Sentence splitting regex: splits on Chinese or English sentence-ending
# punctuation followed by optional whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\n])\s*")


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling both Chinese and English punctuation.

    Args:
        text: Input text to split.

    Returns:
        List of non-empty sentence strings.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


class ExtractSummarizer:
    """
    Extracts the most relevant sentences from scored chunks
    to form a concise extractive summary.
    """

    def __init__(self) -> None:
        self._scorer = BM25Scorer()

    def summarize(
        self,
        query: str,
        chunks: list[ScoredChunk],
        max_sentences: int = 5,
    ) -> str:
        """
        Generate an extractive summary from scored chunks.

        Splits chunks into sentences, re-scores each sentence against
        the query via BM25, and selects the top-N most relevant sentences.

        Args:
            query: The original user query.
            chunks: Pre-selected top-K scored chunks.
            max_sentences: Maximum number of sentences in the summary.

        Returns:
            Summary text formatted as bullet points, or empty string
            if no summary can be generated.
        """
        if not chunks:
            return ""

        try:
            # Collect all sentences from all chunks
            all_sentences: list[str] = []
            for chunk in chunks:
                all_sentences.extend(_split_sentences(chunk.text))

            if not all_sentences:
                return ""

            # Single sentence: return it directly
            if len(all_sentences) == 1:
                return f"- {all_sentences[0]}\n"

            # Re-score sentences against the query
            sentence_scores = self._scorer.score(query, all_sentences)

            # Sort by score descending, take top-N
            scored_pairs = sorted(
                zip(all_sentences, sentence_scores),
                key=lambda pair: pair[1],
                reverse=True,
            )
            selected = [sent for sent, _ in scored_pairs[:max_sentences]]

            if not selected:
                return ""

            return "\n".join(f"- {s}" for s in selected) + "\n"

        except Exception as e:
            logger.warning("ExtractSummarizer.summarize failed: %s", e)
            return ""
