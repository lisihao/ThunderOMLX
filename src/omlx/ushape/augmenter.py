# SPDX-License-Identifier: Apache-2.0
"""
U-Shape Reinforcement augmenter for ThunderOMLX.

Orchestrates BM25 scoring and extractive summarization to inject
query-relevant context summaries into the prompt tail, mitigating
the "Lost in the Middle" attention degradation in LLMs.

Does NOT reorder original documents (preserves prefix cache).
"""

import copy
import logging
from typing import Any

from .bm25_scorer import BM25Scorer
from .extractor import ExtractSummarizer
from .types import ScoredChunk, UShapeConfig

logger = logging.getLogger(__name__)

_SUMMARY_HEADER = (
    "\n\n---\n"
    "[Context Summary]\n"
    "\u4ee5\u4e0b\u662f\u4e0e\u4f60\u7684\u95ee\u9898\u6700\u76f8\u5173\u7684\u4e0a\u4e0b\u6587\u6458\u8981:\n"
)
_SUMMARY_FOOTER = "---\n"


class UShapeAugmenter:
    """
    Augments LLM prompt messages with query-relevant context summaries.

    The summary is appended to the last user message (high attention zone)
    without reordering existing messages (protecting prefix cache).
    """

    def __init__(self, config: UShapeConfig) -> None:
        """
        Initialize the augmenter.

        Args:
            config: U-Shape configuration controlling augmentation behavior.
        """
        self._config = config
        self._scorer = BM25Scorer()
        self._summarizer = ExtractSummarizer()

    def should_augment(
        self, messages: list[dict[str, Any]], token_count: int
    ) -> bool:
        """
        Check whether augmentation should be applied.

        Args:
            messages: Chat messages (used for future heuristics).
            token_count: Current prompt token count.

        Returns:
            True if augmentation is enabled and token count exceeds threshold.
        """
        if not self._config.enabled:
            return False
        if token_count < self._config.min_prompt_tokens:
            return False
        # Need at least one user message to extract query from
        return any(m.get("role") == "user" for m in messages)

    def augment_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Augment messages with a query-relevant context summary.

        Extracts the query from the last user message, collects context
        from earlier messages, scores context chunks via BM25, generates
        an extractive summary, and appends it to the last user message.

        Args:
            messages: Original chat messages. NOT modified.

        Returns:
            New message list with summary injected, or the original
            messages unchanged if augmentation fails or is disabled.
        """
        if not self._config.enabled:
            return messages

        try:
            query = _extract_query(messages)
            if not query:
                logger.debug("U-Shape: no user query found, skipping.")
                return messages

            context = _extract_context(messages)
            if not context:
                logger.debug("U-Shape: no context to augment, skipping.")
                return messages

            chunks = _chunk_text(context, self._config.chunk_size)
            if not chunks:
                logger.debug("U-Shape: no chunks generated, skipping.")
                return messages

            # Score chunks
            chunk_scores = self._scorer.score(query, chunks)
            scored_chunks = [
                ScoredChunk(text=text, score=score, source_index=idx)
                for idx, (text, score) in enumerate(zip(chunks, chunk_scores))
            ]

            # Select top-K by score
            top_k = sorted(
                scored_chunks, key=lambda c: c.score, reverse=True
            )[: self._config.top_k_chunks]

            if not top_k or all(c.score == 0.0 for c in top_k):
                logger.debug("U-Shape: no relevant chunks found, skipping.")
                return messages

            # Generate extractive summary
            summary = self._summarizer.summarize(
                query=query,
                chunks=top_k,
                max_sentences=self._config.summary_sentences,
            )
            if not summary:
                logger.debug("U-Shape: summary is empty, skipping.")
                return messages

            # Inject summary into a deep copy (immutability)
            augmented = copy.deepcopy(messages)
            _inject_summary(augmented, summary)

            logger.info(
                "U-Shape augmentation applied: %d chunks scored, "
                "top-%d selected, %d summary sentences.",
                len(chunks),
                len(top_k),
                summary.count("\n"),
            )
            return augmented

        except Exception as e:
            logger.error("U-Shape augmentation failed: %s", e, exc_info=True)
            return messages


def _extract_query(messages: list[dict[str, Any]]) -> str:
    """Extract the content of the last user message as the query."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _extract_context(messages: list[dict[str, Any]]) -> str:
    """
    Collect context from system prompt and all messages before the last
    user message. The last user message itself is excluded (it's the query).
    """
    # Find index of last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx <= 0:
        # No preceding context available
        return ""

    parts: list[str] = []
    for msg in messages[:last_user_idx]:
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content)

    return "\n\n".join(parts)


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into fixed-size character chunks."""
    if not text or chunk_size <= 0:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _inject_summary(messages: list[dict[str, Any]], summary: str) -> None:
    """
    Append the summary to the last user message content.

    Modifies the messages list in-place (caller ensures it's a deep copy).
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            original = msg.get("content", "")
            msg["content"] = (
                f"{original}{_SUMMARY_HEADER}{summary}{_SUMMARY_FOOTER}"
            )
            return
