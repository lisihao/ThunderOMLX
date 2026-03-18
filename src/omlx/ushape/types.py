# SPDX-License-Identifier: Apache-2.0
"""
Data types for the U-Shape Reinforcement module.

Defines configuration and scored chunk structures used across
the BM25 scorer, extractive summarizer, and message augmenter.
"""

from dataclasses import dataclass


@dataclass
class UShapeConfig:
    """
    Configuration for the U-Shape Reinforcement module.

    Controls when and how prompt augmentation is applied to mitigate
    the "Lost in the Middle" attention degradation in LLMs.
    """

    enabled: bool = False  # default off, opt-in
    min_prompt_tokens: int = 4096  # below this, don't trigger augmentation
    top_k_chunks: int = 3  # top-K relevant chunks to select
    summary_max_tokens: int = 512  # max tokens for generated summary
    summary_sentences: int = 5  # max sentences to extract for summary
    chunk_size: int = 512  # context chunk size (characters)


@dataclass
class ScoredChunk:
    """
    A text chunk with its BM25 relevance score and original position.

    Attributes:
        text: The chunk text content.
        score: BM25 relevance score against the query.
        source_index: Original position index in the context sequence.
    """

    text: str
    score: float
    source_index: int
