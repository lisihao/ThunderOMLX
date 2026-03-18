# SPDX-License-Identifier: Apache-2.0
"""
BM25 scorer for U-Shape Reinforcement.

Provides Chinese+English mixed tokenization and BM25-based chunk scoring
using the bm25s library with custom tokenization to support CJK characters.
"""

import logging
import re

import bm25s
from bm25s.tokenization import Tokenized

logger = logging.getLogger(__name__)

# Pre-compiled regex for mixed Chinese/English tokenization.
# Chinese characters are treated as individual tokens;
# English words and numbers are treated as contiguous tokens.
_MIXED_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


def mixed_tokenize(texts: list[str]) -> list[list[str]]:
    """
    Tokenize texts with mixed Chinese and English content.

    Chinese characters become single-character tokens.
    English words and numbers become word-level tokens.
    All output is lowercased for consistent matching.

    Args:
        texts: List of strings to tokenize.

    Returns:
        List of token lists, one per input text.
    """
    return [_MIXED_TOKEN_RE.findall(text.lower()) for text in texts]


def _build_tokenized(
    token_lists: list[list[str]],
    base_vocab: dict[str, int] | None = None,
) -> tuple[Tokenized, dict[str, int]]:
    """
    Build a bm25s Tokenized object from token lists.

    Args:
        token_lists: List of token lists to encode.
        base_vocab: Optional existing vocabulary to extend.

    Returns:
        Tuple of (Tokenized object, vocabulary dict).
    """
    vocab = dict(base_vocab) if base_vocab else {}
    all_ids: list[list[int]] = []

    for tokens in token_lists:
        ids: list[int] = []
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
            ids.append(vocab[token])
        all_ids.append(ids)

    return Tokenized(ids=all_ids, vocab=vocab), vocab


class BM25Scorer:
    """
    Scores text chunks against a query using BM25.

    Uses bm25s with custom mixed tokenization to support
    Chinese and English content.
    """

    def score(self, query: str, chunks: list[str]) -> list[float]:
        """
        Calculate BM25 relevance scores for each chunk against the query.

        Args:
            query: The query string.
            chunks: List of text chunks to score.

        Returns:
            List of float scores, one per chunk. Returns all zeros
            if fewer than 2 chunks (BM25 needs at least 2 documents)
            or on any error.
        """
        if not chunks or len(chunks) < 2:
            return [0.0] * len(chunks)

        try:
            # Tokenize and index corpus
            corpus_tokens = mixed_tokenize(chunks)
            corpus_tokenized, corpus_vocab = _build_tokenized(corpus_tokens)

            model = bm25s.BM25()
            model.index(corpus_tokenized, show_progress=False)

            # Tokenize query, extending corpus vocab for unseen tokens
            query_tokens = mixed_tokenize([query])
            query_tokenized, _ = _build_tokenized(
                query_tokens, base_vocab=corpus_vocab
            )

            # Retrieve all chunks scored against query
            results, scores = model.retrieve(
                query_tokenized, k=len(chunks), show_progress=False
            )

            # Map doc_id -> score (results are sorted by score, not by index)
            score_map: dict[int, float] = {}
            for doc_id, doc_score in zip(results[0], scores[0]):
                score_map[int(doc_id)] = float(doc_score)

            # Return scores in original chunk order
            return [score_map.get(i, 0.0) for i in range(len(chunks))]

        except Exception as e:
            logger.warning("BM25Scorer.score failed: %s", e)
            return [0.0] * len(chunks)
