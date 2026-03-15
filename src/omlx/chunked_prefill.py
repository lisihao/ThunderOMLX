# SPDX-License-Identifier: Apache-2.0
"""
Chunked Prefill for long prompts in oMLX.

This module implements a simple fixed-size chunking strategy to reduce
memory peak and first-token latency for long prompts in MLX inference.

Key benefits:
- Reduces memory peak by processing tokens in fixed-size chunks
- Enables early output (lower first-token latency for long prompts)
- Maintains backward compatibility with traditional prefill

The chunking strategy is simple and MVP-focused:
- Split input tokens into fixed-size chunks
- Process each chunk sequentially
- Merge KV caches between chunks using mlx_lm's _merge_caches
"""

import logging
import os
from typing import Any, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


class ChunkedPrefillConfig:
    """Configuration for chunked prefill strategy."""

    def __init__(
        self,
        chunk_size: int = 512,
        enable_chunking: bool = False,
        min_tokens_for_chunking: int = 2560,
    ):
        """
        Initialize chunked prefill configuration.

        Args:
            chunk_size: Number of tokens per chunk (default: 512).
                       Smaller chunks = lower memory peak but more forward passes.
                       Larger chunks = fewer forward passes but higher memory peak.
            enable_chunking: Whether to enable chunked prefill (default: False).
            min_tokens_for_chunking: Minimum prompt length to trigger chunking
                                    (default: 2560 tokens).
        """
        self.chunk_size = max(1, int(chunk_size))
        self.enable_chunking = bool(enable_chunking)
        self.min_tokens_for_chunking = max(1, int(min_tokens_for_chunking))

        logger.info(
            f"ChunkedPrefill configured: "
            f"enabled={self.enable_chunking}, "
            f"chunk_size={self.chunk_size}, "
            f"min_tokens={self.min_tokens_for_chunking}"
        )

    @staticmethod
    def from_env() -> "ChunkedPrefillConfig":
        """Create configuration from environment variables.

        Environment variables:
            OMLX_ENABLE_CHUNKED_PREFILL: Enable chunked prefill (default: false)
            OMLX_CHUNK_SIZE: Chunk size in tokens (default: 512)
            OMLX_MIN_TOKENS_FOR_CHUNKING: Min tokens to trigger chunking (default: 2560)
        """
        enable = os.getenv("OMLX_ENABLE_CHUNKED_PREFILL", "false").lower() == "true"
        chunk_size = int(os.getenv("OMLX_CHUNK_SIZE", "512"))
        min_tokens = int(os.getenv("OMLX_MIN_TOKENS_FOR_CHUNKING", "2560"))

        return ChunkedPrefillConfig(
            chunk_size=chunk_size,
            enable_chunking=enable,
            min_tokens_for_chunking=min_tokens,
        )


class ChunkedPrefillEngine:
    """Engine for chunked prefill processing.

    This engine wraps a standard prefill function and adds chunking logic.
    It maintains backward compatibility - if chunking is disabled or not needed,
    it delegates to the original prefill function.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[ChunkedPrefillConfig] = None,
    ):
        """
        Initialize the chunked prefill engine.

        Args:
            model: The MLX model (not directly used, but stored for context).
            config: ChunkedPrefillConfig instance. If None, loads from env.
        """
        self.model = model
        self.config = config or ChunkedPrefillConfig.from_env()

    def should_use_chunking(self, tokens: mx.array) -> bool:
        """Determine if chunking should be used for the given tokens.

        Args:
            tokens: Input token IDs (shape: [seq_len] or [batch, seq_len]).

        Returns:
            True if chunking should be used, False otherwise.
        """
        if not self.config.enable_chunking:
            return False

        # Get sequence length (handle both 1D and 2D token arrays)
        if tokens.ndim == 1:
            seq_len = tokens.shape[0]
        elif tokens.ndim == 2:
            seq_len = tokens.shape[1]  # batch dimension is 0
        else:
            logger.warning(f"Unexpected token shape: {tokens.shape}, skipping chunking")
            return False

        return seq_len >= self.config.min_tokens_for_chunking

    def prefill(
        self,
        model: Any,
        tokens: mx.array,
        cache: Optional[List[Any]] = None,
        prefill_fn: Optional[Any] = None,
    ) -> Tuple[mx.array, List[Any]]:
        """
        Run prefill with optional chunking.

        This method wraps the original prefill operation and applies chunking
        if conditions are met. If chunking is disabled or the prompt is too short,
        it delegates to the original prefill_fn.

        Args:
            model: The MLX model (passed to prefill_fn).
            tokens: Input token IDs (shape: [seq_len] or [batch, seq_len]).
            cache: Optional KV cache from previous requests/chunks.
            prefill_fn: Callable that performs a single prefill step.
                       Signature: prefill_fn(model, tokens, cache) -> (logits, cache)

        Returns:
            Tuple of (logits, updated_cache).

        Raises:
            ValueError: If prefill_fn is None or chunking produces invalid results.
        """
        if prefill_fn is None:
            raise ValueError("prefill_fn is required")

        # Fall back to traditional prefill if chunking is not needed
        if not self.should_use_chunking(tokens):
            logits, cache = prefill_fn(model, tokens, cache)
            return logits, cache

        logger.info(
            f"Using chunked prefill: tokens={tokens.shape}, "
            f"chunk_size={self.config.chunk_size}"
        )

        # Determine sequence length and chunking parameters
        if tokens.ndim == 1:
            seq_len = tokens.shape[0]
            is_1d = True
        else:
            seq_len = tokens.shape[1]
            is_1d = False

        chunk_size = self.config.chunk_size
        all_caches = cache if cache is not None else []
        logits = None

        # Process tokens in chunks
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)

            # Extract chunk
            if is_1d:
                chunk = tokens[start_idx:end_idx]
            else:
                chunk = tokens[:, start_idx:end_idx]

            logger.debug(
                f"Processing chunk {chunk_idx + 1}/{num_chunks}: "
                f"tokens [{start_idx}:{end_idx}]"
            )

            # Forward pass for this chunk
            try:
                chunk_logits, chunk_cache = prefill_fn(
                    model, chunk, all_caches if all_caches else None
                )

                # Force evaluation to ensure Metal GPU operations complete
                # This prevents "commit an already committed command buffer" error
                mx.eval(chunk_logits)
                if chunk_cache:
                    # Also eval cache to ensure all tensors are materialized
                    for cache_layer in chunk_cache:
                        if isinstance(cache_layer, tuple):
                            mx.eval(cache_layer[0])  # key
                            mx.eval(cache_layer[1])  # value
                        else:
                            mx.eval(cache_layer)
            except Exception as e:
                logger.error(
                    f"Prefill failed at chunk {chunk_idx}: {e}. "
                    "Falling back to traditional prefill."
                )
                # Fall back to processing the entire sequence at once
                # This is a safety mechanism for unexpect cache structures
                logits, cache = prefill_fn(model, tokens, cache)
                return logits, cache

            # Update cache for next chunk
            try:
                if all_caches is None or len(all_caches) == 0:
                    all_caches = chunk_cache
                else:
                    # Merge caches by concatenating along sequence dimension
                    all_caches = self._concatenate_caches(all_caches, chunk_cache)
            except Exception as e:
                logger.error(
                    f"Cache merge failed at chunk {chunk_idx}: {e}. "
                    "Falling back to traditional prefill."
                )
                # Fall back on cache merge error
                logits, cache = prefill_fn(model, tokens, cache)
                return logits, cache

            # Keep only the last logits (we only care about the final output)
            logits = chunk_logits

        logger.info(
            f"Chunked prefill completed: {num_chunks} chunks, "
            f"cache size: {len(all_caches)} layers"
        )

        return logits, all_caches

    @staticmethod
    def _concatenate_caches(
        cache1: List[Any], cache2: List[Any]
    ) -> List[Any]:
        """Concatenate KV caches from two chunks along sequence dimension.

        For each layer, concatenate the KV tensors from both chunks.
        This assumes cache format is compatible with mx.concatenate.

        Args:
            cache1: Cache from first chunk (list of layer caches).
            cache2: Cache from second chunk (list of layer caches).

        Returns:
            Concatenated cache (extended along sequence dimension).

        Raises:
            ValueError: If cache structures are incompatible.
        """
        if not cache1 or not cache2:
            return cache1 or cache2

        if len(cache1) != len(cache2):
            raise ValueError(
                f"Cache length mismatch: {len(cache1)} vs {len(cache2)}"
            )

        merged = []
        for c1, c2 in zip(cache1, cache2):
            try:
                # Handle both tuple (k, v) and direct array formats
                if isinstance(c1, tuple) and isinstance(c2, tuple):
                    # Tuple format: (key, value)
                    k1, v1 = c1
                    k2, v2 = c2
                    merged_k = mx.concatenate([k1, k2], axis=-2)  # seq_len axis
                    merged_v = mx.concatenate([v1, v2], axis=-2)
                    merged.append((merged_k, merged_v))
                else:
                    # Direct array format (less common but handle it)
                    merged_cache = mx.concatenate([c1, c2], axis=-2)
                    merged.append(merged_cache)
            except Exception as e:
                raise ValueError(
                    f"Failed to concatenate cache layer: {e}. "
                    f"c1 shape: {c1.shape if hasattr(c1, 'shape') else 'unknown'}, "
                    f"c2 shape: {c2.shape if hasattr(c2, 'shape') else 'unknown'}"
                ) from e

        return merged
