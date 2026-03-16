"""
Speculative Decoding Engine for ThunderOMLX.

Implements speculative decoding to accelerate inference using a small draft model
to generate candidate tokens and a large target model to verify them in parallel.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for Speculative Decoding."""

    enabled: bool = False
    draft_model_path: Optional[str] = None
    num_speculative_tokens: int = 4  # K
    draft_max_batch_size: int = 1
    acceptance_threshold: float = 0.0  # 0 = greedy matching

    @classmethod
    def from_env(cls) -> "SpeculativeConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("OMLX_ENABLE_SPECULATIVE_DECODING", "false").lower()
            == "true",
            draft_model_path=os.getenv("OMLX_DRAFT_MODEL_PATH"),
            num_speculative_tokens=int(
                os.getenv("OMLX_NUM_SPECULATIVE_TOKENS", "4")
            ),
            draft_max_batch_size=int(os.getenv("OMLX_DRAFT_MAX_BATCH_SIZE", "1")),
        )


class SpeculativeDecodingEngine:
    """
    Speculative Decoding Engine.

    Manages draft model and target model, coordinating the speculative generation process.
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_tokenizer,
        draft_model_path: str,
        num_speculative_tokens: int = 4,
        config: Optional[SpeculativeConfig] = None,
    ):
        """
        Initialize Speculative Decoding Engine.

        Args:
            target_model: The large target model (already loaded)
            target_tokenizer: Tokenizer for target model
            draft_model_path: Path to draft model directory
            num_speculative_tokens: Number of speculative tokens (K)
            config: Optional configuration
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.K = num_speculative_tokens
        self.config = config or SpeculativeConfig()

        # Load draft model
        logger.info(f"Loading draft model from {draft_model_path}")
        self.draft_model, self.draft_tokenizer = mlx_load(draft_model_path)
        logger.info(
            f"Draft model loaded: {self.draft_model.__class__.__name__} "
            f"(~{self._estimate_model_size(self.draft_model):.1f} MB)"
        )

        # KV Cache for both models
        self.target_cache = None
        self.draft_cache = None

        # Statistics
        self.stats = {
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "total_rejected_tokens": 0,
            "total_bonus_tokens": 0,
            "total_forward_passes": 0,
        }

    @staticmethod
    def _estimate_model_size(model: nn.Module) -> float:
        """Estimate model size in MB."""
        # model.parameters() returns dict of {name: array}
        import numpy as np

        total_params = sum(
            np.prod(p.shape) for p in model.parameters().values() if hasattr(p, "shape")
        )
        # Rough estimate: 4 bytes per parameter for float32
        return total_params * 4 / (1024 * 1024)

    def _create_cache(self, model: nn.Module) -> List:
        """Create KV cache for model."""
        # MLX uses list of (key, value) tuples for cache
        # Initialize empty cache (will be created on first forward pass)
        return None

    def _prefill(self, prompt_tokens: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Prefill both models with prompt tokens.

        Args:
            prompt_tokens: Input prompt tokens [1, seq_len]

        Returns:
            (target_logits, draft_logits)
        """
        logger.debug(f"Prefilling with {prompt_tokens.shape[1]} tokens")

        # Target model prefill
        target_logits = self.target_model(prompt_tokens, cache=self.target_cache)
        if isinstance(target_logits, tuple):
            target_logits, self.target_cache = target_logits

        # Draft model prefill
        draft_logits = self.draft_model(prompt_tokens, cache=self.draft_cache)
        if isinstance(draft_logits, tuple):
            draft_logits, self.draft_cache = draft_logits

        mx.eval(target_logits, draft_logits)

        self.stats["total_forward_passes"] += 2

        return target_logits, draft_logits

    def _draft_generate(self, current_token: int) -> List[int]:
        """
        Draft model generates K candidate tokens using greedy sampling.

        Args:
            current_token: Starting token

        Returns:
            List of K candidate tokens
        """
        draft_tokens = []
        token = current_token

        for _ in range(self.K):
            # Forward pass with single token
            input_ids = mx.array([[token]])
            logits = self.draft_model(input_ids, cache=self.draft_cache)

            if isinstance(logits, tuple):
                logits, self.draft_cache = logits

            # Greedy sampling: argmax
            next_token = mx.argmax(logits[0, -1, :]).item()
            draft_tokens.append(next_token)
            token = next_token

            self.stats["total_forward_passes"] += 1

        mx.eval(self.draft_cache)  # Ensure cache is evaluated

        self.stats["total_draft_tokens"] += len(draft_tokens)

        return draft_tokens

    def _verify_and_accept(
        self, draft_tokens: List[int], current_position: int
    ) -> Tuple[int, Optional[int]]:
        """
        Target model verifies candidate tokens in parallel.

        Args:
            draft_tokens: List of K candidate tokens from draft model
            current_position: Current position in sequence

        Returns:
            (num_accepted, bonus_token)
            - num_accepted: Number of accepted tokens (0 to K)
            - bonus_token: If < K tokens accepted, generate 1 bonus token; else None
        """
        # Parallel forward pass with all K candidate tokens
        input_ids = mx.array([draft_tokens])
        logits = self.target_model(input_ids, cache=self.target_cache)

        if isinstance(logits, tuple):
            logits, new_cache = logits
        else:
            new_cache = None

        mx.eval(logits)

        self.stats["total_forward_passes"] += 1

        # Verify each token
        accepted_count = 0
        for i, draft_token in enumerate(draft_tokens):
            # Get target model's prediction at position i
            target_probs = mx.softmax(logits[0, i, :], axis=-1)
            target_token = mx.argmax(target_probs).item()

            if target_token == draft_token:
                # Accept
                accepted_count += 1
            else:
                # Reject: stop here
                logger.debug(
                    f"Rejected at position {i}: "
                    f"draft={draft_token}, target={target_token}"
                )
                break

        # Update target cache
        if new_cache is not None:
            if accepted_count > 0:
                # Accept tokens: update cache to include accepted tokens
                self.target_cache = self._trim_cache(new_cache, accepted_count)
            # If accepted_count == 0, keep old cache (no tokens accepted)

        # Update draft cache: rollback to match accepted tokens
        if accepted_count < self.K:
            self._rollback_draft_cache(accepted_count)

        self.stats["total_accepted_tokens"] += accepted_count
        self.stats["total_rejected_tokens"] += self.K - accepted_count

        # Generate bonus token if not all accepted
        bonus_token = None
        if accepted_count < self.K:
            # Use target model's prediction at rejection point
            if accepted_count < len(draft_tokens):
                target_probs = mx.softmax(logits[0, accepted_count, :], axis=-1)
                bonus_token = mx.argmax(target_probs).item()
                self.stats["total_bonus_tokens"] += 1

        return accepted_count, bonus_token

    def _trim_cache(self, cache, num_tokens: int):
        """
        Trim cache to keep only first num_tokens.

        Args:
            cache: KV cache (list of (key, value) tuples)
            num_tokens: Number of tokens to keep

        Returns:
            Trimmed cache
        """
        if cache is None:
            return None

        # MLX cache format: list of (key, value) tuples
        # Each key/value shape: [batch, num_heads, seq_len, head_dim]
        trimmed = []
        for key, value in cache:
            # Trim sequence dimension
            trimmed_key = key[:, :, :num_tokens, :]
            trimmed_value = value[:, :, :num_tokens, :]
            trimmed.append((trimmed_key, trimmed_value))

        return trimmed

    def _rollback_draft_cache(self, num_accepted: int):
        """
        Rollback draft cache to match number of accepted tokens.

        Args:
            num_accepted: Number of accepted tokens (0 to K-1)
        """
        if self.draft_cache is None:
            return

        # Rollback: remove last (K - num_accepted) tokens from draft cache
        num_to_remove = self.K - num_accepted

        if num_to_remove > 0:
            # Get current cache length
            if len(self.draft_cache) > 0:
                cache_len = self.draft_cache[0][0].shape[2]  # seq_len dimension
                new_len = cache_len - num_to_remove

                if new_len > 0:
                    self.draft_cache = self._trim_cache(self.draft_cache, new_len)
                else:
                    # Reset cache if all tokens rejected
                    self.draft_cache = None

    def generate_speculative(
        self,
        prompt_tokens: mx.array,
        max_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> Iterator[int]:
        """
        Generate tokens using Speculative Decoding.

        Args:
            prompt_tokens: Input prompt tokens [1, seq_len]
            max_tokens: Maximum number of tokens to generate
            eos_token_id: Optional EOS token ID to stop generation

        Yields:
            Generated tokens
        """
        # Reset statistics
        self.stats = {
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "total_rejected_tokens": 0,
            "total_bonus_tokens": 0,
            "total_forward_passes": 0,
        }

        # 1. Prefill phase
        target_logits, draft_logits = self._prefill(prompt_tokens)

        # Get first token from target model
        current_token = mx.argmax(target_logits[0, -1, :]).item()
        yield current_token

        # Update draft cache with first token (sync with target)
        # This ensures draft cache is aligned with generated sequence
        first_token_input = mx.array([[current_token]])
        first_token_logits = self.draft_model(first_token_input, cache=self.draft_cache)
        if isinstance(first_token_logits, tuple):
            first_token_logits, self.draft_cache = first_token_logits
        mx.eval(self.draft_cache)

        # Update target cache with first token
        first_token_logits_target = self.target_model(first_token_input, cache=self.target_cache)
        if isinstance(first_token_logits_target, tuple):
            first_token_logits_target, self.target_cache = first_token_logits_target
        mx.eval(self.target_cache)

        num_generated = 1

        # 2. Decode loop with speculative decoding
        while num_generated < max_tokens:
            # 2.1 Draft phase: generate K candidates
            draft_tokens = self._draft_generate(current_token)

            # 2.2 Verify phase: parallel verification
            accepted_count, bonus_token = self._verify_and_accept(
                draft_tokens, num_generated
            )

            # 2.3 Yield accepted tokens
            for i in range(accepted_count):
                token = draft_tokens[i]
                yield token
                num_generated += 1
                current_token = token

                # Check EOS
                if eos_token_id is not None and token == eos_token_id:
                    logger.debug(f"EOS token encountered at {num_generated}")
                    return

                if num_generated >= max_tokens:
                    break

            # 2.4 Yield bonus token (if any)
            if bonus_token is not None and num_generated < max_tokens:
                yield bonus_token
                num_generated += 1
                current_token = bonus_token

                # Check EOS
                if eos_token_id is not None and bonus_token == eos_token_id:
                    logger.debug(f"EOS token encountered at {num_generated}")
                    return

        # Log final statistics
        self._log_statistics()

    def _log_statistics(self):
        """Log performance statistics."""
        total_draft = self.stats["total_draft_tokens"]
        total_accepted = self.stats["total_accepted_tokens"]
        total_bonus = self.stats["total_bonus_tokens"]

        if total_draft > 0:
            acceptance_rate = total_accepted / total_draft
            logger.info(
                f"Speculative Decoding Stats: "
                f"acceptance_rate={acceptance_rate:.2%}, "
                f"accepted={total_accepted}, "
                f"drafted={total_draft}, "
                f"bonus={total_bonus}, "
                f"forward_passes={self.stats['total_forward_passes']}"
            )

    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        total_draft = self.stats["total_draft_tokens"]
        if total_draft == 0:
            return 0.0
        return self.stats["total_accepted_tokens"] / total_draft

    def get_speedup_ratio(self) -> float:
        """
        Estimate speedup ratio based on acceptance rate.

        Theoretical speedup = 1 + α * K
        where α = acceptance rate
        """
        alpha = self.get_acceptance_rate()
        return 1.0 + alpha * self.K
