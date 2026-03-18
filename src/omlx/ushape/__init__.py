# SPDX-License-Identifier: Apache-2.0
"""
U-Shape Reinforcement module for ThunderOMLX.

Mitigates the "Lost in the Middle" attention degradation by appending
BM25-selected context summaries to the prompt tail (high attention zone).
"""

from .augmenter import UShapeAugmenter
from .types import ScoredChunk, UShapeConfig

__all__ = ["UShapeConfig", "ScoredChunk", "UShapeAugmenter"]
