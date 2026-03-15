"""
ContextPilot integration for ThunderOMLX.

Provides cross-request cache optimization through context indexing,
reordering, and deduplication.
"""

from .adapter import ContextBlock, ContextIndex, ContextPilotAdapter, OptimizedRequest

__all__ = [
    "ContextBlock",
    "ContextIndex",
    "ContextPilotAdapter",
    "OptimizedRequest",
]
