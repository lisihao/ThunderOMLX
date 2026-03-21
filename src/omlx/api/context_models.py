# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Context Optimization API.

These models define the request and response schemas for:
- Context optimization for better cache hit rates
- Message reordering and deduplication
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContextOptimizeRequest(BaseModel):
    """Request for optimizing message context for better cache hit rates."""

    messages: List[Dict[str, Any]] = Field(
        ...,
        description="Messages to optimize",
    )
    token_budget: Optional[int] = Field(
        None,
        description="Optional token budget constraint",
    )
    features: Optional[List[str]] = Field(
        None,
        description="Optimization features to enable (e.g., prefix_align, semantic_dedup)",
    )
    previous_requests: Optional[List[List[Dict[str, Any]]]] = Field(
        None,
        description="Previous requests for context alignment",
    )


class ContextOptimizeResponse(BaseModel):
    """Response containing optimized message context."""

    optimized_messages: List[Dict[str, Any]] = Field(
        ...,
        description="Optimized messages with reordered content",
    )
    estimated_tokens: int = Field(
        ...,
        description="Estimated token count after optimization",
    )
    prefix_hash: Optional[str] = Field(
        None,
        description="Hash of stable system prompt prefix",
    )
    dedup_count: int = Field(
        0,
        description="Number of deduplicated content blocks",
    )
    cache_hint: Optional[str] = Field(
        None,
        description="Caching strategy hint (e.g., system_prompt_stable)",
    )


class CompactSubmitRequest(BaseModel):
    """Request for async context compaction via local model summarization."""
    messages: List[Dict[str, Any]] = Field(
        ...,
        description="Messages to compact/summarize",
    )
    token_budget: Optional[int] = Field(
        None,
        description="Target token count after compaction",
    )
    model: Optional[str] = Field(
        None,
        description="Model to use for summarization (None = default)",
    )
    custom_instructions: Optional[str] = Field(
        None,
        description="Custom instructions for summarization",
    )


class CompactSubmitResponse(BaseModel):
    """Response from compact submission."""
    task_id: str = Field(..., description="Task ID for polling")
    status: str = Field("pending", description="Task status")


class CompactStatusResponse(BaseModel):
    """Response from compact status polling."""
    task_id: str
    status: str = Field(..., description="pending | running | done | failed")
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Compaction result (only when status=done)",
    )
    error: Optional[str] = Field(None, description="Error message (only when status=failed)")
