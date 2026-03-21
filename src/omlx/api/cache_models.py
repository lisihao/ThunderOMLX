# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Named Prompt Cache API.

These models define the request and response schemas for:
- Saving prompt KV cache with KVTC compression
- Loading and pre-registering cached prompts
- Listing and managing named caches
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptCacheSaveRequest(BaseModel):
    """Request for saving a prompt's KV cache with KVTC compression."""

    name: str = Field(
        ...,
        description="Cache name (e.g., 'system-coding-v1')",
    )
    prompt: str = Field(
        ...,
        description="Prompt text to prefill and cache",
    )
    model: Optional[str] = Field(
        None,
        description="Model name (None = default loaded model)",
    )
    compress: bool = Field(
        True,
        description="Apply KVTC compression (recommended)",
    )


class PromptCacheSaveResponse(BaseModel):
    """Response from saving a prompt cache."""

    name: str
    model: str
    token_count: int
    file_size_bytes: int
    compression_ratio: float
    encode_time_ms: float


class PromptCacheLoadRequest(BaseModel):
    """Request for loading a named prompt cache."""

    name: str = Field(
        ...,
        description="Cache name to load",
    )


class PromptCacheLoadResponse(BaseModel):
    """Response from loading a prompt cache."""

    name: str
    model: str
    token_count: int
    ready: bool
    load_time_ms: float


class PromptCacheInfo(BaseModel):
    """Metadata for a single named prompt cache."""

    name: str
    model: str
    token_count: int
    file_size_bytes: int
    compression_ratio: float
    compressed: bool
    created_at: float


class PromptCacheListResponse(BaseModel):
    """Response listing all named prompt caches."""

    caches: List[PromptCacheInfo]
    total_size_bytes: int
    total_count: int


class PromptCacheDeleteResponse(BaseModel):
    """Response from deleting a prompt cache."""

    name: str
    deleted: bool
