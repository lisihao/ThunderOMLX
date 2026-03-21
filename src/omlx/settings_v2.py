# SPDX-License-Identifier: Apache-2.0
"""
Global settings management for oMLX using Pydantic v2.

This module provides a centralized settings system with:
- Pydantic v2 BaseSettings for type safety and validation
- Hierarchical configuration (Default → settings.json → env vars → CLI)
- Automatic directory creation
- System resource detection (RAM, SSD capacity)
- Settings persistence to JSON file (v1 format compatible)

Usage:
    from omlx.settings_v2 import GlobalSettingsV2

    # Load settings with all priority levels
    settings = GlobalSettingsV2.load(base_path=Path.home() / ".omlx", cli_args=args)

    # Access values
    print(settings.server.port)
    print(settings.model.get_max_model_memory_bytes())

    # Save to file (v1 JSON format)
    settings.save()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import parse_size

if TYPE_CHECKING:
    from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)

# Settings file version for future migrations
SETTINGS_VERSION = "1.0"

# Default base path
DEFAULT_BASE_PATH = Path.home() / ".omlx"


def get_system_memory() -> int:
    """
    Return total system RAM in bytes.

    Uses psutil if available, falls back to os.sysconf on Unix.

    Returns:
        Total RAM in bytes.
    """
    try:
        import psutil

        return psutil.virtual_memory().total
    except ImportError:
        pass

    # Fallback for Unix systems
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size
    except (AttributeError, ValueError):
        pass

    # Default to 16GB if detection fails
    logger.warning("Could not detect system memory, defaulting to 16GB")
    return 16 * 1024**3


def _adaptive_system_reserve(total: int) -> int:
    """Adaptive system reservation: 20% of total, clamped to [2GB, 8GB]."""
    reserve = int(total * 0.20)
    min_reserve = 2 * 1024**3
    max_reserve = 8 * 1024**3
    return max(min_reserve, min(reserve, max_reserve))


def get_ssd_capacity(path: str | Path) -> int:
    """
    Return disk capacity in bytes for the given path.

    Args:
        path: Path to check disk capacity for.

    Returns:
        Total disk capacity in bytes.
    """
    path = Path(path).expanduser().resolve()

    # Ensure parent directory exists for capacity check
    check_path = path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    try:
        usage = shutil.disk_usage(check_path)
        return usage.total
    except OSError as e:
        logger.warning(f"Could not get disk capacity for {path}: {e}")
        # Default to 500GB if detection fails
        return 500 * 1024**3


def parse_bool(v: Any) -> bool:
    """
    Parse boolean values from strings.

    Args:
        v: Value to parse (bool, str, or other)

    Returns:
        Parsed boolean value.
    """
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes", "on")
    return bool(v)


# Type annotation for boolean strings
BoolStr = type("BoolStr", (), {})()  # Placeholder


class ServerSettingsV2(BaseModel):
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be 1-65535, got {v}")
        return v


class ModelSettingsV2(BaseModel):
    """Model configuration settings."""

    model_dirs: list[str] = Field(default_factory=list)
    model_dir: Optional[str] = None
    max_model_memory: str = "auto"

    def get_model_dirs(self, base_path: Path) -> list[Path]:
        """
        Get the resolved model directory paths.

        Args:
            base_path: Base oMLX directory.

        Returns:
            List of resolved model directory paths.
        """
        if self.model_dirs:
            return [Path(d).expanduser().resolve() for d in self.model_dirs]
        if self.model_dir:
            return [Path(self.model_dir).expanduser().resolve()]
        return [base_path / "models"]

    def get_model_dir(self, base_path: Path) -> Path:
        """
        Get the primary (first) resolved model directory path.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Resolved primary model directory path.
        """
        return self.get_model_dirs(base_path)[0]

    def get_max_model_memory_bytes(self) -> Optional[int]:
        """
        Get max model memory in bytes, or None if disabled.

        Returns:
            Max model memory in bytes (90% of usable RAM if "auto"),
            or None if disabled (no limit).
        """
        value = self.max_model_memory.strip().lower()
        if value == "disabled":
            return None
        if value == "auto":
            total = get_system_memory()
            reserve = _adaptive_system_reserve(total)
            return max(1 * 1024**3, int((total - reserve) * 0.9))
        return parse_size(self.max_model_memory)

    @field_validator("max_model_memory")
    @classmethod
    def validate_max_model_memory(cls, v: str) -> str:
        """Validate max_model_memory format."""
        value = v.strip().lower()
        if value in ("auto", "disabled"):
            return v
        try:
            parse_size(v)
        except ValueError as e:
            raise ValueError(f"Invalid max_model_memory: {e}") from e
        return v

    @model_validator(mode="after")
    def migrate_deprecated(self) -> Self:
        """Migrate deprecated model_dir to model_dirs."""
        if not self.model_dirs and self.model_dir:
            self.model_dirs = [self.model_dir]
        return self


class SchedulerSettingsV2(BaseModel):
    """Scheduler configuration settings."""

    max_num_seqs: int = 8
    completion_batch_size: int = 8

    @field_validator("max_num_seqs", "completion_batch_size")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate that values are positive."""
        if v <= 0:
            raise ValueError(f"Value must be > 0, got {v}")
        return v


class CacheSettingsV2(BaseModel):
    """Cache configuration settings."""

    enabled: bool = Field(default=True)
    ssd_cache_dir: Optional[str] = None
    ssd_cache_max_size: str = "auto"
    hot_cache_max_size: str = "0"
    initial_cache_blocks: int = 256

    def get_ssd_cache_dir(self, base_path: Path) -> Path:
        """
        Get the resolved SSD cache directory path.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Resolved SSD cache directory path.
        """
        if self.ssd_cache_dir:
            return Path(self.ssd_cache_dir).expanduser().resolve()
        return base_path / "cache"

    def get_ssd_cache_max_size_bytes(self, base_path: Path) -> int:
        """
        Get max SSD cache size in bytes.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Max SSD cache size in bytes (10% of SSD if "auto").
        """
        if self.ssd_cache_max_size.lower() == "auto":
            cache_dir = self.get_ssd_cache_dir(base_path)
            return int(get_ssd_capacity(cache_dir) * 0.1)
        return parse_size(self.ssd_cache_max_size)

    def get_hot_cache_max_size_bytes(self) -> int:
        """Get hot cache max size in bytes. 0 means disabled."""
        return parse_size(self.hot_cache_max_size)

    @field_validator("ssd_cache_max_size", "hot_cache_max_size")
    @classmethod
    def validate_size_format(cls, v: str, info) -> str:
        """Validate size string format."""
        value = v.lower()
        if value == "auto":
            return v
        try:
            parse_size(v)
        except ValueError as e:
            field_name = info.field_name
            raise ValueError(f"Invalid {field_name}: {e}") from e
        return v

    # KVTC compression (Phase 3: optional alternative to lz4 for SSD blocks)
    kvtc_enabled: bool = Field(
        default=False,
        description="Use KVTC compression for SSD cache blocks (4-8x vs lz4 2-3x). "
        "Requires per-model calibration on first use.",
    )
    kvtc_energy: float = Field(
        default=0.995,
        description="PCA energy retention threshold (0.9-0.999). "
        "Higher = better quality, lower compression.",
    )
    kvtc_bits: int = Field(
        default=4,
        description="Quantization bit depth (2, 4, or 8). "
        "Lower = more compression, slightly more loss.",
    )
    kvtc_group_size: int = Field(
        default=64,
        description="Quantization group size for per-group scaling.",
    )
    kvtc_adaptive: bool = Field(
        default=False,
        description="Automatically choose KVTC or lz4 per block based on token count. "
        "KVTC for small blocks (faster SSD load), lz4 for large blocks (lower decode overhead).",
    )
    kvtc_threshold: int = Field(
        default=2048,
        description="Token count threshold for adaptive mode. "
        "Blocks with token_count <= threshold use KVTC, otherwise lz4.",
    )

    @field_validator("initial_cache_blocks")
    @classmethod
    def validate_initial_cache_blocks(cls, v: int) -> int:
        """Validate initial_cache_blocks is positive."""
        if v <= 0:
            raise ValueError(f"initial_cache_blocks must be > 0, got {v}")
        return v

    @field_validator("kvtc_energy")
    @classmethod
    def validate_kvtc_energy(cls, v: float) -> float:
        if not (0.5 <= v <= 1.0):
            raise ValueError(f"kvtc_energy must be 0.5-1.0, got {v}")
        return v

    @field_validator("kvtc_bits")
    @classmethod
    def validate_kvtc_bits(cls, v: int) -> int:
        if v not in (2, 4, 8):
            raise ValueError(f"kvtc_bits must be 2, 4, or 8, got {v}")
        return v


class MemorySettingsV2(BaseModel):
    """Process-level memory enforcement settings."""

    max_process_memory: str = "auto"

    def get_max_process_memory_bytes(self) -> Optional[int]:
        """
        Get max process memory in bytes, or None if disabled.

        - "auto": system RAM minus 8GB
        - "disabled": None (no enforcement)
        - "XX%": percentage of system RAM (10-99%)

        Returns:
            Max process memory in bytes, or None if disabled.
        """
        value = self.max_process_memory.strip().lower()
        if value == "disabled":
            return None
        if value == "auto":
            total = get_system_memory()
            reserve = _adaptive_system_reserve(total)
            return total - reserve
        # Parse percentage like "80%"
        percent_str = value.rstrip("%")
        try:
            percent = int(percent_str)
        except ValueError:
            # Try parsing as absolute size (e.g., "32GB")
            return parse_size(self.max_process_memory)
        if not 10 <= percent <= 99:
            raise ValueError(f"max_process_memory must be 10-99%, got {percent}%")
        return int(get_system_memory() * percent / 100)

    @field_validator("max_process_memory")
    @classmethod
    def validate_max_process_memory(cls, v: str) -> str:
        """Validate max_process_memory format."""
        value = v.strip().lower()
        if value in ("auto", "disabled"):
            return v
        # Check if it's a percentage
        if value.endswith("%"):
            try:
                percent = int(value[:-1])
                if not 10 <= percent <= 99:
                    raise ValueError(f"Percentage must be 10-99%, got {percent}%")
            except ValueError as e:
                raise ValueError(f"Invalid percentage format: {e}") from e
            return v
        # Try parsing as size
        try:
            parse_size(v)
        except ValueError as e:
            raise ValueError(f"Invalid max_process_memory: {e}") from e
        return v


class SubKeyEntryV2(BaseModel):
    """A sub API key entry for API-only authentication."""

    key: str
    name: str = ""
    created_at: str = ""


class AuthSettingsV2(BaseModel):
    """Authentication configuration settings."""

    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    skip_api_key_verification: bool = False
    sub_keys: list[SubKeyEntryV2] = Field(default_factory=list)


class MCPSettingsV2(BaseModel):
    """MCP (Model Context Protocol) configuration settings."""

    config_path: Optional[str] = None


class HuggingFaceSettingsV2(BaseModel):
    """HuggingFace Hub configuration settings."""

    endpoint: str = ""

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint is a valid URL."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid huggingface endpoint: '{v}' (must start with http:// or https://)"
            )
        return v


class SamplingSettingsV2(BaseModel):
    """Default sampling parameters for generation."""

    max_context_window: int = 32768
    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    @field_validator("max_context_window", "max_tokens")
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        """Validate that values are positive."""
        if v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be > 0, got {v}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {v}")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """Validate top_p is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"top_p must be 0.0-1.0, got {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k is non-negative."""
        if v < 0:
            raise ValueError(f"top_k must be >= 0, got {v}")
        return v


class LoggingSettingsV2(BaseModel):
    """Logging configuration settings."""

    log_dir: Optional[str] = None
    retention_days: int = 7

    def get_log_dir(self, base_path: Path) -> Path:
        """
        Get the resolved log directory path.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Resolved log directory path.
        """
        if self.log_dir:
            return Path(self.log_dir).expanduser().resolve()
        return base_path / "logs"

    @field_validator("retention_days")
    @classmethod
    def validate_retention_days(cls, v: int) -> int:
        """Validate retention_days is positive."""
        if v <= 0:
            raise ValueError(f"retention_days must be > 0, got {v}")
        return v


class UISettingsV2(BaseModel):
    """Admin UI settings."""

    language: str = "en"


class ClaudeCodeSettingsV2(BaseModel):
    """Claude Code integration settings."""

    context_scaling_enabled: bool = False
    target_context_size: int = 200000
    mode: str = "cloud"
    opus_model: Optional[str] = None
    sonnet_model: Optional[str] = None
    haiku_model: Optional[str] = None

    @field_validator("target_context_size")
    @classmethod
    def validate_target_context_size(cls, v: int) -> int:
        """Validate target_context_size is positive."""
        if v <= 0:
            raise ValueError(f"target_context_size must be > 0, got {v}")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is one of allowed values."""
        valid_modes = {"local", "cloud"}
        if v not in valid_modes:
            raise ValueError(
                f"Invalid claude_code mode: '{v}' (must be one of {sorted(valid_modes)})"
            )
        return v


class IntegrationSettingsV2(BaseModel):
    """Other integrations settings (Codex, OpenCode, OpenClaw)."""

    codex_model: Optional[str] = None
    opencode_model: Optional[str] = None
    openclaw_model: Optional[str] = None
    openclaw_tools_profile: str = "coding"


class CloudSettingsV2(BaseModel):
    """Cloud backend settings for edge-cloud routing."""

    enabled: bool = False

    # API Keys
    deepseek_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    glm_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    chatgpt_access_token: Optional[str] = None

    # Budget
    daily_budget: float = 5.0
    monthly_budget: float = 100.0

    # Routing
    prefer_local: bool = True
    fallback_enabled: bool = True

    # Cloud model names
    cloud_models: list[str] = [
        "deepseek-r1", "deepseek-v3",
        "glm-5", "glm-4-flash", "glm-4-plus",
        "gpt-5.2", "gpt-5.1", "gpt-5.1-codex-max", "gpt-5.1-codex-mini",
        "gpt-4o", "gpt-4o-mini",
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2-flash", "gemini-2-pro",
    ]

    # Queue scheduling
    max_queue_depth: int = 200
    agent_fair_share: float = 0.6
    concurrency: dict[str, int] = Field(
        default_factory=lambda: {
            "deepseek": 5, "glm": 5, "openai": 3, "chatgpt": 2, "gemini": 5,
        }
    )

    # Context optimization
    context_pilot_enabled: bool = True
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85
    semantic_cache_ttl: int = 14400

    # Conversation store
    conversation_store_enabled: bool = True
    conversation_db_path: Optional[str] = None

    # Intelligent routing (model="auto")
    intelligent_routing_enabled: bool = False
    intelligent_routing_shadow: bool = True  # Shadow mode: log decisions but don't override
    local_overflow_threshold: int = 4        # Local queue depth before overflow to cloud
    session_pin_threshold: int = 3           # Consecutive same-model turns before pinning


class GlobalSettingsV2(BaseSettings):
    """
    Global settings for oMLX using Pydantic v2.

    Combines all settings sections and provides methods for:
    - Loading from file with CLI/env overrides
    - Saving to file
    - Directory management
    - Validation
    """

    model_config = SettingsConfigDict(
        env_prefix="OMLX_",
        env_nested_delimiter="__",
        extra="ignore",
        validate_assignment=True,
    )

    version: str = SETTINGS_VERSION
    base_path: Path = Field(default=DEFAULT_BASE_PATH)

    server: ServerSettingsV2 = Field(default_factory=ServerSettingsV2)
    model: ModelSettingsV2 = Field(default_factory=ModelSettingsV2)
    memory: MemorySettingsV2 = Field(default_factory=MemorySettingsV2)
    scheduler: SchedulerSettingsV2 = Field(default_factory=SchedulerSettingsV2)
    cache: CacheSettingsV2 = Field(default_factory=CacheSettingsV2)
    auth: AuthSettingsV2 = Field(default_factory=AuthSettingsV2)
    mcp: MCPSettingsV2 = Field(default_factory=MCPSettingsV2)
    huggingface: HuggingFaceSettingsV2 = Field(default_factory=HuggingFaceSettingsV2)
    sampling: SamplingSettingsV2 = Field(default_factory=SamplingSettingsV2)
    logging: LoggingSettingsV2 = Field(default_factory=LoggingSettingsV2)
    claude_code: ClaudeCodeSettingsV2 = Field(default_factory=ClaudeCodeSettingsV2)
    integrations: IntegrationSettingsV2 = Field(default_factory=IntegrationSettingsV2)
    cloud: CloudSettingsV2 = Field(default_factory=CloudSettingsV2)
    ui: UISettingsV2 = Field(default_factory=UISettingsV2)

    @classmethod
    def load(
        cls,
        base_path: Optional[Path | str] = None,
        cli_args: Optional[Any] = None,
    ) -> GlobalSettingsV2:
        """
        Load settings with priority hierarchy: Default → file → env → CLI.

        Four-layer loading:
        1. Default values (from field defaults and SettingsConfigDict)
        2. Load from settings.json file
        3. Apply OMLX_* environment variables
        4. Apply CLI argument overrides

        Args:
            base_path: Base directory for oMLX (default: ~/.omlx).
            cli_args: Argparse namespace with CLI arguments.

        Returns:
            Loaded GlobalSettingsV2 instance.
        """
        # Resolve base path
        if base_path:
            resolved_base = Path(base_path).expanduser().resolve()
        else:
            resolved_base = DEFAULT_BASE_PATH

        # Start with defaults
        settings = cls(base_path=resolved_base)

        # Load from file if exists (layer 2)
        settings_file = resolved_base / "settings.json"
        if settings_file.exists():
            settings._load_from_file(settings_file)
            logger.debug(f"Loaded settings from {settings_file}")

        # Apply environment variable overrides (layer 3)
        settings._apply_env_overrides()

        # Apply CLI argument overrides (layer 4)
        if cli_args:
            settings._apply_cli_overrides(cli_args)

        return settings

    def _apply_env_overrides(self) -> None:
        """Apply OMLX_* environment variable overrides."""
        # Server settings
        if host := os.getenv("OMLX_HOST"):
            self.server.host = host
        if port := os.getenv("OMLX_PORT"):
            try:
                self.server.port = int(port)
            except ValueError:
                logger.warning(f"Invalid OMLX_PORT value: {port}")
        if log_level := os.getenv("OMLX_LOG_LEVEL"):
            self.server.log_level = log_level

        # Model settings
        if model_dir := os.getenv("OMLX_MODEL_DIR"):
            dirs = [d.strip() for d in model_dir.split(",") if d.strip()]
            self.model.model_dirs = dirs
            self.model.model_dir = dirs[0] if dirs else None
        if max_model_memory := os.getenv("OMLX_MAX_MODEL_MEMORY"):
            self.model.max_model_memory = max_model_memory

        # Memory enforcement settings
        if max_process_memory := os.getenv("OMLX_MAX_PROCESS_MEMORY"):
            self.memory.max_process_memory = max_process_memory

        # Scheduler settings
        if max_num_seqs := os.getenv("OMLX_MAX_NUM_SEQS"):
            try:
                self.scheduler.max_num_seqs = int(max_num_seqs)
            except ValueError:
                logger.warning(f"Invalid OMLX_MAX_NUM_SEQS value: {max_num_seqs}")
        if completion_batch := os.getenv("OMLX_COMPLETION_BATCH_SIZE"):
            try:
                self.scheduler.completion_batch_size = int(completion_batch)
            except ValueError:
                logger.warning(
                    f"Invalid OMLX_COMPLETION_BATCH_SIZE: {completion_batch}"
                )

        # Cache settings
        if cache_enabled := os.getenv("OMLX_CACHE_ENABLED"):
            self.cache.enabled = cache_enabled.lower() in ("true", "1", "yes", "on")
        if ssd_cache_dir := os.getenv("OMLX_SSD_CACHE_DIR"):
            self.cache.ssd_cache_dir = ssd_cache_dir
        if ssd_cache_max := os.getenv("OMLX_SSD_CACHE_MAX_SIZE"):
            self.cache.ssd_cache_max_size = ssd_cache_max
        if hot_cache_max := os.getenv("OMLX_HOT_CACHE_MAX_SIZE"):
            self.cache.hot_cache_max_size = hot_cache_max
        if initial_blocks := os.getenv("OMLX_INITIAL_CACHE_BLOCKS"):
            try:
                self.cache.initial_cache_blocks = int(initial_blocks)
            except ValueError:
                logger.warning(
                    f"Invalid OMLX_INITIAL_CACHE_BLOCKS value: {initial_blocks}"
                )

        # Auth settings
        if api_key := os.getenv("OMLX_API_KEY"):
            self.auth.api_key = api_key
        if secret_key := os.getenv("OMLX_SECRET_KEY"):
            self.auth.secret_key = secret_key
        if skip_api_key_verification := os.getenv("OMLX_SKIP_API_KEY_VERIFICATION"):
            self.auth.skip_api_key_verification = skip_api_key_verification.lower() in ("true", "1", "yes", "on")

        # MCP settings
        if mcp_config := os.getenv("OMLX_MCP_CONFIG"):
            self.mcp.config_path = mcp_config

        # HuggingFace settings
        if hf_endpoint := os.getenv("OMLX_HF_ENDPOINT"):
            self.huggingface.endpoint = hf_endpoint

        # Sampling settings
        if max_context_window := os.getenv("OMLX_MAX_CONTEXT_WINDOW"):
            try:
                self.sampling.max_context_window = int(max_context_window)
            except ValueError:
                logger.warning(f"Invalid OMLX_MAX_CONTEXT_WINDOW: {max_context_window}")
        if max_tokens := os.getenv("OMLX_MAX_TOKENS"):
            try:
                self.sampling.max_tokens = int(max_tokens)
            except ValueError:
                logger.warning(f"Invalid OMLX_MAX_TOKENS: {max_tokens}")
        if temperature := os.getenv("OMLX_TEMPERATURE"):
            try:
                self.sampling.temperature = float(temperature)
            except ValueError:
                logger.warning(f"Invalid OMLX_TEMPERATURE: {temperature}")
        if top_p := os.getenv("OMLX_TOP_P"):
            try:
                self.sampling.top_p = float(top_p)
            except ValueError:
                logger.warning(f"Invalid OMLX_TOP_P: {top_p}")
        if top_k := os.getenv("OMLX_TOP_K"):
            try:
                self.sampling.top_k = int(top_k)
            except ValueError:
                logger.warning(f"Invalid OMLX_TOP_K: {top_k}")
        if repetition_penalty := os.getenv("OMLX_REPETITION_PENALTY"):
            try:
                self.sampling.repetition_penalty = float(repetition_penalty)
            except ValueError:
                logger.warning(f"Invalid OMLX_REPETITION_PENALTY: {repetition_penalty}")

        # Logging settings
        if log_dir := os.getenv("OMLX_LOG_DIR"):
            self.logging.log_dir = log_dir
        if retention_days := os.getenv("OMLX_LOG_RETENTION_DAYS"):
            try:
                self.logging.retention_days = int(retention_days)
            except ValueError:
                logger.warning(f"Invalid OMLX_LOG_RETENTION_DAYS: {retention_days}")

    def _load_from_file(self, path: Path) -> None:
        """
        Load settings from a JSON file.

        Args:
            path: Path to the settings JSON file.
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Check version for future migrations
            version = data.get("version", SETTINGS_VERSION)
            if version != SETTINGS_VERSION:
                logger.info(
                    f"Settings file version {version} differs from "
                    f"current {SETTINGS_VERSION}, migrating..."
                )

            # Load each section
            if "server" in data:
                self.server = ServerSettingsV2(**data["server"])
            if "model" in data:
                self.model = ModelSettingsV2(**data["model"])
            if "memory" in data:
                self.memory = MemorySettingsV2(**data["memory"])
            if "scheduler" in data:
                self.scheduler = SchedulerSettingsV2(**data["scheduler"])
            if "cache" in data:
                self.cache = CacheSettingsV2(**data["cache"])
            if "auth" in data:
                auth_data = data["auth"]
                # Convert sub_keys list to SubKeyEntryV2 objects
                if "sub_keys" in auth_data:
                    auth_data["sub_keys"] = [
                        SubKeyEntryV2(**sk) for sk in auth_data["sub_keys"]
                    ]
                self.auth = AuthSettingsV2(**auth_data)
            if "mcp" in data:
                self.mcp = MCPSettingsV2(**data["mcp"])
            if "huggingface" in data:
                self.huggingface = HuggingFaceSettingsV2(**data["huggingface"])
            if "sampling" in data:
                self.sampling = SamplingSettingsV2(**data["sampling"])
            if "logging" in data:
                self.logging = LoggingSettingsV2(**data["logging"])
            if "claude_code" in data:
                self.claude_code = ClaudeCodeSettingsV2(**data["claude_code"])
            if "integrations" in data:
                self.integrations = IntegrationSettingsV2(**data["integrations"])
            if "cloud" in data:
                self.cloud = CloudSettingsV2(**data["cloud"])
            if "ui" in data:
                self.ui = UISettingsV2(**data["ui"])

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse settings file {path}: {e}")
        except OSError as e:
            logger.warning(f"Failed to read settings file {path}: {e}")

    def _apply_cli_overrides(self, args: Any) -> None:
        """
        Apply CLI argument overrides.

        Args:
            args: Argparse namespace with CLI arguments.
        """
        # Server settings
        if hasattr(args, "host") and args.host is not None:
            self.server.host = args.host
        if hasattr(args, "port") and args.port is not None:
            self.server.port = args.port
        if hasattr(args, "log_level") and args.log_level is not None:
            self.server.log_level = args.log_level

        # Model settings
        if hasattr(args, "model_dir") and args.model_dir is not None:
            dirs = [d.strip() for d in args.model_dir.split(",") if d.strip()]
            self.model.model_dirs = dirs
            self.model.model_dir = dirs[0] if dirs else None
        if hasattr(args, "max_model_memory") and args.max_model_memory is not None:
            self.model.max_model_memory = args.max_model_memory

        # Memory enforcement settings
        if (
            hasattr(args, "max_process_memory")
            and args.max_process_memory is not None
        ):
            self.memory.max_process_memory = args.max_process_memory

        # Scheduler settings
        if hasattr(args, "max_num_seqs") and args.max_num_seqs is not None:
            self.scheduler.max_num_seqs = args.max_num_seqs
        if (
            hasattr(args, "completion_batch_size")
            and args.completion_batch_size is not None
        ):
            self.scheduler.completion_batch_size = args.completion_batch_size

        # Cache settings
        if hasattr(args, "cache_enabled") and args.cache_enabled is not None:
            self.cache.enabled = args.cache_enabled
        if hasattr(args, "ssd_cache_dir") and args.ssd_cache_dir is not None:
            self.cache.ssd_cache_dir = args.ssd_cache_dir
        if hasattr(args, "ssd_cache_max_size") and args.ssd_cache_max_size is not None:
            self.cache.ssd_cache_max_size = args.ssd_cache_max_size
        if (
            hasattr(args, "initial_cache_blocks")
            and args.initial_cache_blocks is not None
        ):
            self.cache.initial_cache_blocks = args.initial_cache_blocks

        # Auth settings
        if hasattr(args, "api_key") and args.api_key is not None:
            self.auth.api_key = args.api_key

        # MCP settings
        if hasattr(args, "mcp_config") and args.mcp_config is not None:
            self.mcp.config_path = args.mcp_config

        # HuggingFace settings
        if hasattr(args, "hf_endpoint") and args.hf_endpoint is not None:
            self.huggingface.endpoint = args.hf_endpoint

    def save(self) -> None:
        """Save current settings to the settings file (v1 JSON format)."""
        self.ensure_directories()

        settings_file = self.base_path / "settings.json"
        data = {
            "version": self.version,
            "server": self.server.model_dump(),
            "model": {
                "model_dirs": self.model.model_dirs,
                "model_dir": self.model.model_dir,
                "max_model_memory": self.model.max_model_memory,
            },
            "memory": self.memory.model_dump(),
            "scheduler": self.scheduler.model_dump(),
            "cache": self.cache.model_dump(),
            "auth": {
                "api_key": self.auth.api_key,
                "secret_key": self.auth.secret_key,
                "skip_api_key_verification": self.auth.skip_api_key_verification,
                "sub_keys": [sk.model_dump() for sk in self.auth.sub_keys],
            },
            "mcp": self.mcp.model_dump(),
            "huggingface": self.huggingface.model_dump(),
            "sampling": self.sampling.model_dump(),
            "logging": self.logging.model_dump(),
            "claude_code": self.claude_code.model_dump(),
            "integrations": self.integrations.model_dump(),
            "cloud": self.cloud.model_dump(),
            "ui": self.ui.model_dump(),
        }

        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved settings to {settings_file}")
        except OSError as e:
            logger.error(f"Failed to save settings to {settings_file}: {e}")
            raise

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_path,
            *self.model.get_model_dirs(self.base_path),
            self.cache.get_ssd_cache_dir(self.base_path),
            self.logging.get_log_dir(self.base_path),
        ]

        for directory in directories:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {directory}")
                except OSError as e:
                    logger.error(f"Failed to create directory {directory}: {e}")
                    raise

    def validate(self) -> list[str]:
        """
        Validate all settings.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Server validation
        if not 1 <= self.server.port <= 65535:
            errors.append(f"Invalid port: {self.server.port} (must be 1-65535)")

        valid_log_levels = {"trace", "debug", "info", "warning", "error", "critical"}
        if self.server.log_level.lower() not in valid_log_levels:
            errors.append(
                f"Invalid log_level: {self.server.log_level} "
                f"(must be one of {valid_log_levels})"
            )

        # Model validation
        try:
            self.model.get_max_model_memory_bytes()
        except ValueError as e:
            errors.append(f"Invalid max_model_memory: {e}")

        # Memory enforcement validation
        try:
            self.memory.get_max_process_memory_bytes()
        except ValueError as e:
            errors.append(f"Invalid max_process_memory: {e}")

        # Scheduler validation
        if self.scheduler.max_num_seqs <= 0:
            errors.append(
                f"Invalid max_num_seqs: {self.scheduler.max_num_seqs} (must be > 0)"
            )
        if self.scheduler.completion_batch_size <= 0:
            errors.append(
                f"Invalid completion_batch_size: "
                f"{self.scheduler.completion_batch_size} (must be > 0)"
            )

        # Cache validation
        try:
            self.cache.get_ssd_cache_max_size_bytes(self.base_path)
        except ValueError as e:
            errors.append(f"Invalid ssd_cache_max_size: {e}")

        try:
            self.cache.get_hot_cache_max_size_bytes()
        except ValueError as e:
            errors.append(f"Invalid hot_cache_max_size: {e}")

        # HuggingFace validation (endpoint validation is done in field_validator)

        return errors

    def to_scheduler_config(self) -> SchedulerConfig:
        """
        Convert settings to SchedulerConfig for engine initialization.

        Returns:
            SchedulerConfig instance with values from settings.
        """
        from .scheduler import SchedulerConfig

        return SchedulerConfig(
            max_num_seqs=self.scheduler.max_num_seqs,
            completion_batch_size=self.scheduler.completion_batch_size,
            initial_cache_blocks=self.cache.initial_cache_blocks,
            kvtc_enabled=self.cache.kvtc_enabled,
            kvtc_energy=self.cache.kvtc_energy,
            kvtc_bits=self.cache.kvtc_bits,
            kvtc_group_size=self.cache.kvtc_group_size,
            kvtc_adaptive=self.cache.kvtc_adaptive,
            kvtc_threshold=self.cache.kvtc_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert all settings to a dictionary (v1 format)."""
        return {
            "version": self.version,
            "base_path": str(self.base_path),
            "server": self.server.model_dump(),
            "model": {
                "model_dirs": self.model.model_dirs,
                "model_dir": self.model.model_dir,
                "max_model_memory": self.model.max_model_memory,
            },
            "memory": self.memory.model_dump(),
            "scheduler": self.scheduler.model_dump(),
            "cache": self.cache.model_dump(),
            "auth": {
                "api_key": self.auth.api_key,
                "secret_key": self.auth.secret_key,
                "skip_api_key_verification": self.auth.skip_api_key_verification,
                "sub_keys": [sk.model_dump() for sk in self.auth.sub_keys],
            },
            "mcp": self.mcp.model_dump(),
            "huggingface": self.huggingface.model_dump(),
            "sampling": self.sampling.model_dump(),
            "logging": self.logging.model_dump(),
            "claude_code": self.claude_code.model_dump(),
            "integrations": self.integrations.model_dump(),
            "cloud": self.cloud.model_dump(),
            "ui": self.ui.model_dump(),
        }


# Global singleton instance
_global_settings: Optional[GlobalSettingsV2] = None


def get_settings() -> GlobalSettingsV2:
    """
    Get the global settings instance.

    Returns:
        The global GlobalSettingsV2 instance.

    Raises:
        RuntimeError: If settings have not been initialized.
    """
    global _global_settings
    if _global_settings is None:
        raise RuntimeError(
            "Settings not initialized. Call init_settings() first."
        )
    return _global_settings


def init_settings(
    base_path: Optional[str | Path] = None,
    cli_args: Optional[Any] = None,
) -> GlobalSettingsV2:
    """
    Initialize global settings (call once at startup).

    Args:
        base_path: Base directory for oMLX (default: ~/.omlx).
        cli_args: Argparse namespace with CLI arguments.

    Returns:
        The initialized GlobalSettingsV2 instance.
    """
    global _global_settings
    _global_settings = GlobalSettingsV2.load(base_path=base_path, cli_args=cli_args)
    logger.info(f"Initialized settings with base_path: {_global_settings.base_path}")
    return _global_settings


def reset_settings() -> None:
    """
    Reset global settings (primarily for testing).

    This clears the global singleton, allowing init_settings to be called again.
    """
    global _global_settings
    _global_settings = None
