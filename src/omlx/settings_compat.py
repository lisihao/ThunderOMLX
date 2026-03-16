# SPDX-License-Identifier: Apache-2.0
"""
Backward compatibility layer for settings v1 to v2 migration.

This module handles:
- Converting v1 settings.json to v2 GlobalSettingsV2
- Converting v2 GlobalSettingsV2 back to v1 format
- Field renaming (model_dir → model_dirs)
- Type conversions (string booleans → bool)
"""

from typing import Any

from .settings_v2 import GlobalSettingsV2, parse_bool


def convert_v1_to_v2(v1_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert v1 settings.json format to v2 format.

    Handles:
    - model_dir → model_dirs migration
    - Boolean string parsing
    - Preserves all other fields as-is

    Args:
        v1_dict: v1 settings dictionary loaded from settings.json

    Returns:
        v2-compatible dictionary that can be used to initialize GlobalSettingsV2
    """
    v2_dict = {}

    # Copy version if present
    if "version" in v1_dict:
        v2_dict["version"] = v1_dict["version"]

    # Process each section
    if "server" in v1_dict:
        v2_dict["server"] = _convert_server_settings(v1_dict["server"])

    if "model" in v1_dict:
        v2_dict["model"] = _convert_model_settings(v1_dict["model"])

    if "memory" in v1_dict:
        v2_dict["memory"] = v1_dict["memory"].copy()

    if "scheduler" in v1_dict:
        v2_dict["scheduler"] = v1_dict["scheduler"].copy()

    if "cache" in v1_dict:
        v2_dict["cache"] = _convert_cache_settings(v1_dict["cache"])

    if "auth" in v1_dict:
        v2_dict["auth"] = v1_dict["auth"].copy()

    if "mcp" in v1_dict:
        v2_dict["mcp"] = v1_dict["mcp"].copy()

    if "huggingface" in v1_dict:
        v2_dict["huggingface"] = v1_dict["huggingface"].copy()

    if "sampling" in v1_dict:
        v2_dict["sampling"] = v1_dict["sampling"].copy()

    if "logging" in v1_dict:
        v2_dict["logging"] = v1_dict["logging"].copy()

    if "claude_code" in v1_dict:
        v2_dict["claude_code"] = v1_dict["claude_code"].copy()

    if "integrations" in v1_dict:
        v2_dict["integrations"] = v1_dict["integrations"].copy()

    if "ui" in v1_dict:
        v2_dict["ui"] = v1_dict["ui"].copy()

    return v2_dict


def _convert_server_settings(server_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert server settings from v1 to v2."""
    result = server_dict.copy()

    # Handle boolean string fields if present
    if "cors_origins" in result and isinstance(result["cors_origins"], str):
        result["cors_origins"] = [o.strip() for o in result["cors_origins"].split(",")]

    return result


def _convert_model_settings(model_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert model settings from v1 to v2."""
    result = model_dict.copy()

    # Migrate model_dir → model_dirs if needed
    if "model_dir" in result and "model_dirs" not in result:
        # If model_dir is set but model_dirs is not, populate model_dirs
        model_dir = result["model_dir"]
        if model_dir:
            result["model_dirs"] = [model_dir]
        else:
            result["model_dirs"] = []

    # Ensure model_dirs is a list
    if "model_dirs" in result and isinstance(result["model_dirs"], str):
        result["model_dirs"] = [d.strip() for d in result["model_dirs"].split(",")]

    return result


def _convert_cache_settings(cache_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert cache settings from v1 to v2."""
    result = cache_dict.copy()

    # Handle boolean string for enabled field
    if "enabled" in result and isinstance(result["enabled"], str):
        result["enabled"] = parse_bool(result["enabled"])

    return result


def convert_v2_to_v1(v2: GlobalSettingsV2) -> dict[str, Any]:
    """
    Convert v2 GlobalSettingsV2 to v1 settings.json format.

    Reverses the migration:
    - model_dirs → model_dir (uses first entry)
    - All other fields as-is

    Args:
        v2: GlobalSettingsV2 instance

    Returns:
        v1-compatible dictionary suitable for JSON serialization
    """
    model_dirs = v2.model.model_dirs
    model_dir = model_dirs[0] if model_dirs else v2.model.model_dir

    return {
        "version": v2.version,
        "server": v2.server.model_dump(),
        "model": {
            "model_dirs": v2.model.model_dirs,
            "model_dir": model_dir,
            "max_model_memory": v2.model.max_model_memory,
        },
        "memory": v2.memory.model_dump(),
        "scheduler": v2.scheduler.model_dump(),
        "cache": v2.cache.model_dump(),
        "auth": {
            "api_key": v2.auth.api_key,
            "secret_key": v2.auth.secret_key,
            "skip_api_key_verification": v2.auth.skip_api_key_verification,
            "sub_keys": [sk.model_dump() for sk in v2.auth.sub_keys],
        },
        "mcp": v2.mcp.model_dump(),
        "huggingface": v2.huggingface.model_dump(),
        "sampling": v2.sampling.model_dump(),
        "logging": v2.logging.model_dump(),
        "claude_code": v2.claude_code.model_dump(),
        "integrations": v2.integrations.model_dump(),
        "ui": v2.ui.model_dump(),
    }
