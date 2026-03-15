# SPDX-License-Identifier: Apache-2.0
"""
Settings proxy — re-exports everything from settings_v2.

This file keeps all existing ``from omlx.settings import ...`` statements
working while the canonical implementation lives in ``settings_v2.py``
(Pydantic v2 BaseSettings).

Usage (unchanged):
    from omlx.settings import init_settings, get_settings
    settings = get_settings()
"""

# Re-export public API (functions) -----------------------------------------------
from .settings_v2 import (
    SETTINGS_VERSION,
    DEFAULT_BASE_PATH,
    get_system_memory,
    _adaptive_system_reserve,
    get_ssd_capacity,
    parse_bool,
    get_settings,
    init_settings,
    reset_settings,
)

# Re-export classes with v1-compatible names ------------------------------------
from .settings_v2 import GlobalSettingsV2 as GlobalSettings
from .settings_v2 import SubKeyEntryV2 as SubKeyEntry
from .settings_v2 import ServerSettingsV2 as ServerSettings
from .settings_v2 import ModelSettingsV2 as ModelSettings
from .settings_v2 import SchedulerSettingsV2 as SchedulerSettings
from .settings_v2 import CacheSettingsV2 as CacheSettings
from .settings_v2 import MemorySettingsV2 as MemorySettings
from .settings_v2 import AuthSettingsV2 as AuthSettings
from .settings_v2 import MCPSettingsV2 as MCPSettings
from .settings_v2 import HuggingFaceSettingsV2 as HuggingFaceSettings
from .settings_v2 import SamplingSettingsV2 as SamplingSettings
from .settings_v2 import LoggingSettingsV2 as LoggingSettings
from .settings_v2 import UISettingsV2 as UISettings
from .settings_v2 import ClaudeCodeSettingsV2 as ClaudeCodeSettings
from .settings_v2 import IntegrationSettingsV2 as IntegrationSettings

# Keep ``from omlx.settings import *`` working
__all__ = [
    "SETTINGS_VERSION",
    "DEFAULT_BASE_PATH",
    "get_system_memory",
    "_adaptive_system_reserve",
    "get_ssd_capacity",
    "parse_bool",
    "get_settings",
    "init_settings",
    "reset_settings",
    "GlobalSettings",
    "SubKeyEntry",
    "ServerSettings",
    "ModelSettings",
    "SchedulerSettings",
    "CacheSettings",
    "MemorySettings",
    "AuthSettings",
    "MCPSettings",
    "HuggingFaceSettings",
    "SamplingSettings",
    "LoggingSettings",
    "UISettings",
    "ClaudeCodeSettings",
    "IntegrationSettings",
]
