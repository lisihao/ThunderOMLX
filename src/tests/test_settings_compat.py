# SPDX-License-Identifier: Apache-2.0
"""
Test suite for settings compatibility layer.

Tests the conversion between v1 and v2 settings formats.
"""

import json
import tempfile
from pathlib import Path

import pytest

from omlx.settings_v2 import GlobalSettingsV2
from omlx.settings_compat import convert_v1_to_v2, convert_v2_to_v1


class TestConvertV1toV2:
    """Test v1 to v2 conversion."""

    def test_convert_empty_dict(self):
        """Test converting empty dictionary."""
        result = convert_v1_to_v2({})
        assert isinstance(result, dict)

    def test_convert_version(self):
        """Test version is preserved."""
        result = convert_v1_to_v2({"version": "1.0"})
        assert result["version"] == "1.0"

    def test_convert_server_settings(self):
        """Test server settings conversion."""
        v1_data = {
            "server": {
                "host": "0.0.0.0",
                "port": 9000,
                "log_level": "debug",
            }
        }
        result = convert_v1_to_v2(v1_data)
        assert result["server"]["host"] == "0.0.0.0"
        assert result["server"]["port"] == 9000
        assert result["server"]["log_level"] == "debug"

    def test_convert_model_dir_to_dirs(self):
        """Test migration of model_dir to model_dirs."""
        v1_data = {
            "model": {
                "model_dir": "/path/to/models",
                "max_model_memory": "16GB",
            }
        }
        result = convert_v1_to_v2(v1_data)
        assert result["model"]["model_dirs"] == ["/path/to/models"]
        assert result["model"]["model_dir"] == "/path/to/models"

    def test_convert_model_dirs_preserved(self):
        """Test that model_dirs is preserved if already set."""
        v1_data = {
            "model": {
                "model_dir": "/old/path",
                "model_dirs": ["/new/path1", "/new/path2"],
            }
        }
        result = convert_v1_to_v2(v1_data)
        # model_dirs should be preserved, not overwritten
        assert result["model"]["model_dirs"] == ["/new/path1", "/new/path2"]

    def test_convert_cache_boolean_string(self):
        """Test conversion of boolean string in cache settings."""
        v1_data = {
            "cache": {
                "enabled": "true",
                "ssd_cache_max_size": "50GB",
            }
        }
        result = convert_v1_to_v2(v1_data)
        # Should preserve the string for Pydantic to parse
        assert result["cache"]["enabled"] in (True, "true")

    def test_convert_all_sections(self):
        """Test that all sections are converted."""
        v1_data = {
            "version": "1.0",
            "server": {"port": 8000},
            "model": {"model_dir": "/models"},
            "memory": {"max_process_memory": "auto"},
            "scheduler": {"max_num_seqs": 16},
            "cache": {"enabled": True},
            "auth": {"api_key": "test-key"},
            "mcp": {"config_path": "/path/to/mcp"},
            "huggingface": {"endpoint": "https://huggingface.co"},
            "sampling": {"temperature": 1.5},
            "logging": {"retention_days": 14},
            "claude_code": {"mode": "local"},
            "integrations": {"codex_model": "test"},
            "ui": {"language": "en"},
        }
        result = convert_v1_to_v2(v1_data)

        # Verify all sections are present
        expected_sections = [
            "version",
            "server",
            "model",
            "memory",
            "scheduler",
            "cache",
            "auth",
            "mcp",
            "huggingface",
            "sampling",
            "logging",
            "claude_code",
            "integrations",
            "ui",
        ]
        for section in expected_sections:
            assert section in result

    def test_convert_to_valid_globalettings(self):
        """Test that converted data can instantiate GlobalSettingsV2."""
        v1_data = {
            "version": "1.0",
            "server": {"port": 8888},
            "model": {"model_dir": "/custom/models"},
        }
        converted = convert_v1_to_v2(v1_data)
        # Should not raise any validation errors
        settings = GlobalSettingsV2(**converted)
        assert settings.server.port == 8888


class TestConvertV2toV1:
    """Test v2 to v1 conversion."""

    def test_convert_basic_settings(self):
        """Test basic v2 to v1 conversion."""
        settings = GlobalSettingsV2()
        settings.server.port = 9999
        settings.cache.enabled = False

        result = convert_v2_to_v1(settings)
        assert result["server"]["port"] == 9999
        assert result["cache"]["enabled"] is False

    def test_convert_model_dirs_to_dir(self):
        """Test conversion of model_dirs to model_dir."""
        settings = GlobalSettingsV2()
        settings.model.model_dirs = ["/path1", "/path2"]

        result = convert_v2_to_v1(settings)
        # Should use first path as model_dir
        assert result["model"]["model_dir"] == "/path1"
        assert result["model"]["model_dirs"] == ["/path1", "/path2"]

    def test_convert_empty_model_dirs(self):
        """Test conversion with empty model_dirs."""
        settings = GlobalSettingsV2()
        settings.model.model_dirs = []
        settings.model.model_dir = "/fallback/path"

        result = convert_v2_to_v1(settings)
        assert result["model"]["model_dir"] == "/fallback/path"

    def test_convert_version_preserved(self):
        """Test that version is preserved."""
        settings = GlobalSettingsV2()
        result = convert_v2_to_v1(settings)
        assert result["version"] == "1.0"

    def test_convert_all_sections_present(self):
        """Test that all sections are present in conversion."""
        settings = GlobalSettingsV2()
        result = convert_v2_to_v1(settings)

        expected_sections = [
            "version",
            "server",
            "model",
            "memory",
            "scheduler",
            "cache",
            "auth",
            "mcp",
            "huggingface",
            "sampling",
            "logging",
            "claude_code",
            "integrations",
            "ui",
        ]
        for section in expected_sections:
            assert section in result

    def test_convert_auth_sub_keys(self):
        """Test conversion of auth sub_keys."""
        settings = GlobalSettingsV2()
        from omlx.settings_v2 import SubKeyEntryV2

        settings.auth.sub_keys = [
            SubKeyEntryV2(key="test-key-1", name="Test Key 1"),
            SubKeyEntryV2(key="test-key-2", name="Test Key 2"),
        ]

        result = convert_v2_to_v1(settings)
        assert len(result["auth"]["sub_keys"]) == 2
        assert result["auth"]["sub_keys"][0]["key"] == "test-key-1"


class TestRoundTripConversion:
    """Test round-trip conversion (v1 -> v2 -> v1)."""

    def test_roundtrip_basic(self):
        """Test basic round-trip conversion."""
        v1_original = {
            "version": "1.0",
            "server": {"port": 8888},
            "model": {"model_dir": "/models"},
            "cache": {"enabled": True},
        }

        # v1 -> v2
        v2_data = convert_v1_to_v2(v1_original)
        settings = GlobalSettingsV2(**v2_data)

        # v2 -> v1
        v1_converted = convert_v2_to_v1(settings)

        # Verify key values are preserved
        assert v1_converted["server"]["port"] == v1_original["server"]["port"]
        assert v1_converted["cache"]["enabled"] == v1_original["cache"]["enabled"]

    def test_roundtrip_complex(self):
        """Test complex round-trip with all sections."""
        v1_original = {
            "version": "1.0",
            "server": {
                "host": "192.168.1.1",
                "port": 7000,
                "log_level": "debug",
            },
            "model": {
                "model_dir": "/custom/models",
                "max_model_memory": "32GB",
            },
            "memory": {"max_process_memory": "80%"},
            "scheduler": {"max_num_seqs": 32},
            "cache": {
                "enabled": True,
                "ssd_cache_max_size": "100GB",
            },
            "auth": {
                "api_key": "test-key",
                "skip_api_key_verification": False,
            },
            "sampling": {
                "temperature": 1.5,
                "top_p": 0.9,
            },
        }

        # v1 -> v2
        v2_data = convert_v1_to_v2(v1_original)
        settings = GlobalSettingsV2(**v2_data)

        # v2 -> v1
        v1_converted = convert_v2_to_v1(settings)

        # Verify all values
        assert v1_converted["server"]["host"] == "192.168.1.1"
        assert v1_converted["server"]["port"] == 7000
        assert v1_converted["model"]["max_model_memory"] == "32GB"
        assert v1_converted["memory"]["max_process_memory"] == "80%"
        assert v1_converted["scheduler"]["max_num_seqs"] == 32
        assert v1_converted["cache"]["enabled"] is True
        assert v1_converted["sampling"]["temperature"] == 1.5


class TestCompatibilityEdgeCases:
    """Test edge cases in compatibility conversion."""

    def test_null_auth_fields(self):
        """Test handling of null auth fields."""
        v1_data = {
            "auth": {
                "api_key": None,
                "secret_key": None,
            }
        }
        result = convert_v1_to_v2(v1_data)
        assert result["auth"]["api_key"] is None
        assert result["auth"]["secret_key"] is None

    def test_empty_cors_origins(self):
        """Test handling of empty CORS origins."""
        v1_data = {
            "server": {
                "cors_origins": [],
            }
        }
        result = convert_v1_to_v2(v1_data)
        assert result["server"]["cors_origins"] == []

    def test_missing_optional_fields(self):
        """Test handling of missing optional fields."""
        v1_data = {
            "server": {"port": 8000},
            # Only port, no other fields
        }
        result = convert_v1_to_v2(v1_data)
        # Should not raise, and port should be set
        assert result["server"]["port"] == 8000

    def test_preserve_unknown_fields(self):
        """Test that conversion doesn't break with unexpected fields."""
        v1_data = {
            "server": {"port": 8000},
            "unknown_section": {"unknown_field": "value"},
        }
        # Should not raise
        result = convert_v1_to_v2(v1_data)
        # Unknown sections won't be in result, but known sections should be fine
        assert result["server"]["port"] == 8000


class TestFileRoundTripConversion:
    """Test round-trip through JSON files."""

    def test_load_v1_save_v2_load_again(self):
        """Test loading v1 JSON, saving as v2, and loading again."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings_file = tmpdir_path / "settings.json"

            # Create original v1 format file
            v1_data = {
                "version": "1.0",
                "server": {"port": 8888},
                "model": {"model_dir": str(tmpdir_path / "models")},
            }
            with open(settings_file, "w") as f:
                json.dump(v1_data, f)

            # Load v1 format
            settings1 = GlobalSettingsV2.load(base_path=tmpdir_path)
            assert settings1.server.port == 8888

            # Save as v2 (which uses v1 format)
            settings1.save()

            # Load again
            settings2 = GlobalSettingsV2.load(base_path=tmpdir_path)
            assert settings2.server.port == 8888

    def test_settings_survives_file_cycle(self):
        """Test that settings survive a save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create and configure settings
            settings1 = GlobalSettingsV2(base_path=tmpdir_path)
            settings1.server.port = 9999
            settings1.cache.enabled = False
            settings1.sampling.temperature = 1.5
            settings1.save()

            # Load from file
            settings2 = GlobalSettingsV2.load(base_path=tmpdir_path)

            # Verify all values
            assert settings2.server.port == 9999
            assert settings2.cache.enabled is False
            assert settings2.sampling.temperature == 1.5
