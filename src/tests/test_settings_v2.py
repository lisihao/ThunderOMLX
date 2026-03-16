# SPDX-License-Identifier: Apache-2.0
"""
Test suite for GlobalSettingsV2.

Tests cover:
- Default values
- Field validation
- Boolean parsing
- Size parsing
- Environment variable overrides
- JSON file loading
- CLI argument overrides
- Runtime calculation methods
- Backward compatibility
- Performance
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from omlx.config import parse_size
from omlx.settings_v2 import (
    GlobalSettingsV2,
    ServerSettingsV2,
    ModelSettingsV2,
    SchedulerSettingsV2,
    CacheSettingsV2,
    MemorySettingsV2,
    AuthSettingsV2,
    parse_bool,
    reset_settings,
)


class TestDefaultValues:
    """Test that all 13 configuration sections have correct default values."""

    def test_server_defaults(self):
        """Test server settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.server.host == "127.0.0.1"
        assert settings.server.port == 8000
        assert settings.server.log_level == "info"
        assert settings.server.cors_origins == ["*"]

    def test_model_defaults(self):
        """Test model settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.model.model_dirs == []
        assert settings.model.model_dir is None
        assert settings.model.max_model_memory == "auto"

    def test_scheduler_defaults(self):
        """Test scheduler settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.scheduler.max_num_seqs == 8
        assert settings.scheduler.completion_batch_size == 8

    def test_cache_defaults(self):
        """Test cache settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.cache.enabled is True
        assert settings.cache.ssd_cache_dir is None
        assert settings.cache.ssd_cache_max_size == "auto"
        assert settings.cache.hot_cache_max_size == "0"
        assert settings.cache.initial_cache_blocks == 256

    def test_memory_defaults(self):
        """Test memory settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.memory.max_process_memory == "auto"

    def test_auth_defaults(self):
        """Test auth settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.auth.api_key is None
        assert settings.auth.secret_key is None
        assert settings.auth.skip_api_key_verification is False
        assert settings.auth.sub_keys == []

    def test_mcp_defaults(self):
        """Test MCP settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.mcp.config_path is None

    def test_huggingface_defaults(self):
        """Test HuggingFace settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.huggingface.endpoint == ""

    def test_sampling_defaults(self):
        """Test sampling settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.sampling.max_context_window == 32768
        assert settings.sampling.max_tokens == 32768
        assert settings.sampling.temperature == 1.0
        assert settings.sampling.top_p == 0.95
        assert settings.sampling.top_k == 0
        assert settings.sampling.repetition_penalty == 1.0

    def test_logging_defaults(self):
        """Test logging settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.logging.log_dir is None
        assert settings.logging.retention_days == 7

    def test_ui_defaults(self):
        """Test UI settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.ui.language == "en"

    def test_claude_code_defaults(self):
        """Test Claude Code settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.claude_code.context_scaling_enabled is False
        assert settings.claude_code.target_context_size == 200000
        assert settings.claude_code.mode == "cloud"
        assert settings.claude_code.opus_model is None
        assert settings.claude_code.sonnet_model is None
        assert settings.claude_code.haiku_model is None

    def test_integrations_defaults(self):
        """Test integration settings defaults."""
        settings = GlobalSettingsV2()
        assert settings.integrations.codex_model is None
        assert settings.integrations.opencode_model is None
        assert settings.integrations.openclaw_model is None
        assert settings.integrations.openclaw_tools_profile == "coding"


class TestValidation:
    """Test field validation."""

    def test_invalid_port_too_low(self):
        """Test that port < 1 is rejected."""
        with pytest.raises(ValueError):
            ServerSettingsV2(port=0)

    def test_invalid_port_too_high(self):
        """Test that port > 65535 is rejected."""
        with pytest.raises(ValueError):
            ServerSettingsV2(port=99999)

    def test_valid_port_range(self):
        """Test valid port range."""
        server = ServerSettingsV2(port=8080)
        assert server.port == 8080

    def test_invalid_temperature_too_low(self):
        """Test that temperature < 0 is rejected."""
        with pytest.raises(ValueError):
            from omlx.settings_v2 import SamplingSettingsV2
            SamplingSettingsV2(temperature=-0.1)

    def test_invalid_temperature_too_high(self):
        """Test that temperature > 2.0 is rejected."""
        with pytest.raises(ValueError):
            from omlx.settings_v2 import SamplingSettingsV2
            SamplingSettingsV2(temperature=2.1)

    def test_valid_temperature_range(self):
        """Test valid temperature range."""
        from omlx.settings_v2 import SamplingSettingsV2
        sampling = SamplingSettingsV2(temperature=1.5)
        assert sampling.temperature == 1.5

    def test_invalid_top_p_too_high(self):
        """Test that top_p > 1.0 is rejected."""
        with pytest.raises(ValueError):
            from omlx.settings_v2 import SamplingSettingsV2
            SamplingSettingsV2(top_p=1.1)

    def test_valid_top_p_range(self):
        """Test valid top_p range."""
        from omlx.settings_v2 import SamplingSettingsV2
        sampling = SamplingSettingsV2(top_p=0.9)
        assert sampling.top_p == 0.9

    def test_invalid_max_num_seqs(self):
        """Test that max_num_seqs <= 0 is rejected."""
        with pytest.raises(ValueError):
            SchedulerSettingsV2(max_num_seqs=0)

    def test_valid_max_num_seqs(self):
        """Test valid max_num_seqs."""
        scheduler = SchedulerSettingsV2(max_num_seqs=16)
        assert scheduler.max_num_seqs == 16

    def test_invalid_retention_days(self):
        """Test that retention_days <= 0 is rejected."""
        with pytest.raises(ValueError):
            from omlx.settings_v2 import LoggingSettingsV2
            LoggingSettingsV2(retention_days=0)

    def test_valid_retention_days(self):
        """Test valid retention_days."""
        from omlx.settings_v2 import LoggingSettingsV2
        logging_settings = LoggingSettingsV2(retention_days=30)
        assert logging_settings.retention_days == 30

    def test_invalid_huggingface_endpoint(self):
        """Test that invalid HuggingFace endpoint is rejected."""
        with pytest.raises(ValueError):
            from omlx.settings_v2 import HuggingFaceSettingsV2
            HuggingFaceSettingsV2(endpoint="not-a-url")

    def test_valid_huggingface_endpoint(self):
        """Test valid HuggingFace endpoint."""
        from omlx.settings_v2 import HuggingFaceSettingsV2
        hf = HuggingFaceSettingsV2(endpoint="https://huggingface.co/api")
        assert hf.endpoint == "https://huggingface.co/api"

    def test_invalid_claude_code_mode(self):
        """Test that invalid claude_code mode is rejected."""
        with pytest.raises(ValueError):
            from omlx.settings_v2 import ClaudeCodeSettingsV2
            ClaudeCodeSettingsV2(mode="invalid")

    def test_valid_claude_code_mode(self):
        """Test valid claude_code mode."""
        from omlx.settings_v2 import ClaudeCodeSettingsV2
        cc = ClaudeCodeSettingsV2(mode="local")
        assert cc.mode == "local"


class TestBoolParsing:
    """Test boolean string parsing."""

    def test_parse_bool_true_strings(self):
        """Test parsing various 'true' representations."""
        assert parse_bool("true") is True
        assert parse_bool("True") is True
        assert parse_bool("TRUE") is True
        assert parse_bool("1") is True
        assert parse_bool("yes") is True
        assert parse_bool("Yes") is True
        assert parse_bool("on") is True
        assert parse_bool("ON") is True

    def test_parse_bool_false_strings(self):
        """Test parsing various 'false' representations."""
        assert parse_bool("false") is False
        assert parse_bool("False") is False
        assert parse_bool("0") is False
        assert parse_bool("no") is False
        assert parse_bool("off") is False

    def test_parse_bool_actual_bools(self):
        """Test parsing actual boolean values."""
        assert parse_bool(True) is True
        assert parse_bool(False) is False

    def test_parse_bool_other_types(self):
        """Test parsing other types (cast to bool)."""
        assert parse_bool(1) is True
        assert parse_bool(0) is False
        assert parse_bool([1]) is True
        assert parse_bool([]) is False


class TestSizeParsing:
    """Test size string parsing."""

    def test_parse_bytes(self):
        """Test parsing byte sizes."""
        assert parse_size("1024") == 1024
        assert parse_size("512") == 512

    def test_parse_kilobytes(self):
        """Test parsing KB sizes."""
        assert parse_size("1KB") == 1024
        assert parse_size("1Kb") == 1024

    def test_parse_megabytes(self):
        """Test parsing MB sizes."""
        assert parse_size("1MB") == 1024 * 1024
        assert parse_size("1Mb") == 1024 * 1024

    def test_parse_gigabytes(self):
        """Test parsing GB sizes."""
        assert parse_size("1GB") == 1024**3
        assert parse_size("1Gb") == 1024**3
        assert parse_size("32GB") == 32 * 1024**3

    def test_parse_terabytes(self):
        """Test parsing TB sizes."""
        assert parse_size("1TB") == 1024**4
        assert parse_size("1Tb") == 1024**4

    def test_parse_case_insensitive(self):
        """Test that size parsing is case-insensitive."""
        assert parse_size("1gb") == 1024**3
        assert parse_size("1Gb") == 1024**3


class TestEnvVarOverrides:
    """Test environment variable overrides."""

    def test_env_server_port(self):
        """Test OMLX_SERVER__PORT environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            os.environ["OMLX_SERVER__PORT"] = "9000"
            try:
                settings = GlobalSettingsV2.load(base_path=tmpdir_path)
                assert settings.server.port == 9000
            finally:
                del os.environ["OMLX_SERVER__PORT"]

    def test_env_model_max_memory(self):
        """Test OMLX_MODEL__MAX_MODEL_MEMORY environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            os.environ["OMLX_MAX_MODEL_MEMORY"] = "16GB"
            try:
                settings = GlobalSettingsV2.load(base_path=tmpdir_path)
                assert settings.model.max_model_memory == "16GB"
            finally:
                del os.environ["OMLX_MAX_MODEL_MEMORY"]

    def test_env_cache_enabled(self):
        """Test OMLX_CACHE_ENABLED environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            os.environ["OMLX_CACHE_ENABLED"] = "false"
            try:
                settings = GlobalSettingsV2.load(base_path=tmpdir_path)
                assert settings.cache.enabled is False
            finally:
                del os.environ["OMLX_CACHE_ENABLED"]

    def test_env_temperature(self):
        """Test OMLX_TEMPERATURE environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            os.environ["OMLX_TEMPERATURE"] = "1.5"
            try:
                settings = GlobalSettingsV2.load(base_path=tmpdir_path)
                assert settings.sampling.temperature == 1.5
            finally:
                del os.environ["OMLX_TEMPERATURE"]

    def test_env_model_dir(self):
        """Test OMLX_MODEL_DIR environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            os.environ["OMLX_MODEL_DIR"] = "/path/to/models"
            try:
                settings = GlobalSettingsV2.load(base_path=tmpdir_path)
                assert "/path/to/models" in settings.model.model_dirs
            finally:
                del os.environ["OMLX_MODEL_DIR"]


class TestJsonLoading:
    """Test loading settings from JSON file."""

    def test_load_from_json_file(self):
        """Test loading settings from settings.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings_file = tmpdir_path / "settings.json"

            # Create settings file
            data = {
                "version": "1.0",
                "server": {"host": "0.0.0.0", "port": 9000},
                "model": {"model_dirs": ["/custom/models"]},
                "sampling": {"temperature": 1.5},
                "cache": {"enabled": False},
            }
            with open(settings_file, "w") as f:
                json.dump(data, f)

            # Load settings
            settings = GlobalSettingsV2.load(base_path=tmpdir_path)
            assert settings.server.host == "0.0.0.0"
            assert settings.server.port == 9000
            assert "/custom/models" in settings.model.model_dirs
            assert settings.sampling.temperature == 1.5
            assert settings.cache.enabled is False

    def test_load_v1_format(self):
        """Test loading v1 format settings.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings_file = tmpdir_path / "settings.json"

            # Create v1 format settings file
            data = {
                "version": "1.0",
                "server": {"port": 8888},
                "model": {"model_dir": "/old/model/dir"},
            }
            with open(settings_file, "w") as f:
                json.dump(data, f)

            # Load settings - should migrate model_dir to model_dirs
            settings = GlobalSettingsV2.load(base_path=tmpdir_path)
            assert settings.server.port == 8888
            assert settings.model.model_dir == "/old/model/dir"
            assert "/old/model/dir" in settings.model.model_dirs

    def test_load_partial_json_file(self):
        """Test loading partial settings file (only override what's present)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings_file = tmpdir_path / "settings.json"

            # Create partial settings file
            data = {
                "version": "1.0",
                "server": {"port": 7000},
            }
            with open(settings_file, "w") as f:
                json.dump(data, f)

            # Load settings
            settings = GlobalSettingsV2.load(base_path=tmpdir_path)
            # Override from file
            assert settings.server.port == 7000
            # Use defaults for everything else
            assert settings.server.host == "127.0.0.1"
            assert settings.cache.enabled is True


class TestCliOverrides:
    """Test CLI argument overrides."""

    def test_cli_port_override(self):
        """Test CLI argument overrides settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create settings file with port 8000
            settings_file = tmpdir_path / "settings.json"
            data = {"version": "1.0", "server": {"port": 8000}}
            with open(settings_file, "w") as f:
                json.dump(data, f)

            # Create CLI args mock
            class Args:
                port = 9999
                host = None
                model_dir = None
                max_model_memory = None
                max_process_memory = None
                max_num_seqs = None
                completion_batch_size = None
                cache_enabled = None
                ssd_cache_dir = None
                ssd_cache_max_size = None
                initial_cache_blocks = None
                api_key = None
                mcp_config = None
                hf_endpoint = None
                log_level = None

            # Load with CLI override
            settings = GlobalSettingsV2.load(base_path=tmpdir_path, cli_args=Args())
            assert settings.server.port == 9999  # CLI overrides file

    def test_cli_model_dir_override(self):
        """Test CLI model_dir override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            class Args:
                model_dir = "/cli/models"
                port = None
                host = None
                max_model_memory = None
                max_process_memory = None
                max_num_seqs = None
                completion_batch_size = None
                cache_enabled = None
                ssd_cache_dir = None
                ssd_cache_max_size = None
                initial_cache_blocks = None
                api_key = None
                mcp_config = None
                hf_endpoint = None
                log_level = None

            settings = GlobalSettingsV2.load(base_path=tmpdir_path, cli_args=Args())
            assert "/cli/models" in settings.model.model_dirs


class TestRuntimeMethods:
    """Test runtime calculation methods."""

    def test_get_max_model_memory_auto(self):
        """Test get_max_model_memory_bytes with 'auto'."""
        settings = GlobalSettingsV2()
        settings.model.max_model_memory = "auto"
        memory_bytes = settings.model.get_max_model_memory_bytes()
        # Should be a reasonable number (not None, > 0)
        assert memory_bytes is not None
        assert memory_bytes > 0

    def test_get_max_model_memory_disabled(self):
        """Test get_max_model_memory_bytes with 'disabled'."""
        settings = GlobalSettingsV2()
        settings.model.max_model_memory = "disabled"
        memory_bytes = settings.model.get_max_model_memory_bytes()
        assert memory_bytes is None

    def test_get_max_model_memory_absolute(self):
        """Test get_max_model_memory_bytes with absolute value."""
        settings = GlobalSettingsV2()
        settings.model.max_model_memory = "16GB"
        memory_bytes = settings.model.get_max_model_memory_bytes()
        assert memory_bytes == 16 * 1024**3

    def test_get_model_dirs(self):
        """Test get_model_dirs method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.model.model_dirs = ["/path1", "/path2"]
            dirs = settings.model.get_model_dirs(tmpdir_path)
            assert len(dirs) == 2
            assert all(isinstance(d, Path) for d in dirs)

    def test_get_model_dir_primary(self):
        """Test get_model_dir returns first directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.model.model_dirs = ["/path1", "/path2"]
            primary_dir = settings.model.get_model_dir(tmpdir_path)
            assert str(primary_dir).endswith("path1")

    def test_get_ssd_cache_dir(self):
        """Test get_ssd_cache_dir method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.cache.ssd_cache_dir = "/custom/cache"
            cache_dir = settings.cache.get_ssd_cache_dir(tmpdir_path)
            assert str(cache_dir).endswith("custom/cache")

    def test_get_ssd_cache_dir_default(self):
        """Test get_ssd_cache_dir returns default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            cache_dir = settings.cache.get_ssd_cache_dir(tmpdir_path)
            assert str(cache_dir).endswith("cache")

    def test_get_ssd_cache_max_size_auto(self):
        """Test get_ssd_cache_max_size_bytes with 'auto'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.cache.ssd_cache_max_size = "auto"
            size = settings.cache.get_ssd_cache_max_size_bytes(tmpdir_path)
            assert size > 0

    def test_get_hot_cache_max_size(self):
        """Test get_hot_cache_max_size_bytes method."""
        settings = GlobalSettingsV2()
        settings.cache.hot_cache_max_size = "8GB"
        size = settings.cache.get_hot_cache_max_size_bytes()
        assert size == 8 * 1024**3

    def test_get_log_dir(self):
        """Test get_log_dir method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.logging.log_dir = "/custom/logs"
            log_dir = settings.logging.get_log_dir(tmpdir_path)
            assert str(log_dir).endswith("custom/logs")

    def test_get_log_dir_default(self):
        """Test get_log_dir returns default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            log_dir = settings.logging.get_log_dir(tmpdir_path)
            assert str(log_dir).endswith("logs")

    def test_get_max_process_memory_auto(self):
        """Test get_max_process_memory_bytes with 'auto'."""
        settings = GlobalSettingsV2()
        settings.memory.max_process_memory = "auto"
        memory_bytes = settings.memory.get_max_process_memory_bytes()
        assert memory_bytes is not None
        assert memory_bytes > 0

    def test_get_max_process_memory_disabled(self):
        """Test get_max_process_memory_bytes with 'disabled'."""
        settings = GlobalSettingsV2()
        settings.memory.max_process_memory = "disabled"
        memory_bytes = settings.memory.get_max_process_memory_bytes()
        assert memory_bytes is None

    def test_get_max_process_memory_percentage(self):
        """Test get_max_process_memory_bytes with percentage."""
        settings = GlobalSettingsV2()
        settings.memory.max_process_memory = "80%"
        memory_bytes = settings.memory.get_max_process_memory_bytes()
        assert memory_bytes is not None
        assert memory_bytes > 0


class TestBackwardCompat:
    """Test backward compatibility with v1 format."""

    def test_model_dir_migration(self):
        """Test automatic migration from model_dir to model_dirs."""
        # Create model settings with model_dir set
        model_settings = ModelSettingsV2(model_dir="/old/path")
        # model_validator should have migrated to model_dirs
        assert "/old/path" in model_settings.model_dirs

    def test_v1_json_load_and_save(self):
        """Test that v1 format can be loaded and saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings_file = tmpdir_path / "settings.json"

            # Use a model dir within the temp directory
            temp_model_dir = tmpdir_path / "models"
            temp_model_dir.mkdir(parents=True, exist_ok=True)

            # Create v1 format settings file
            v1_data = {
                "version": "1.0",
                "server": {"port": 8888},
                "model": {"model_dir": str(temp_model_dir)},
                "cache": {"enabled": True},
            }
            with open(settings_file, "w") as f:
                json.dump(v1_data, f)

            # Load and save
            settings = GlobalSettingsV2.load(base_path=tmpdir_path)
            settings.save()

            # Load the saved file and verify structure
            with open(settings_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data["version"] == "1.0"
            assert saved_data["server"]["port"] == 8888
            assert saved_data["model"]["model_dir"] is not None


class TestValidationMethod:
    """Test the validate() method."""

    def test_validate_success(self):
        """Test that valid settings pass validation."""
        settings = GlobalSettingsV2()
        errors = settings.validate()
        assert errors == []

    def test_validate_invalid_port(self):
        """Test validation catches invalid port."""
        settings = GlobalSettingsV2()
        settings.server.port = 99999
        errors = settings.validate()
        assert any("port" in e.lower() for e in errors)

    def test_validate_invalid_log_level(self):
        """Test validation catches invalid log level."""
        settings = GlobalSettingsV2()
        settings.server.log_level = "invalid_level"
        errors = settings.validate()
        assert any("log_level" in e.lower() for e in errors)


class TestSaveMethod:
    """Test the save() method."""

    def test_save_creates_file(self):
        """Test that save() creates settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.save()

            settings_file = tmpdir_path / "settings.json"
            assert settings_file.exists()

    def test_save_preserves_values(self):
        """Test that save() preserves all values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.server.port = 9999
            settings.cache.enabled = False
            settings.sampling.temperature = 1.5
            settings.save()

            # Load and verify
            loaded = GlobalSettingsV2.load(base_path=tmpdir_path)
            assert loaded.server.port == 9999
            assert loaded.cache.enabled is False
            assert loaded.sampling.temperature == 1.5


class TestEnsureDirectories:
    """Test the ensure_directories() method."""

    def test_ensure_directories_creates_base(self):
        """Test that ensure_directories creates base path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir) / "omlx"
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.ensure_directories()

            assert tmpdir_path.exists()

    def test_ensure_directories_creates_cache(self):
        """Test that ensure_directories creates cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.ensure_directories()

            cache_dir = tmpdir_path / "cache"
            assert cache_dir.exists()

    def test_ensure_directories_creates_logs(self):
        """Test that ensure_directories creates logs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)
            settings.ensure_directories()

            logs_dir = tmpdir_path / "logs"
            assert logs_dir.exists()


class TestToDict:
    """Test the to_dict() method."""

    def test_to_dict_contains_all_sections(self):
        """Test that to_dict includes all 13 sections."""
        settings = GlobalSettingsV2()
        d = settings.to_dict()

        expected_sections = [
            "version",
            "base_path",
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
            assert section in d, f"Missing section: {section}"

    def test_to_dict_values_match_settings(self):
        """Test that to_dict values match settings properties."""
        settings = GlobalSettingsV2()
        settings.server.port = 9999
        settings.cache.enabled = False
        d = settings.to_dict()

        assert d["server"]["port"] == 9999
        assert d["cache"]["enabled"] is False


class TestToSchedulerConfig:
    """Test the to_scheduler_config() method."""

    def test_to_scheduler_config_basic(self):
        """Test converting settings to SchedulerConfig."""
        settings = GlobalSettingsV2()
        settings.scheduler.max_num_seqs = 16
        settings.scheduler.completion_batch_size = 32
        settings.cache.initial_cache_blocks = 512

        config = settings.to_scheduler_config()
        assert config.max_num_seqs == 16
        assert config.completion_batch_size == 32
        assert config.initial_cache_blocks == 512


class TestGlobalSettingsSingleton:
    """Test the global settings singleton pattern."""

    def test_init_settings_creates_singleton(self):
        """Test that init_settings creates global instance."""
        reset_settings()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            from omlx.settings_v2 import init_settings, get_settings

            init_settings(base_path=tmpdir_path)
            settings = get_settings()
            assert isinstance(settings, GlobalSettingsV2)

    def test_get_settings_without_init_raises(self):
        """Test that get_settings raises without init."""
        reset_settings()
        with pytest.raises(RuntimeError):
            from omlx.settings_v2 import get_settings
            get_settings()


class TestPerformance:
    """Test performance characteristics."""

    def test_initialization_time(self):
        """Test that initialization is reasonably fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            start = time.time()
            settings = GlobalSettingsV2.load(base_path=tmpdir_path)
            elapsed = time.time() - start

            # Should be much faster than 1 second
            assert elapsed < 1.0, f"Initialization took {elapsed:.3f}s"

    def test_save_performance(self):
        """Test that save() is fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            settings = GlobalSettingsV2(base_path=tmpdir_path)

            start = time.time()
            settings.save()
            elapsed = time.time() - start

            # Should be much faster than 0.5 second
            assert elapsed < 0.5, f"Save took {elapsed:.3f}s"

    def test_validate_performance(self):
        """Test that validate() is fast."""
        settings = GlobalSettingsV2()

        start = time.time()
        errors = settings.validate()
        elapsed = time.time() - start

        # Should be much faster than 0.1 second
        assert elapsed < 0.1, f"Validation took {elapsed:.3f}s"
