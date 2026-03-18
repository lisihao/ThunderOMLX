# SPDX-License-Identifier: Apache-2.0
"""
Test for structured startup configuration output.

Verifies that _print_startup_config outputs machine-readable config
in [SECTION] key=value format.
"""

import sys
import os
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_print_startup_config():
    """Test that _print_startup_config outputs all expected sections."""
    from omlx.cli import _print_startup_config
    from omlx.scheduler import SchedulerConfig

    # Create a mock settings object with required attributes
    class MockServer:
        host = "127.0.0.1"
        port = 8000
        log_level = "info"

    class MockModel:
        max_model_memory = "32GB"

    class MockMemory:
        max_process_memory = "auto"

    class MockAuth:
        api_key = ""

    class MockSettings:
        base_path = "/Users/test/.omlx"
        server = MockServer()
        model = MockModel()
        memory = MockMemory()
        auth = MockAuth()

    settings = MockSettings()
    scheduler_config = SchedulerConfig()
    model_dirs = ["/Users/test/models/llama-3b", "/Users/test/models/qwen-7b"]
    mcp_config = None

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        _print_startup_config(settings, scheduler_config, model_dirs, mcp_config)

    output = f.getvalue()

    # Verify all sections present
    assert "[SERVER]" in output, "Missing [SERVER] section"
    assert "[MODEL]" in output, "Missing [MODEL] section"
    assert "[SCHEDULER]" in output, "Missing [SCHEDULER] section"
    assert "[CACHE]" in output, "Missing [CACHE] section"
    assert "[OPTIMIZATION]" in output, "Missing [OPTIMIZATION] section"
    assert "[AUTH]" in output, "Missing [AUTH] section"

    # Verify key values
    assert "host" in output and "127.0.0.1" in output
    assert "port" in output and "8000" in output
    assert "log_level" in output and "info" in output
    assert "max_num_seqs" in output and "256" in output
    assert "completion_batch_size" in output and "32" in output
    assert "paged_cache_block_size" in output and "256" in output
    assert "enable_prompt_padding" in output and "True" in output

    # MCP section should NOT be present when mcp_config is None
    assert "[MCP]" not in output

    print("Output preview:")
    print(output[:500])
    print(f"  ... ({len(output)} total chars)")


def test_print_startup_config_with_mcp():
    """Test MCP section appears when mcp_config is provided."""
    from omlx.cli import _print_startup_config
    from omlx.scheduler import SchedulerConfig

    class MockServer:
        host = "0.0.0.0"
        port = 9000
        log_level = "debug"

    class MockModel:
        max_model_memory = "48GB"

    class MockMemory:
        max_process_memory = "80%"

    class MockAuth:
        api_key = "sk-test-key-123"

    class MockSettings:
        base_path = "/Users/test/.omlx"
        server = MockServer()
        model = MockModel()
        memory = MockMemory()
        auth = MockAuth()

    settings = MockSettings()
    scheduler_config = SchedulerConfig()
    model_dirs = ["/Users/test/models"]
    mcp_config = "/Users/test/.omlx/mcp.json"

    f = io.StringIO()
    with redirect_stdout(f):
        _print_startup_config(settings, scheduler_config, model_dirs, mcp_config)

    output = f.getvalue()

    # MCP section should be present
    assert "[MCP]" in output, "Missing [MCP] section"
    assert "mcp.json" in output

    # Auth key should be masked
    assert "***" in output, "API key should be masked"
    assert "sk-test-key-123" not in output, "API key should NOT appear in output"


def test_format_bytes():
    """Test byte formatting helper."""
    from omlx.cli import _format_bytes

    assert _format_bytes(0) == "0"
    assert _format_bytes(1024) == "1.0KB"
    assert _format_bytes(1024 * 1024) == "1.0MB"
    assert _format_bytes(1024 * 1024 * 1024) == "1.0GB"
    assert _format_bytes(50 * 1024 * 1024 * 1024) == "50.0GB"
    assert _format_bytes(500) == "500B"


if __name__ == "__main__":
    tests = [
        test_format_bytes,
        test_print_startup_config,
        test_print_startup_config_with_mcp,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed")
