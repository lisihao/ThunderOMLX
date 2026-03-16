#!/usr/bin/env python3
"""
Integration test for Chunked Prefill with Scheduler.

This script verifies that:
1. Scheduler imports successfully with chunked prefill
2. ChunkedPrefillEngine initializes correctly
3. Environment variables are read properly
4. Graceful fallback works when not available
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that scheduler imports successfully."""
    logger.info("=" * 80)
    logger.info("TEST 1: Scheduler Import")
    logger.info("=" * 80)

    try:
        from omlx.scheduler import Scheduler, SchedulerConfig
        logger.info("✓ Scheduler imported successfully")

        from omlx.chunked_prefill import ChunkedPrefillEngine, ChunkedPrefillConfig
        logger.info("✓ ChunkedPrefillEngine imported successfully")

        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}", exc_info=True)
        return False


def test_chunked_prefill_config():
    """Test ChunkedPrefillConfig initialization."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: ChunkedPrefillConfig")
    logger.info("=" * 80)

    try:
        from omlx.chunked_prefill import ChunkedPrefillConfig

        # Test 1: From environment (disabled by default)
        config = ChunkedPrefillConfig.from_env()
        logger.info(f"✓ Config from env: enabled={config.enable_chunking}, "
                   f"chunk_size={config.chunk_size}, "
                   f"min_tokens={config.min_tokens_for_chunking}")
        assert config.enable_chunking == False, "Should be disabled by default"

        # Test 2: Enable via environment
        os.environ["OMLX_ENABLE_CHUNKED_PREFILL"] = "true"
        os.environ["OMLX_CHUNK_SIZE"] = "256"
        os.environ["OMLX_MIN_TOKENS_FOR_CHUNKING"] = "512"

        config2 = ChunkedPrefillConfig.from_env()
        logger.info(f"✓ Config from env (enabled): enabled={config2.enable_chunking}, "
                   f"chunk_size={config2.chunk_size}, "
                   f"min_tokens={config2.min_tokens_for_chunking}")
        assert config2.enable_chunking == True, "Should be enabled"
        assert config2.chunk_size == 256, "Chunk size should be 256"
        assert config2.min_tokens_for_chunking == 512, "Min tokens should be 512"

        # Cleanup
        del os.environ["OMLX_ENABLE_CHUNKED_PREFILL"]
        del os.environ["OMLX_CHUNK_SIZE"]
        del os.environ["OMLX_MIN_TOKENS_FOR_CHUNKING"]

        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}", exc_info=True)
        return False


def test_engine_creation():
    """Test ChunkedPrefillEngine creation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: ChunkedPrefillEngine Creation")
    logger.info("=" * 80)

    try:
        from omlx.chunked_prefill import ChunkedPrefillEngine, ChunkedPrefillConfig
        import mlx.core as mx

        # Create a mock model
        class MockModel:
            pass

        model = MockModel()
        config = ChunkedPrefillConfig(enable_chunking=True, chunk_size=512)

        engine = ChunkedPrefillEngine(model, config)
        logger.info(f"✓ Engine created: enabled={engine.config.enable_chunking}, "
                   f"chunk_size={engine.config.chunk_size}")

        return True
    except Exception as e:
        logger.error(f"✗ Engine creation failed: {e}", exc_info=True)
        return False


def test_scheduler_initialization():
    """Test Scheduler initialization with chunked prefill."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Scheduler Initialization")
    logger.info("=" * 80)

    try:
        from omlx.scheduler import Scheduler, SchedulerConfig, HAS_CHUNKED_PREFILL
        import mlx.core as mx

        logger.info(f"✓ HAS_CHUNKED_PREFILL flag: {HAS_CHUNKED_PREFILL}")

        # Create mock model and tokenizer
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {})()

        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 2

        model = MockModel()
        tokenizer = MockTokenizer()
        config = SchedulerConfig()

        # Initialize scheduler
        scheduler = Scheduler(model, tokenizer, config)

        if HAS_CHUNKED_PREFILL:
            logger.info(f"✓ Scheduler.chunked_prefill_engine: {scheduler.chunked_prefill_engine}")
            logger.info(f"✓ Scheduler.chunked_prefill_config: {scheduler.chunked_prefill_config}")
            assert scheduler.chunked_prefill_engine is not None, "Engine should be initialized"
            assert scheduler.chunked_prefill_config is not None, "Config should be initialized"
        else:
            logger.info("✓ ChunkedPrefill not available (graceful fallback)")
            assert scheduler.chunked_prefill_engine is None
            assert scheduler.chunked_prefill_config is None

        return True
    except Exception as e:
        logger.error(f"✗ Scheduler initialization failed: {e}", exc_info=True)
        return False


def main():
    """Run all integration tests."""
    logger.info("\nChunked Prefill Integration Tests")
    logger.info("=" * 80)

    tests = [
        ("Import", test_imports),
        ("Config", test_chunked_prefill_config),
        ("Engine", test_engine_creation),
        ("Scheduler Init", test_scheduler_initialization),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}", exc_info=True)
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

    logger.info(f"\nTotal: {passed}/{total} passed")
    logger.info("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
