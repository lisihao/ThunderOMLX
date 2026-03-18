# SPDX-License-Identifier: Apache-2.0
"""Tests for bfloat16 batch eval optimization."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_extract_tensor_bytes_batch_basic():
    """Test batch extraction with mixed dtypes."""
    import mlx.core as mx
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes, _extract_tensor_bytes_batch

    # Create mixed dtype arrays
    arrays = {
        "layer0_k": mx.ones((4, 8), dtype=mx.bfloat16),
        "layer0_v": mx.ones((4, 8), dtype=mx.bfloat16),
        "layer1_k": mx.ones((4, 8), dtype=mx.float16),
        "layer1_v": mx.ones((4, 8), dtype=mx.float16),
    }
    mx.eval(*arrays.values())

    result = _extract_tensor_bytes_batch(arrays)

    assert len(result) == 4
    for name in arrays:
        raw, dtype_str, shape = result[name]
        assert isinstance(raw, bytes)
        assert len(raw) > 0
        assert shape == [4, 8]

    # bf16 arrays should have BF16 dtype string
    assert result["layer0_k"][1] == "BF16"
    assert result["layer0_v"][1] == "BF16"
    # f16 arrays should have F16 dtype string
    assert result["layer1_k"][1] == "F16"
    assert result["layer1_v"][1] == "F16"


def test_extract_tensor_bytes_batch_matches_individual():
    """Test batch results match individual extraction."""
    import mlx.core as mx
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes, _extract_tensor_bytes_batch

    arrays = {
        f"layer{i}_{kv}": mx.random.normal((2, 16), dtype=mx.bfloat16)
        for i in range(5)
        for kv in ("k", "v")
    }
    mx.eval(*arrays.values())

    # Individual extraction
    individual = {}
    for name, arr in arrays.items():
        individual[name] = _extract_tensor_bytes(arr)

    # Batch extraction
    batch = _extract_tensor_bytes_batch(arrays)

    # Results should match
    for name in arrays:
        assert individual[name][0] == batch[name][0], f"Bytes mismatch for {name}"
        assert individual[name][1] == batch[name][1], f"Dtype mismatch for {name}"
        assert individual[name][2] == batch[name][2], f"Shape mismatch for {name}"


def test_extract_tensor_bytes_batch_all_bf16():
    """Test batch extraction with all bfloat16 arrays."""
    import mlx.core as mx
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes_batch

    arrays = {
        f"layer{i}_{kv}": mx.ones((4, 32), dtype=mx.bfloat16)
        for i in range(40)
        for kv in ("k", "v")
    }
    mx.eval(*arrays.values())

    result = _extract_tensor_bytes_batch(arrays)
    assert len(result) == 80
    for name in arrays:
        assert result[name][1] == "BF16"


def test_extract_tensor_bytes_batch_empty():
    """Test batch extraction with empty dict."""
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes_batch

    result = _extract_tensor_bytes_batch({})
    assert len(result) == 0


def test_batch_eval_performance():
    """Test that batch eval is faster than individual eval for bfloat16."""
    import mlx.core as mx
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes, _extract_tensor_bytes_batch

    # Simulate realistic block: 40 layers x 2 (K+V) = 80 arrays
    arrays = {
        f"layer{i}_{kv}": mx.random.normal((1, 256, 64), dtype=mx.bfloat16)
        for i in range(40)
        for kv in ("k", "v")
    }
    mx.eval(*arrays.values())

    # Individual extraction timing
    start = time.perf_counter()
    for _ in range(3):
        for name, arr in arrays.items():
            _extract_tensor_bytes(arr)
    individual_time = (time.perf_counter() - start) / 3

    # Batch extraction timing
    start = time.perf_counter()
    for _ in range(3):
        _extract_tensor_bytes_batch(arrays)
    batch_time = (time.perf_counter() - start) / 3

    speedup = individual_time / batch_time if batch_time > 0 else float('inf')
    print(f"  Individual: {individual_time*1000:.1f}ms")
    print(f"  Batch:      {batch_time*1000:.1f}ms")
    print(f"  Speedup:    {speedup:.1f}x")

    # Batch should be at least as fast (allow some variance)
    # The main benefit is reducing Metal sync count from 80 to 1
    assert batch_time <= individual_time * 1.2, (
        f"Batch ({batch_time*1000:.1f}ms) should not be slower than "
        f"individual ({individual_time*1000:.1f}ms)"
    )


def test_ssd_path_batch_conversion():
    """Test SSD path numpy conversion with batch bf16 handling."""
    import mlx.core as mx
    import numpy as np

    arrays = {
        f"layer{i}_{kv}": mx.ones((2, 8), dtype=mx.bfloat16)
        for i in range(5)
        for kv in ("k", "v")
    }
    mx.eval(*arrays.values())

    # Simulate the optimized SSD path
    arrays_as_numpy = {}
    bf16_views = {}
    for name, arr in arrays.items():
        if arr.dtype == mx.bfloat16:
            bf16_views[name] = arr.view(mx.uint16)
        else:
            arrays_as_numpy[name] = np.array(arr, copy=True)

    if bf16_views:
        mx.eval(*bf16_views.values())
        for name, u16 in bf16_views.items():
            arrays_as_numpy[name] = np.array(u16, copy=True)

    assert len(arrays_as_numpy) == 10
    for name, np_arr in arrays_as_numpy.items():
        assert np_arr.shape == (2, 8)
        assert np_arr.dtype == np.uint16  # bf16 stored as uint16


if __name__ == "__main__":
    tests = [
        test_extract_tensor_bytes_batch_basic,
        test_extract_tensor_bytes_batch_matches_individual,
        test_extract_tensor_bytes_batch_all_bf16,
        test_extract_tensor_bytes_batch_empty,
        test_batch_eval_performance,
        test_ssd_path_batch_conversion,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed")
