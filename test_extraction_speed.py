#!/usr/bin/env python3
"""Direct test of extraction speed: MLX arrays → bytes vs MLX → numpy."""

import time
import mlx.core as mx
import numpy as np

# Simulate KV cache tensors (64 layers × 2 tensors per layer)
num_layers = 64
num_tensors = num_layers * 2  # keys + values

print("🔧 Creating test data...")
print(f"   Layers: {num_layers}, Tensors: {num_tensors}")

# Create random MLX arrays (similar to KV cache)
# Shape: (batch, num_heads, seq_len, head_dim) → (1, 32, 8192, 128) for Q5_K_M
arrays = {}
for i in range(num_tensors):
    # Use bfloat16 (typical for KV cache)
    arr = mx.random.uniform(shape=(1, 32, 8192, 128), dtype=mx.bfloat16)
    arrays[f"layer_{i}"] = arr

print(f"   Total arrays: {len(arrays)}")
print(f"   Array shape: {arrays['layer_0'].shape}")
print(f"   Dtype: {arrays['layer_0'].dtype}")

# Materialize arrays
print("\n⏳ Materializing arrays (mx.eval + mx.synchronize)...")
start = time.perf_counter()
mx.eval(*arrays.values())
mx.synchronize()
materialize_time_ms = (time.perf_counter() - start) * 1000
print(f"   ✅ Materialized in {materialize_time_ms:.1f}ms")

# ============================================================================
# Approach A: Direct byte extraction (current Step 1)
# ============================================================================
print("\n" + "="*80)
print("🔍 Approach A: Direct byte extraction (MLX → bytes)")
print("="*80)

def extract_bytes_mlx(arr):
    """Extract bytes from MLX array (current implementation)."""
    if arr.dtype == mx.bfloat16:
        u16 = arr.view(mx.uint16)
        mx.eval(u16)
        raw = bytes(memoryview(u16))
    else:
        raw = bytes(memoryview(arr))
    return raw

start = time.perf_counter()
tensors_raw_a = {}
for name, arr in arrays.items():
    tensors_raw_a[name] = extract_bytes_mlx(arr)
approach_a_time_ms = (time.perf_counter() - start) * 1000

total_bytes_a = sum(len(raw) for raw in tensors_raw_a.values())
print(f"   Time: {approach_a_time_ms:.1f}ms")
print(f"   Total bytes: {total_bytes_a / 1024 / 1024:.1f} MB")
print(f"   Throughput: {total_bytes_a / 1024 / 1024 / (approach_a_time_ms / 1000):.1f} MB/s")

# ============================================================================
# Approach B: MLX → numpy conversion
# ============================================================================
print("\n" + "="*80)
print("🔍 Approach B: MLX → numpy conversion")
print("="*80)

start = time.perf_counter()
arrays_as_numpy = {}
for name, arr in arrays.items():
    # Handle bfloat16 by converting to uint16 first
    if arr.dtype == mx.bfloat16:
        u16 = arr.view(mx.uint16)
        mx.eval(u16)
        arrays_as_numpy[name] = np.array(u16, copy=True)
    else:
        arrays_as_numpy[name] = np.array(arr, copy=True)
approach_b_time_ms = (time.perf_counter() - start) * 1000

total_bytes_b = sum(arr_np.nbytes for arr_np in arrays_as_numpy.values())
print(f"   Time: {approach_b_time_ms:.1f}ms")
print(f"   Total bytes: {total_bytes_b / 1024 / 1024:.1f} MB")
print(f"   Throughput: {total_bytes_b / 1024 / 1024 / (approach_b_time_ms / 1000):.1f} MB/s")

# ============================================================================
# Approach C: numpy → bytes extraction (background thread simulation)
# ============================================================================
print("\n" + "="*80)
print("🔍 Approach C: numpy → bytes extraction (background)")
print("="*80)

start = time.perf_counter()
tensors_raw_c = {}
for name, arr_np in arrays_as_numpy.items():
    raw = bytes(memoryview(arr_np))
    tensors_raw_c[name] = raw
approach_c_time_ms = (time.perf_counter() - start) * 1000

total_bytes_c = sum(len(raw) for raw in tensors_raw_c.values())
print(f"   Time: {approach_c_time_ms:.1f}ms")
print(f"   Total bytes: {total_bytes_c / 1024 / 1024:.1f} MB")
print(f"   Throughput: {total_bytes_c / 1024 / 1024 / (approach_c_time_ms / 1000):.1f} MB/s")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("📊 Summary")
print("="*80)
print(f"Approach A (Direct MLX → bytes):   {approach_a_time_ms:.1f}ms")
print(f"Approach B (MLX → numpy):          {approach_b_time_ms:.1f}ms")
print(f"Approach C (numpy → bytes):        {approach_c_time_ms:.1f}ms")
print(f"Approach B + C (Step 2 total):     {approach_b_time_ms + approach_c_time_ms:.1f}ms")
print()
print(f"Speedup B vs A:                    {approach_a_time_ms / approach_b_time_ms:.2f}x")
print(f"Speedup (B+C) vs A:                {approach_a_time_ms / (approach_b_time_ms + approach_c_time_ms):.2f}x")
print()

if approach_b_time_ms < approach_a_time_ms:
    savings_ms = approach_a_time_ms - approach_b_time_ms
    print(f"✅ Step 2 optimization saves {savings_ms:.1f}ms on inference thread")
    print(f"   Background thread cost: {approach_c_time_ms:.1f}ms (async, no blocking)")
else:
    overhead_ms = approach_b_time_ms - approach_a_time_ms
    print(f"❌ Step 2 adds {overhead_ms:.1f}ms overhead, not beneficial")
