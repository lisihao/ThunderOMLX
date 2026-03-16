#!/usr/bin/env python3
"""
Profile ThunderOMLX performance to find bottlenecks.
"""

import cProfile
import pstats
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mlx.core as mx
from mlx_lm import load, generate

def profile_generation():
    """Profile a generation run to find bottlenecks."""

    print("=" * 80)
    print("🔍 ThunderOMLX Performance Profiling")
    print("=" * 80)

    # Load model
    print("\n⏳ Loading model...")
    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    model, tokenizer = load(str(model_path))

    # Prepare prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * 100  # ~2000 tokens
    print(f"\n📝 Prompt: {len(prompt)} characters")

    # Profile generation
    print("\n🚀 Starting profiled generation (128 tokens)...")

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()

    # Run generation
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=128,
        verbose=False
    )

    end = time.perf_counter()
    profiler.disable()

    # Print results
    print(f"\n✅ Generation completed in {end - start:.2f}s")
    print(f"📊 Throughput: {128 / (end - start):.1f} tok/s")

    # Analyze profile
    print("\n" + "=" * 80)
    print("📊 CPU PROFILE - Top 30 Functions by Cumulative Time")
    print("=" * 80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)

    profile_output = s.getvalue()
    print(profile_output)

    # Save to file
    with open("cpu_profile.txt", "w") as f:
        f.write(profile_output)
    print("\n💾 CPU profile saved to: cpu_profile.txt")

    # Analyze by tottime (self time)
    print("\n" + "=" * 80)
    print("📊 CPU PROFILE - Top 20 Functions by Self Time (tottime)")
    print("=" * 80)

    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)

    tottime_output = s2.getvalue()
    print(tottime_output)

    # Save to file
    with open("cpu_profile_tottime.txt", "w") as f:
        f.write(tottime_output)
    print("\n💾 Self-time profile saved to: cpu_profile_tottime.txt")

    # Analyze specific modules
    print("\n" + "=" * 80)
    print("📊 ThunderOMLX Specific Modules")
    print("=" * 80)

    s3 = io.StringIO()
    ps3 = pstats.Stats(profiler, stream=s3)

    # Filter for omlx modules
    print("\n🔍 omlx.cache.* modules:")
    ps3.print_stats("omlx/cache")

    print("\n🔍 omlx.contextpilot.* modules:")
    ps3.print_stats("omlx/contextpilot")

    print("\n🔍 omlx.server.* modules:")
    ps3.print_stats("omlx/server")

    print("\n" + "=" * 80)
    print("✅ Profiling complete")
    print("=" * 80)

if __name__ == "__main__":
    profile_generation()
