#!/usr/bin/env python3
"""
Run ThunderOMLX server with cProfile enabled.
"""
import cProfile
import pstats
import io
import sys
import signal
import atexit

# Profile object
profiler = cProfile.Profile()

def save_profile():
    """Save profile on exit."""
    profiler.disable()
    
    print("\n" + "="*80)
    print("📊 SERVER PROFILE - Top 30 Functions by Cumulative Time")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    
    profile_output = s.getvalue()
    print(profile_output)
    
    # Save to file
    with open("server_process_profile.txt", "w") as f:
        f.write(profile_output)
    print("\n💾 Profile saved to: server_process_profile.txt")
    
    # Also save by tottime
    print("\n" + "="*80)
    print("📊 Top 20 Functions by Self Time")
    print("="*80)
    
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)
    
    tottime_output = s2.getvalue()
    print(tottime_output)
    
    with open("server_process_profile_tottime.txt", "w") as f:
        f.write(tottime_output)
    print("\n💾 Self-time profile saved to: server_process_profile_tottime.txt")

# Register exit handler
atexit.register(save_profile)
signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

# Start profiling
profiler.enable()

# Import and run omlx server
from omlx.cli import serve_command
import argparse

# Mimic CLI args
args = argparse.Namespace(
    model='/Users/lisihao/models/qwen3.5-35b-mlx',
    port=8000,
    host='127.0.0.1',
    log_level='info',
    base_path=None,
    mcp_config=None,
    paged_ssd_cache_dir=None,
    no_cache=False,
    model_dir=None,
    max_model_memory='auto',
    max_process_memory=None,
)

serve_command(args)
