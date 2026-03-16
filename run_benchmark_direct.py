#!/usr/bin/env python3
"""
直接调用 oMLX benchmark 模块（社区标准测试方法）
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    from omlx.admin.benchmark import (
        BenchmarkRequest,
        create_run,
        run_benchmark,
    )
    from omlx.engine_pool import EnginePool
    from omlx.config import parse_size
    from omlx.settings import init_settings
    
    print("="*80)
    print("🔍 oMLX Benchmark (社区标准测试)")
    print("="*80)
    
    # Initialize settings
    settings = init_settings(base_path=None, cli_args=None)
    
    # Create engine pool
    model_dirs = settings.model.get_model_dirs(settings.base_path)
    engine_pool = EnginePool(
        model_dirs=[str(d) for d in model_dirs],
        max_model_memory=settings.model.get_max_model_memory_bytes(),
        scheduler_config=None,
        global_settings=settings,
    )
    
    # Benchmark request (matching community standard: pp2048/tg128)
    request = BenchmarkRequest(
        model_id="qwen3.5-35b-mlx",
        prompt_lengths=[1024, 2048, 4096],  # Community standard
        generation_length=128,
        batch_sizes=[]  # Skip batch tests for now
    )
    
    # Create benchmark run
    run = create_run(request)
    print(f"\n📊 Benchmark ID: {run.bench_id}")
    print(f"   Model: {request.model_id}")
    print(f"   Prompt lengths: {request.prompt_lengths}")
    print(f"   Generation length: {request.generation_length}\n")
    
    # Stream events
    async def stream_events():
        while True:
            try:
                event = await asyncio.wait_for(run.queue.get(), timeout=1.0)
                event_type = event.get("type")
                
                if event_type == "progress":
                    phase = event.get("phase", "")
                    message = event.get("message", "")
                    current = event.get("current", 0)
                    total = event.get("total", 0)
                    print(f"[{phase}] {message} ({current}/{total})")
                
                elif event_type == "result":
                    data = event.get("data", {})
                    test_type = data.get("test_type")
                    pp = data.get("pp")
                    gen_tps = data.get("gen_tps")
                    processing_tps = data.get("processing_tps")
                    ttft_ms = data.get("ttft_ms")
                    print(f"\n✅ Result: {test_type} pp{pp}/tg128")
                    print(f"   Prefill TPS: {processing_tps:.1f} tok/s")
                    print(f"   Generation TPS: {gen_tps:.1f} tok/s")
                    print(f"   TTFT: {ttft_ms:.1f}ms\n")
                
                elif event_type == "done":
                    summary = event.get("summary", {})
                    print(f"\n🎉 Benchmark complete!")
                    print(f"   Total time: {summary.get('total_time')}s")
                    print(f"   Total tests: {summary.get('total_tests')}")
                    break
                
                elif event_type == "upload_done":
                    data = event.get("data", {})
                    owner_hash = data.get("owner_hash")
                    success = data.get("success", 0)
                    total_uploads = data.get("total", 0)
                    print(f"\n📤 Upload to omlx.ai:")
                    print(f"   Success: {success}/{total_uploads}")
                    if owner_hash:
                        print(f"   Owner hash: {owner_hash}")
                    break
                
                elif event_type == "error":
                    print(f"\n❌ Error: {event.get('message')}")
                    break
            
            except asyncio.TimeoutError:
                # Check if run is still running
                if run.status != "running":
                    break
    
    # Run benchmark
    stream_task = asyncio.create_task(stream_events())
    await run_benchmark(run, engine_pool)
    await stream_task
    
    # Print final results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    for r in run.results:
        if r.get("test_type") == "single":
            pp = r.get("pp")
            print(f"\npp{pp}/tg128:")
            print(f"  Prefill TPS: {r.get('processing_tps'):.1f} tok/s")
            print(f"  Generation TPS: {r.get('gen_tps'):.1f} tok/s")
            print(f"  TTFT: {r.get('ttft_ms'):.1f}ms")
            print(f"  TPOT: {r.get('tpot_ms'):.2f}ms")
            print(f"  E2E: {r.get('e2e_latency_s'):.3f}s")
    
    # Save results
    import json
    with open("omlx_benchmark_results.json", "w") as f:
        json.dump({
            "bench_id": run.bench_id,
            "request": {
                "model_id": request.model_id,
                "prompt_lengths": request.prompt_lengths,
                "generation_length": request.generation_length,
            },
            "results": run.results,
            "status": run.status,
        }, f, indent=2)
    
    print("\n💾 Results saved to: omlx_benchmark_results.json")
    
    # Compare with community
    pp2048_result = next((r for r in run.results if r.get("pp") == 2048), None)
    if pp2048_result:
        our_tps = pp2048_result.get("gen_tps", 0)
        community_baseline = 71.3
        diff = our_tps - community_baseline
        pct = (diff / community_baseline) * 100
        
        print("\n" + "="*80)
        print("📈 vs oMLX Community Baseline")
        print("="*80)
        print(f"  Hardware: M4 Pro 48GB")
        print(f"  Model: Qwen3.5-35B-A3B (4-bit)")
        print(f"  Test: pp2048/tg128")
        print("")
        print(f"  ThunderOMLX: {our_tps:.1f} tok/s")
        print(f"  Community: {community_baseline:.1f} tok/s")
        print(f"  Difference: {diff:+.1f} tok/s ({pct:+.1f}%)")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
