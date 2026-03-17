#!/usr/bin/env python3
"""
Benchmark Prefill + Generation performance with detailed profiling.

Features:
- 🔍 Integrated performance profiling (using omlx.profiling framework)
- 📊 Detailed bottleneck analysis
- 💾 Automatic results export (JSON + human-readable)
- 📈 Multi-trial averaging with variance analysis

Usage:
    python benchmark_prefill_generation.py \\
        --model ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf \\
        --input-length 8192 \\
        --output-length 128 \\
        --warmup 1 \\
        --trials 5

Metrics:
- Processing TPS: Total tokens / Total time (includes Prefill + Generation)
- TTFT (Time To First Token): Prefill latency
- Generation TPS: Output tokens / Generation time
- Profiling breakdown: prefill.total, prefill.model_forward, prefill.synchronize, etc.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Prefill + Generation with profiling")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--input-length", type=int, default=8192, help="Input prompt length in tokens")
    parser.add_argument("--output-length", type=int, default=128, help="Output generation length in tokens")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--trials", type=int, default=5, help="Number of test trials")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable detailed profiling")

    args = parser.parse_args()

    # Enable profiling if requested
    if args.enable_profiling:
        os.environ['OMLX_ENABLE_PROFILING'] = 'true'
        print("🔍 Profiling ENABLED")

    from omlx.engine.batched import BatchedEngine
    from omlx.scheduler import SchedulerConfig
    from omlx.profiling import get_global_profiler, print_profiling_stats
    from transformers import AutoTokenizer

    print("=" * 80)
    print("🔍 Prefill + Generation Benchmark (with Profiling)")
    print("=" * 80)

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\nSearching for alternative models...")
        # Search for alternative Q4/Q5 models
        models_dir = Path.home() / "models"
        if models_dir.exists():
            gguf_files = list(models_dir.glob("**/*.gguf"))
            if gguf_files:
                print(f"Found {len(gguf_files)} GGUF model(s):")
                for f in gguf_files[:5]:
                    print(f"  - {f}")
                print(f"\nPlease specify --model with one of the above paths")
        return

    # Load tokenizer
    # Determine tokenizer path: for GGUF files use parent dir, for MLX dirs use the dir itself
    if model_path.is_file():
        tokenizer_path = model_path.parent
    else:
        tokenizer_path = model_path

    print(f"\n⏳ Loading tokenizer from {tokenizer_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True
    )

    # Generate prompt of specified length
    print(f"⏳ Generating {args.input_length} token prompt...")
    base_words = [
        "technology", "innovation", "development", "programming", "software",
        "architecture", "infrastructure", "implementation", "optimization",
        "scalability", "reliability", "maintainability", "security", "performance",
    ]
    text = " ".join(base_words * 2000)
    tokens = tokenizer.encode(text)[:args.input_length]
    prompt = tokenizer.decode(tokens)
    actual_input_tokens = len(tokens)
    print(f"✅ Prompt: {actual_input_tokens} tokens\n")

    # Initialize engine
    print("⏳ Initializing engine...")
    scheduler_config = SchedulerConfig()
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True,
        scheduler_config=scheduler_config
    )

    await engine.start()
    print("✅ Engine started\n")

    try:
        # Get profiler
        profiler = get_global_profiler()

        # Warmup runs
        if args.warmup > 0:
            print(f"🔥 Warmup ({args.warmup} run{'s' if args.warmup > 1 else ''})...")
            for i in range(args.warmup):
                print(f"   Warmup {i+1}/{args.warmup}...", end="", flush=True)
                async for _ in engine.stream_generate(
                    prompt=prompt,
                    max_tokens=args.output_length,
                    temperature=0.0
                ):
                    pass
                print(" ✓")

            # Reset profiler after warmup
            if profiler.enabled:
                profiler.reset()
                print("   Profiler reset after warmup")

            print()

        # Test runs
        print(f"📊 Running {args.trials} trial{'s' if args.trials > 1 else ''}...")
        print("-" * 80)

        results = []

        for trial in range(args.trials):
            wall_start = time.perf_counter()
            first_token_time = None
            output_tokens = 0

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=args.output_length,
                temperature=0.0
            ):
                if output.new_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    output_tokens += 1

            wall_end = time.perf_counter()

            # Calculate metrics
            total_time = wall_end - wall_start
            ttft = first_token_time - wall_start if first_token_time else total_time
            generation_time = wall_end - first_token_time if first_token_time else 0

            total_tokens = actual_input_tokens + output_tokens
            processing_tps = total_tokens / total_time
            generation_tps = output_tokens / generation_time if generation_time > 0 else 0

            results.append({
                "trial": trial + 1,
                "total_time": total_time,
                "ttft": ttft,
                "generation_time": generation_time,
                "input_tokens": actual_input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "processing_tps": processing_tps,
                "generation_tps": generation_tps
            })

            print(f"Trial {trial+1}/{args.trials}:")
            print(f"  Total Time: {total_time:.3f}s")
            print(f"  TTFT: {ttft:.3f}s")
            print(f"  Generation Time: {generation_time:.3f}s")
            print(f"  Input Tokens: {actual_input_tokens}")
            print(f"  Output Tokens: {output_tokens}")
            print(f"  Processing TPS: {processing_tps:.1f} tok/s")
            print(f"  Generation TPS: {generation_tps:.1f} tok/s")
            print()

        # Calculate statistics
        avg_total_time = sum(r["total_time"] for r in results) / len(results)
        avg_ttft = sum(r["ttft"] for r in results) / len(results)
        avg_generation_time = sum(r["generation_time"] for r in results) / len(results)
        avg_processing_tps = sum(r["processing_tps"] for r in results) / len(results)
        avg_generation_tps = sum(r["generation_tps"] for r in results) / len(results)

        # Calculate variance
        import statistics
        processing_tps_values = [r["processing_tps"] for r in results]
        processing_tps_std = statistics.stdev(processing_tps_values) if len(processing_tps_values) > 1 else 0
        processing_tps_cv = (processing_tps_std / avg_processing_tps * 100) if avg_processing_tps > 0 else 0

        # Print summary
        print("=" * 80)
        print(f"📈 Summary (Average over {args.trials} trials)")
        print("=" * 80)
        print(f"Total Time:        {avg_total_time:.3f}s")
        print(f"TTFT:              {avg_ttft:.3f}s")
        print(f"Generation Time:   {avg_generation_time:.3f}s")
        print(f"Processing TPS:    {avg_processing_tps:.1f} ± {processing_tps_std:.1f} tok/s (CV: {processing_tps_cv:.1f}%) ⭐")
        print(f"Generation TPS:    {avg_generation_tps:.1f} tok/s")
        print("=" * 80)

        # Print profiling stats if enabled
        if profiler.enabled:
            print()
            print("=" * 80)
            print("🔍 Detailed Profiling Breakdown")
            print("=" * 80)
            print_profiling_stats(top_n=30, min_percent=0.1)
            print("=" * 80)

            # Get profiling data
            profiling_data = profiler.get_stats()

            # Analyze bottlenecks
            print()
            print("=" * 80)
            print("🔍 Bottleneck Analysis")
            print("=" * 80)

            bottlenecks = [
                (name, op_stats)
                for name, op_stats in profiling_data['top_operations']
                if op_stats['percent'] > 5.0 and 'prefill.' in name
            ]

            if bottlenecks:
                print(f"\nFound {len(bottlenecks)} major bottleneck(s) (占比 > 5%):\n")
                for name, op_stats in bottlenecks:
                    print(f"  ⚠️  {name}")
                    print(f"      - Average Time: {op_stats['avg_ms']:.2f} ms")
                    print(f"      - Percentage: {op_stats['percent']:.1f}%")
                    if op_stats['count'] > 1:
                        print(f"      - Call Count: {op_stats['count']}")
                    print()

                # Optimization suggestions
                print("💡 Optimization Suggestions:")
                for name, op_stats in bottlenecks:
                    if 'model_forward' in name and op_stats['percent'] > 70:
                        print(f"  - {name}: Consider chunked prefill or FlashAttention")
                    elif 'synchronize' in name and op_stats['percent'] > 5:
                        print(f"  - {name}: GPU → CPU sync overhead, consider async operations")
                    elif 'prepare_inputs' in name and op_stats['percent'] > 5:
                        print(f"  - {name}: Optimize input preparation logic")
                    elif 'cache_ops' in name and op_stats['percent'] > 5:
                        print(f"  - {name}: Consider async cache operations")
                print()
            else:
                print("\n✅ No major bottlenecks detected (all operations < 5%)\n")

            print("=" * 80)
        else:
            profiling_data = None

        # Save results
        output_file = Path("/tmp/benchmark_processing_tps.json")
        output_data = {
            "config": {
                "model": str(model_path),
                "input_length": args.input_length,
                "output_length": args.output_length,
                "warmup": args.warmup,
                "trials": args.trials,
                "profiling_enabled": profiler.enabled
            },
            "results": results,
            "averages": {
                "total_time": avg_total_time,
                "ttft": avg_ttft,
                "generation_time": avg_generation_time,
                "processing_tps": avg_processing_tps,
                "processing_tps_std": processing_tps_std,
                "processing_tps_cv": processing_tps_cv,
                "generation_tps": avg_generation_tps
            }
        }

        if profiling_data:
            output_data["profiling"] = profiling_data

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Compare with baseline if available
        baseline_file = Path("/tmp/benchmark_baseline.json")
        if baseline_file.exists():
            print()
            print("=" * 80)
            print("📊 Comparison with Baseline")
            print("=" * 80)

            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)

            baseline_tps = baseline_data["averages"]["processing_tps"]
            current_tps = avg_processing_tps
            diff_tps = current_tps - baseline_tps
            diff_percent = (diff_tps / baseline_tps) * 100

            print(f"Baseline Processing TPS:  {baseline_tps:.1f} tok/s")
            print(f"Current Processing TPS:   {current_tps:.1f} tok/s")
            print(f"Difference:               {diff_tps:+.1f} tok/s ({diff_percent:+.1f}%)")

            if diff_percent >= 2.5:
                print(f"✅ Significant improvement: +{diff_percent:.1f}%")
            elif diff_percent >= 1.0:
                print(f"✅ Slight improvement: +{diff_percent:.1f}%")
            elif diff_percent >= -1.0:
                print(f"⚪ No significant change: {diff_percent:+.1f}%")
            else:
                print(f"⚠️  Performance regression: {diff_percent:.1f}%")

            print("=" * 80)

    finally:
        await engine.stop()
        print("\n✅ Engine stopped")


if __name__ == "__main__":
    asyncio.run(main())
