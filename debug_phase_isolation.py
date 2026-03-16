#!/usr/bin/env python3
"""
Phase Isolation Test - Disable phases one by one to find the culprit

Target: Identify which Phase causes the Metal concurrent error
Method: Create engine configurations with different phases disabled
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_with_phase_config(test_name: str, disable_phases: list):
    """Test with specific phases disabled"""
    print(f"\n{'=' * 80}")
    print(f"🧪 {test_name}")
    print(f"{'=' * 80}")
    print(f"  Disabled: {', '.join(disable_phases) if disable_phases else 'None (All phases enabled)'}")
    print("")

    # Set environment to disable phases
    for phase in disable_phases:
        os.environ[f"DISABLE_{phase}"] = "1"

    from omlx.engine.batched import BatchedEngine

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        # Run 2 concurrent requests (minimal failing case)
        prompts = [
            "Explain the key differences between Python and JavaScript.",
            "What are the main advantages of using TypeScript?"
        ]

        async def single_request(prompt: str, request_id: str):
            tokens = 0
            try:
                async for output in engine.stream_generate(
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.0
                ):
                    if output.new_text:
                        tokens += 1
                return tokens
            except Exception as e:
                print(f"❌ {request_id} failed: {e}")
                raise

        # Concurrent execution
        tasks = [
            single_request(prompts[0], "R1"),
            single_request(prompts[1], "R2")
        ]

        import time
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start

        # Check results
        success = True
        total_tokens = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Request R{i+1} failed: {result}")
                success = False
            else:
                print(f"✅ Request R{i+1} completed: {result} tokens")
                total_tokens += result

        if success:
            print(f"\n🎉 {test_name} - 成功!")
            print(f"   Total tokens: {total_tokens}, Time: {elapsed:.2f}s")
            return True
        else:
            print(f"\n❌ {test_name} - 失败!")
            return False

    except Exception as e:
        print(f"\n❌ {test_name} - 异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()
        # Clean up environment
        for phase in disable_phases:
            os.environ.pop(f"DISABLE_{phase}", None)


async def main():
    print("=" * 80)
    print("🔍 Phase Isolation Test")
    print("=" * 80)
    print("")
    print("Goal: Find which Phase causes Metal concurrent error")
    print("Method: Disable phases progressively to isolate the culprit")
    print("")

    # Test scenarios
    scenarios = [
        ("Baseline (All phases enabled)", []),
        ("Disable Phase 4 (Batch Metal)", ["PHASE4"]),
        ("Disable Phase 2+4", ["PHASE2", "PHASE4"]),
        ("Disable Phase 1+2+4", ["PHASE1", "PHASE2", "PHASE4"]),
    ]

    results = {}

    for test_name, disable_phases in scenarios:
        success = await test_with_phase_config(test_name, disable_phases)
        results[test_name] = success

        if not success:
            print(f"\n⚠️  {test_name} failed - testing next configuration")

        # Brief pause between tests
        await asyncio.sleep(2)

    print("\n" + "=" * 80)
    print("📊 Phase Isolation Results")
    print("=" * 80)

    for test_name, success in results.items():
        status = "✅ Pass" if success else "❌ Fail"
        print(f"  {test_name}: {status}")

    # Analyze results
    print("\n" + "=" * 80)
    print("🔍 Root Cause Analysis")
    print("=" * 80)

    if not results["Baseline (All phases enabled)"]:
        if results.get("Disable Phase 4 (Batch Metal)"):
            print("✅ Phase 4 (Batch Metal eval) is the culprit!")
            print("   Problem: Batch mx.eval() causes Metal command buffer conflicts")
        elif results.get("Disable Phase 2+4"):
            print("✅ Phase 2 (Async save_block) is the culprit!")
            print("   Problem: Background CacheSaveExecutor conflicts with Metal")
        elif results.get("Disable Phase 1+2+4"):
            print("✅ Phase 1 (Async tensor extraction) is the culprit!")
            print("   Problem: mx.synchronize() + background thread extraction causes issues")
        else:
            print("⚠️  Error persists even with all phases disabled")
            print("   Problem may be in base BatchedEngine or MLX itself")
    else:
        print("✅ Test passed! No Metal error detected.")
        print("   (Race condition may require multiple runs to reproduce)")


if __name__ == "__main__":
    asyncio.run(main())
