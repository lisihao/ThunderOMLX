#!/usr/bin/env python3
"""
Run oMLX Admin Panel benchmark via API (matching community methodology).
"""
import asyncio
import aiohttp
import json
import time

ADMIN_URL = "http://127.0.0.1:8000/admin"

async def run_benchmark():
    async with aiohttp.ClientSession() as session:
        # Step 1: Auto-login (if server supports it)
        print("🔑 Attempting auto-login...")
        async with session.get(f"{ADMIN_URL}/auto-login") as resp:
            if resp.status == 200:
                print("✅ Auto-login successful")
            else:
                print(f"⚠️ Auto-login failed ({resp.status}), trying without auth...")
        
        # Step 2: Start benchmark
        print("\n📊 Starting benchmark...")
        benchmark_request = {
            "model_id": "qwen3.5-35b-mlx",
            "prompt_lengths": [1024, 2048],  # Community standard: pp2048/tg128
            "generation_length": 128,
            "batch_sizes": []  # Skip batch tests for now
        }
        
        async with session.post(
            f"{ADMIN_URL}/api/bench/start",
            json=benchmark_request
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"❌ Failed to start benchmark: {resp.status}")
                print(f"   Response: {text}")
                return
            
            result = await resp.json()
            bench_id = result["bench_id"]
            print(f"✅ Benchmark started: {bench_id}")
        
        # Step 3: Stream progress
        print("\n📡 Streaming benchmark progress...\n")
        async with session.get(
            f"{ADMIN_URL}/api/bench/{bench_id}/stream"
        ) as resp:
            buffer = ""
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                
                buffer += chunk.decode('utf-8')
                
                # Process complete SSE messages
                while '\n\n' in buffer:
                    message, buffer = buffer.split('\n\n', 1)
                    if message.startswith('data: '):
                        data_str = message[6:]  # Remove 'data: ' prefix
                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")
                            
                            if event_type == "progress":
                                phase = event.get("phase", "")
                                message_text = event.get("message", "")
                                current = event.get("current", 0)
                                total = event.get("total", 0)
                                print(f"  [{phase}] {message_text} ({current}/{total})")
                            
                            elif event_type == "result":
                                data = event.get("data", {})
                                test_type = data.get("test_type")
                                pp = data.get("pp")
                                gen_tps = data.get("gen_tps")
                                ttft_ms = data.get("ttft_ms")
                                print(f"\n✅ Result: {test_type} pp{pp}")
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
                            
                            elif event_type == "error":
                                print(f"\n❌ Error: {event.get('message')}")
                                return
                        
                        except json.JSONDecodeError:
                            pass
        
        # Step 4: Get final results
        print("\n📋 Fetching final results...")
        async with session.get(f"{ADMIN_URL}/api/bench/{bench_id}/results") as resp:
            if resp.status == 200:
                results = await resp.json()
                
                print("\n" + "="*80)
                print("📊 FINAL RESULTS (oMLX Benchmark - Community Standard)")
                print("="*80)
                
                for r in results.get("results", []):
                    if r.get("test_type") == "single":
                        pp = r.get("pp")
                        print(f"\nTest: pp{pp}/tg128")
                        print(f"  Prefill TPS: {r.get('processing_tps'):.1f} tok/s")
                        print(f"  Generation TPS: {r.get('gen_tps'):.1f} tok/s")
                        print(f"  TTFT: {r.get('ttft_ms'):.1f}ms")
                        print(f"  E2E Latency: {r.get('e2e_latency_s'):.3f}s")
                
                # Save to file
                with open("omlx_benchmark_results.json", "w") as f:
                    json.dump(results, f, indent=2)
                print("\n💾 Results saved to: omlx_benchmark_results.json")
                
                # Compare with community baseline
                pp2048_result = next((r for r in results.get("results", []) if r.get("pp") == 2048), None)
                if pp2048_result:
                    our_tps = pp2048_result.get("gen_tps", 0)
                    community_baseline = 71.3  # From omlx.ai/benchmarks
                    diff = our_tps - community_baseline
                    pct = (diff / community_baseline) * 100
                    
                    print("\n" + "="*80)
                    print("📈 vs Community Baseline (M4 Pro, Qwen3.5-35B-A3B, 4-bit)")
                    print("="*80)
                    print(f"  ThunderOMLX: {our_tps:.1f} tok/s")
                    print(f"  oMLX Community: {community_baseline:.1f} tok/s")
                    print(f"  Difference: {diff:+.1f} tok/s ({pct:+.1f}%)")
                    print("="*80)
            else:
                print(f"❌ Failed to get results: {resp.status}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
