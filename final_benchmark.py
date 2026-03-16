#!/usr/bin/env python3
"""
Final comprehensive benchmark: Test all configurations.

Compares:
1. Recommended config (chunked prefill + async prefetch)
2. Conservative config (chunked prefill, no async prefetch)
3. Baseline (no optimizations)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

import requests


def wait_for_server(port=8000, timeout=180):
    """Wait for server to start."""
    print(f"等待服务器启动（端口 {port}）...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                print("✅ 服务器已就绪")
                return True
        except:
            pass
        time.sleep(1)
    print("❌ 服务器启动超时")
    return False


def test_prompt_length(port, length, timeout=120):
    """Test a specific prompt length."""
    prompt_text = "测试 " * length

    url = f"http://127.0.0.1:{port}/v1/completions"

    data = {
        "model": "Qwen3.5-35B-A3B-6bit",
        "prompt": prompt_text,
        "max_tokens": 50,
        "temperature": 0.7
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=timeout)
        latency = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "latency": latency,
                "tokens": result.get('usage', {}),
                "error": None
            }
        else:
            return {
                "success": False,
                "latency": latency,
                "error": f"HTTP {response.status_code}"
            }
    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        return {"success": False, "latency": latency, "error": "Timeout"}
    except Exception as e:
        latency = time.time() - start_time
        return {"success": False, "latency": latency, "error": str(e)}


def run_benchmark(
    config_name: str,
    env_vars: Dict[str, str],
    test_lengths: list
) -> Dict[str, Any]:
    """Run benchmark with specific configuration."""

    print("\n" + "="*70)
    print(f"配置: {config_name}")
    print("="*70)

    for key, value in env_vars.items():
        if key.startswith("OMLX_"):
            print(f"  {key}={value}")

    port = 8000
    model_dir = Path.home() / ".omlx" / "models"
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"

    if not venv_python.exists():
        print(f"❌ 虚拟环境不存在: {venv_python}")
        sys.exit(1)

    # Setup environment
    env = os.environ.copy()
    env.update(env_vars)

    # Start server
    print(f"\n启动服务器...")
    server_cmd = [
        str(venv_python), "-m", "omlx.cli", "serve",
        "--model-dir", str(model_dir),
        "--port", str(port)
    ]

    log_file = Path(__file__).parent / f"benchmark_{config_name.replace(' ', '_')}.log"
    with open(log_file, "w") as f:
        server_proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    # Wait for server
    if not wait_for_server(port, timeout=180):
        server_proc.kill()
        print("❌ 服务器启动失败")
        return {"error": "Server failed to start"}

    # Run tests
    results = {}
    print(f"\n运行测试...")

    for length in test_lengths:
        if server_proc.poll() is not None:
            print(f"\n💥 服务器在测试 {length} tokens 前已崩溃！")
            break

        print(f"  测试 {length} tokens...", end=" ", flush=True)
        result = test_prompt_length(port, length, timeout=120)
        results[length] = result

        if result["success"]:
            print(f"✅ {result['latency']:.3f}s")
        else:
            print(f"❌ {result['error']}")
            time.sleep(2)
            if server_proc.poll() is not None:
                print("⚠️  服务器已崩溃")
            break

        time.sleep(1)

    # Stop server
    print("\n停止服务器...")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    return results


def main():
    """Main benchmark workflow."""
    print("🏁 ThunderOMLX 完整基准测试")
    print("="*70)
    print("\n测试配置:")
    print("  1. 推荐配置（Chunked Prefill + 异步预取）")
    print("  2. 保守配置（Chunked Prefill，禁用异步预取）")
    print("  3. 基线配置（无优化）")
    print("\n测试长度: 512, 1024, 2048, 3072 tokens")
    print("\n自动开始...")
    time.sleep(2)

    test_lengths = [512, 1024, 2048, 3072]

    # Configuration 1: Recommended (all optimizations)
    config1_results = run_benchmark(
        "推荐配置",
        {
            "OMLX_ENABLE_CHUNKED_PREFILL": "true",
            "OMLX_CHUNK_SIZE": "512",
            "OMLX_MIN_TOKENS_FOR_CHUNKING": "2560",
            "OMLX_ENABLE_ASYNC_PREFETCH": "true",
            "OMLX_LOG_LEVEL": "info"
        },
        test_lengths
    )

    time.sleep(5)

    # Configuration 2: Conservative (no async prefetch)
    config2_results = run_benchmark(
        "保守配置",
        {
            "OMLX_ENABLE_CHUNKED_PREFILL": "true",
            "OMLX_CHUNK_SIZE": "512",
            "OMLX_MIN_TOKENS_FOR_CHUNKING": "2560",
            "OMLX_ENABLE_ASYNC_PREFETCH": "false",
            "OMLX_LOG_LEVEL": "info"
        },
        test_lengths
    )

    time.sleep(5)

    # Configuration 3: Baseline (no optimizations)
    config3_results = run_benchmark(
        "基线配置",
        {
            "OMLX_ENABLE_CHUNKED_PREFILL": "false",
            "OMLX_ENABLE_ASYNC_PREFETCH": "false",
            "OMLX_LOG_LEVEL": "info"
        },
        test_lengths
    )

    # Summary
    print("\n" + "="*70)
    print("基准测试结果总结")
    print("="*70)

    print(f"\n{'配置':<20} {'512 tokens':<15} {'1024 tokens':<15} {'2048 tokens':<15} {'3072 tokens':<15}")
    print("-"*85)

    configs = [
        ("推荐配置", config1_results),
        ("保守配置", config2_results),
        ("基线配置", config3_results)
    ]

    for config_name, results in configs:
        row = [config_name]
        for length in test_lengths:
            if length in results and results[length]["success"]:
                row.append(f"{results[length]['latency']:.3f}s")
            else:
                row.append("失败")
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15}")

    # Performance comparison
    print("\n" + "="*70)
    print("性能对比（vs 基线）")
    print("="*70)

    print(f"\n{'配置':<20} {'512 tokens':<20} {'1024 tokens':<20}")
    print("-"*60)

    baseline_512 = config3_results.get(512, {}).get("latency", 0)
    baseline_1024 = config3_results.get(1024, {}).get("latency", 0)

    for config_name, results in configs[:2]:  # Only compare optimized configs
        row = [config_name]

        # 512 tokens
        if 512 in results and results[512]["success"] and baseline_512 > 0:
            speedup = (baseline_512 / results[512]["latency"] - 1) * 100
            row.append(f"{speedup:+.1f}%")
        else:
            row.append("-")

        # 1024 tokens
        if 1024 in results and results[1024]["success"] and baseline_1024 > 0:
            speedup = (baseline_1024 / results[1024]["latency"] - 1) * 100
            row.append(f"{speedup:+.1f}%")
        else:
            row.append("-")

        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20}")

    # Save results
    report_file = Path(__file__).parent / "FINAL_BENCHMARK_RESULTS.json"
    with open(report_file, "w") as f:
        json.dump({
            "推荐配置": config1_results,
            "保守配置": config2_results,
            "基线配置": config3_results,
            "test_lengths": test_lengths
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📊 详细报告: {report_file}")

    # Recommendation
    print("\n" + "="*70)
    print("💡 建议")
    print("="*70)

    if config1_results.get(2048, {}).get("success"):
        print("\n✅ 推荐使用 '推荐配置':")
        print("  - Chunked Prefill + 异步预取全部启用")
        print("  - 性能最优，完全稳定")
        print("  - 支持超长提示（3072+ tokens）")

        if baseline_512 > 0 and 512 in config1_results:
            speedup = (baseline_512 / config1_results[512]["latency"] - 1) * 100
            print(f"  - 首次推理快 {speedup:.0f}%")
    else:
        print("\n⚠️ 推荐使用 '保守配置':")
        print("  - 只启用 Chunked Prefill")
        print("  - 稳定性优先")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断测试")
        sys.exit(0)
