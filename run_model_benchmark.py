#!/usr/bin/env python3
"""
Automated model benchmark for Chunked Prefill MVP.

Tests both disabled and enabled modes, compares performance.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def check_server_running(port=8000):
    """Check if oMLX server is running."""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def wait_for_server(port=8000, timeout=180):
    """Wait for server to start."""
    print(f"等待服务器启动（端口 {port}）...")
    start = time.time()
    while time.time() - start < timeout:
        if check_server_running(port):
            print("✅ 服务器已就绪")
            return True
        time.sleep(1)
    print("❌ 服务器启动超时")
    return False


def run_inference_test(port=8000, prompt_length=1024):
    """Run a single inference test."""
    # Generate test prompt
    prompt_text = "测试 " * prompt_length

    url = f"http://127.0.0.1:{port}/v1/completions"

    data = {
        "model": "Qwen3.5-35B-A3B-6bit",
        "prompt": prompt_text,
        "max_tokens": 50,
        "temperature": 0.7
    }

    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=60)
        latency = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "latency": latency,
                "tokens": result.get("usage", {}),
                "text": result.get("choices", [{}])[0].get("text", "")[:50]
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "latency": latency
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time
        }


def run_benchmark_suite(port=8000, label="Test"):
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"测试配置: {label}")
    print('='*60)

    test_cases = [
        ("短提示 (512 tokens)", 512),
        ("中等提示 (1024 tokens)", 1024),
        ("长提示 (2048 tokens)", 2048),
    ]

    results = {}

    for name, length in test_cases:
        print(f"\n{name}...")
        result = run_inference_test(port, length)
        results[name] = result

        if result["success"]:
            print(f"  ✅ 延迟: {result['latency']:.3f}s")
            print(f"  📝 生成: {result['text']}")
        else:
            print(f"  ❌ 失败: {result['error']}")

    return results


def compare_results(baseline, chunked):
    """Compare baseline vs chunked results."""
    print(f"\n{'='*60}")
    print("性能对比")
    print('='*60)

    print(f"\n{'测试场景':<30} {'基线':<15} {'分块':<15} {'变化':<15}")
    print('-'*75)

    for test_name in baseline:
        if baseline[test_name]["success"] and chunked[test_name]["success"]:
            base_lat = baseline[test_name]["latency"]
            chunk_lat = chunked[test_name]["latency"]
            change = ((chunk_lat - base_lat) / base_lat) * 100

            print(f"{test_name:<30} {base_lat:>8.3f}s {chunk_lat:>10.3f}s {change:>+10.1f}%")
        else:
            print(f"{test_name:<30} {'失败':<15} {'失败':<15} {'-':<15}")


def main():
    """Main benchmark runner."""
    print("🧪 ThunderOMLX Chunked Prefill - 模型基准测试")
    print("="*60)

    # Configuration
    port = 8000
    model_dir = Path.home() / ".omlx" / "models"

    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)

    # Check if server already running
    if check_server_running(port):
        print("⚠️  检测到服务器已在运行")
        print("请先停止服务器：")
        print("  ps aux | grep 'omlx serve'")
        print("  kill <pid>")
        sys.exit(1)

    print("\n📋 测试计划:")
    print("1. 启动服务器（禁用 Chunked Prefill）- 基线")
    print("2. 运行 3 个测试场景")
    print("3. 停止服务器")
    print("4. 启动服务器（启用 Chunked Prefill）")
    print("5. 运行相同测试")
    print("6. 对比结果")
    print("\n自动开始测试...")

    # Test 1: Baseline (Chunked Prefill disabled)
    print("\n" + "="*60)
    print("第 1 阶段：基线测试（禁用 Chunked Prefill）")
    print("="*60)

    # Ensure environment variable is unset
    env = os.environ.copy()
    if "OMLX_ENABLE_CHUNKED_PREFILL" in env:
        del env["OMLX_ENABLE_CHUNKED_PREFILL"]

    # Start server
    print("\n启动服务器...")
    # Use venv python
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
    if not venv_python.exists():
        print(f"❌ 虚拟环境不存在: {venv_python}")
        sys.exit(1)

    server_cmd = [
        str(venv_python), "-m", "omlx.cli", "serve",
        "--model-dir", str(model_dir),
        "--port", str(port)
    ]

    server_proc = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server
    if not wait_for_server(port):
        server_proc.kill()
        print("❌ 服务器启动失败")
        sys.exit(1)

    # Run baseline tests
    baseline_results = run_benchmark_suite(port, "基线（禁用 Chunked Prefill）")

    # Stop server
    print("\n停止服务器...")
    server_proc.terminate()
    server_proc.wait(timeout=10)
    time.sleep(2)

    # Test 2: Chunked Prefill enabled
    print("\n" + "="*60)
    print("第 2 阶段：测试（启用 Chunked Prefill）")
    print("="*60)

    # Set environment variable
    env["OMLX_ENABLE_CHUNKED_PREFILL"] = "true"
    env["OMLX_CHUNK_SIZE"] = "512"
    env["OMLX_MIN_TOKENS_FOR_CHUNKING"] = "1024"

    # Start server
    print("\n启动服务器（启用 Chunked Prefill）...")
    server_proc = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server
    if not wait_for_server(port):
        server_proc.kill()
        print("❌ 服务器启动失败")
        sys.exit(1)

    # Run chunked tests
    chunked_results = run_benchmark_suite(port, "测试（启用 Chunked Prefill）")

    # Stop server
    print("\n停止服务器...")
    server_proc.terminate()
    server_proc.wait(timeout=10)

    # Compare results
    compare_results(baseline_results, chunked_results)

    # Save results
    report = {
        "baseline": baseline_results,
        "chunked": chunked_results,
        "config": {
            "chunk_size": 512,
            "min_tokens": 1024
        }
    }

    report_file = Path(__file__).parent / "MODEL_BENCHMARK_RESULTS.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📊 完整报告已保存: {report_file}")

    # Go/No-Go decision
    print("\n" + "="*60)
    print("Go/No-Go 决策")
    print("="*60)

    all_success = all(
        baseline_results[k]["success"] and chunked_results[k]["success"]
        for k in baseline_results
    )

    if not all_success:
        print("❌ NO-GO: 部分测试失败")
        return

    # Check latency increase
    long_prompt_key = "长提示 (2048 tokens)"
    if long_prompt_key in baseline_results and long_prompt_key in chunked_results:
        base_lat = baseline_results[long_prompt_key]["latency"]
        chunk_lat = chunked_results[long_prompt_key]["latency"]
        increase = ((chunk_lat - base_lat) / base_lat) * 100

        if increase > 15:
            print(f"⚠️  NO-GO: 长提示延迟增加 {increase:.1f}% (> 15% 阈值)")
        else:
            print(f"✅ GO: 延迟增加 {increase:.1f}% (< 15% 阈值)")
            print("\n建议：可以上生产环境")
    else:
        print("⚠️  警告：长提示测试失败，无法做 Go/No-Go 决策")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断测试")
        sys.exit(0)
