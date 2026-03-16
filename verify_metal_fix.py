#!/usr/bin/env python3
"""
Quick verification script for Metal GPU synchronization fix.

Tests if the chunked prefill can now handle 1024 and 2048 tokens without crashing.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

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

    print(f"\n测试 {length} tokens...")
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=timeout)
        latency = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ 成功: {latency:.3f}s")
            text = result.get('choices', [{}])[0].get('text', '')[:50]
            print(f"  📝 生成: {text}")
            return {"success": True, "latency": latency, "error": None}
        else:
            print(f"  ❌ 失败: HTTP {response.status_code}")
            return {"success": False, "latency": latency, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        print(f"  ❌ 失败: 请求超时 ({latency:.1f}s)")
        return {"success": False, "latency": latency, "error": "Timeout"}
    except Exception as e:
        latency = time.time() - start_time
        print(f"  ❌ 失败: {e}")
        return {"success": False, "latency": latency, "error": str(e)}


def main():
    """Main verification workflow."""
    print("🔧 Metal GPU 同步修复验证")
    print("="*60)

    port = 8000
    model_dir = Path.home() / ".omlx" / "models"
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"

    if not venv_python.exists():
        print(f"❌ 虚拟环境不存在: {venv_python}")
        sys.exit(1)

    # Setup environment with chunked prefill enabled
    env = os.environ.copy()
    env["OMLX_ENABLE_CHUNKED_PREFILL"] = "true"
    env["OMLX_CHUNK_SIZE"] = "512"
    env["OMLX_MIN_TOKENS_FOR_CHUNKING"] = "1024"
    env["OMLX_LOG_LEVEL"] = "info"

    # Start server
    print(f"\n启动服务器（启用 Chunked Prefill）...")
    server_cmd = [
        str(venv_python), "-m", "omlx.cli", "serve",
        "--model-dir", str(model_dir),
        "--port", str(port)
    ]

    log_file = Path(__file__).parent / "verify_fix.log"
    with open(log_file, "w") as f:
        server_proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    print(f"服务器日志: {log_file}")

    # Wait for server
    if not wait_for_server(port, timeout=180):
        server_proc.kill()
        print("❌ 服务器启动失败")
        sys.exit(1)

    print("\n" + "="*60)
    print("运行测试")
    print("="*60)

    # Test critical lengths
    test_lengths = [512, 1024, 2048]
    results = {}

    for length in test_lengths:
        # Check server health before each test
        if server_proc.poll() is not None:
            print(f"\n💥 服务器在测试 {length} tokens 前已崩溃！")
            break

        result = test_prompt_length(port, length, timeout=120)
        results[length] = result

        if not result["success"]:
            print(f"\n💥 失败于 {length} tokens")
            # Give server time to crash if it's going to
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

    # Summary
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)

    all_passed = True
    for length in test_lengths:
        if length not in results:
            print(f"{length} tokens: 未测试")
            continue

        r = results[length]
        if r["success"]:
            print(f"{length} tokens: ✅ 通过 ({r['latency']:.3f}s)")
        else:
            print(f"{length} tokens: ❌ 失败 ({r['error']})")
            all_passed = False

    print("\n" + "="*60)
    if all_passed and len(results) == len(test_lengths):
        print("✅ 修复成功！所有测试通过。")
        print("\n下一步：运行完整基准测试确认性能")
    else:
        print("❌ 修复未完全成功。请查看日志:")
        print(f"   tail -100 {log_file}")

    # Save results
    report_file = Path(__file__).parent / "METAL_FIX_VERIFICATION.json"
    with open(report_file, "w") as f:
        json.dump({
            "fix_applied": True,
            "test_results": results,
            "all_passed": all_passed
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📊 详细报告: {report_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断测试")
        sys.exit(0)
