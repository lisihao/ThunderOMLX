#!/usr/bin/env python3
"""
Debug script for long prompt (2048 tokens) failure.

Tests chunked prefill with increasing prompt lengths to find the breaking point.
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

    print(f"\n{'='*60}")
    print(f"测试提示长度: {length} tokens")
    print('='*60)

    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=timeout)
        latency = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功: 延迟 {latency:.3f}s")
            print(f"   生成: {result.get('choices', [{}])[0].get('text', '')[:50]}")
            return {"success": True, "latency": latency, "error": None}
        else:
            print(f"❌ 失败: HTTP {response.status_code}")
            print(f"   响应: {response.text[:200]}")
            return {"success": False, "latency": latency, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        print(f"❌ 失败: 请求超时 ({latency:.1f}s)")
        return {"success": False, "latency": latency, "error": "Timeout"}
    except requests.exceptions.ConnectionError as e:
        latency = time.time() - start_time
        print(f"❌ 失败: 连接错误 ({latency:.1f}s)")
        print(f"   错误: {e}")
        return {"success": False, "latency": latency, "error": f"Connection: {e}"}
    except Exception as e:
        latency = time.time() - start_time
        print(f"❌ 失败: 未知错误 ({latency:.1f}s)")
        print(f"   错误: {e}")
        return {"success": False, "latency": latency, "error": str(e)}


def check_server_logs(server_proc):
    """Check server stdout/stderr for errors."""
    if server_proc.poll() is not None:
        print("\n⚠️  服务器进程已退出！")
        stdout, stderr = server_proc.communicate()
        if stderr:
            print("STDERR:")
            print(stderr[-2000:])  # Last 2000 chars
        return False
    return True


def main():
    """Main debugging workflow."""
    print("🔍 Chunked Prefill 长提示调试")
    print("="*60)

    port = 8000
    model_dir = Path.home() / ".omlx" / "models"
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"

    if not venv_python.exists():
        print(f"❌ 虚拟环境不存在: {venv_python}")
        sys.exit(1)

    # Test configuration
    enable_chunked = True
    chunk_size = 512
    min_tokens = 1024

    # Prompt lengths to test (gradually increase)
    test_lengths = [512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072]

    print(f"\n配置:")
    print(f"  OMLX_ENABLE_CHUNKED_PREFILL: {enable_chunked}")
    print(f"  OMLX_CHUNK_SIZE: {chunk_size}")
    print(f"  OMLX_MIN_TOKENS_FOR_CHUNKING: {min_tokens}")
    print(f"\n测试序列: {test_lengths}")
    print("\n按 Enter 开始...")
    input()

    # Setup environment
    env = os.environ.copy()
    if enable_chunked:
        env["OMLX_ENABLE_CHUNKED_PREFILL"] = "true"
        env["OMLX_CHUNK_SIZE"] = str(chunk_size)
        env["OMLX_MIN_TOKENS_FOR_CHUNKING"] = str(min_tokens)
    else:
        if "OMLX_ENABLE_CHUNKED_PREFILL" in env:
            del env["OMLX_ENABLE_CHUNKED_PREFILL"]

    env["OMLX_LOG_LEVEL"] = "debug"  # Enable debug logging

    # Start server
    print(f"\n启动服务器（debug 模式）...")
    server_cmd = [
        str(venv_python), "-m", "omlx.cli", "serve",
        "--model-dir", str(model_dir),
        "--port", str(port),
        "--log-level", "debug"
    ]

    # Capture server output
    log_file = Path(__file__).parent / "server_debug.log"
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
        print(f"查看日志: tail -100 {log_file}")
        sys.exit(1)

    # Run tests
    results = {}
    breaking_point = None

    for length in test_lengths:
        # Check if server still alive
        if not check_server_logs(server_proc):
            breaking_point = length
            print(f"\n💥 服务器崩溃于 {length} tokens！")
            break

        result = test_prompt_length(port, length, timeout=120)
        results[length] = result

        if not result["success"]:
            breaking_point = length
            print(f"\n💥 失败于 {length} tokens")

            # Check server status
            time.sleep(2)
            if not check_server_logs(server_proc):
                print("⚠️  服务器已崩溃")
            break

        # Small delay between tests
        time.sleep(2)

    # Stop server
    print("\n停止服务器...")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    # Summary
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    print(f"\n{'长度 (tokens)':<20} {'状态':<15} {'延迟':<15} {'错误'}")
    print("-"*70)

    for length in test_lengths:
        if length not in results:
            print(f"{length:<20} {'未测试':<15} {'-':<15} -")
            continue

        r = results[length]
        status = "✅ 成功" if r["success"] else "❌ 失败"
        latency = f"{r['latency']:.3f}s" if r["latency"] else "-"
        error = r["error"] if r["error"] else "-"

        print(f"{length:<20} {status:<15} {latency:<15} {error[:30]}")

    if breaking_point:
        print(f"\n💥 破坏点: {breaking_point} tokens")
    else:
        print(f"\n✅ 所有测试通过！")

    # Save results
    report_file = Path(__file__).parent / "DEBUG_REPORT.json"
    with open(report_file, "w") as f:
        json.dump({
            "config": {
                "chunked_enabled": enable_chunked,
                "chunk_size": chunk_size,
                "min_tokens": min_tokens
            },
            "breaking_point": breaking_point,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📊 详细报告: {report_file}")
    print(f"📋 服务器日志: {log_file}")

    print("\n建议:")
    if breaking_point:
        safe_limit = breaking_point - 256
        print(f"1. 设置 OMLX_MIN_TOKENS_FOR_CHUNKING={breaking_point} 避开失败点")
        print(f"2. 或设置更保守值: OMLX_MIN_TOKENS_FOR_CHUNKING={safe_limit}")
        print(f"3. 查看日志分析崩溃原因: tail -500 {log_file}")
    else:
        print("1. 所有测试通过，可以放心使用")
        print("2. 考虑逐步降低 MIN_TOKENS_FOR_CHUNKING 到 1024")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断测试")
        sys.exit(0)
