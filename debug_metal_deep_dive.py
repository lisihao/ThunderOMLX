#!/usr/bin/env python3
"""
Metal 并发深度调试

目标：找到 MLX Metal 操作的所有并发冲突点

策略：
1. Hook MLX Metal 操作，记录时序
2. 检测并发访问模式
3. 分析冲突根源
"""
import asyncio
import sys
import threading
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

# 全局追踪器
metal_ops_log = []
metal_ops_lock = threading.Lock()


class MetalOpTracker:
    """Metal 操作追踪器"""

    @staticmethod
    def log_op(op_type, thread_id, details=""):
        """记录 Metal 操作"""
        with metal_ops_lock:
            timestamp = time.time()
            entry = {
                'timestamp': timestamp,
                'op_type': op_type,
                'thread_id': thread_id,
                'thread_name': threading.current_thread().name,
                'details': details
            }
            metal_ops_log.append(entry)

            # 实时检测冲突
            if len(metal_ops_log) > 1:
                last_op = metal_ops_log[-2]
                if last_op['thread_id'] != thread_id:
                    time_gap = (timestamp - last_op['timestamp']) * 1000
                    if time_gap < 10:  # 10ms 内的跨线程操作
                        print(f"⚠️  并发检测: {last_op['op_type']} ({last_op['thread_name']}) → "
                              f"{op_type} ({threading.current_thread().name}) 间隔 {time_gap:.2f}ms")

    @staticmethod
    def analyze():
        """分析操作日志"""
        print("\n" + "=" * 80)
        print("📊 Metal 操作分析")
        print("=" * 80)

        if not metal_ops_log:
            print("无操作记录")
            return

        # 按线程分组
        by_thread = defaultdict(list)
        for entry in metal_ops_log:
            by_thread[entry['thread_name']].append(entry)

        print(f"\n总操作数: {len(metal_ops_log)}")
        print(f"涉及线程: {len(by_thread)}")
        print("")

        for thread_name, ops in sorted(by_thread.items()):
            print(f"📍 {thread_name}: {len(ops)} 次操作")
            op_types = defaultdict(int)
            for op in ops:
                op_types[op['op_type']] += 1
            for op_type, count in sorted(op_types.items()):
                print(f"   - {op_type}: {count}")

        # 检测并发窗口
        print("\n" + "─" * 80)
        print("🔍 并发窗口分析（跨线程操作间隔 < 100ms）")
        print("─" * 80)

        concurrent_windows = []
        for i in range(1, len(metal_ops_log)):
            prev = metal_ops_log[i - 1]
            curr = metal_ops_log[i]

            if prev['thread_id'] != curr['thread_id']:
                gap_ms = (curr['timestamp'] - prev['timestamp']) * 1000
                if gap_ms < 100:
                    concurrent_windows.append({
                        'prev': prev,
                        'curr': curr,
                        'gap_ms': gap_ms
                    })

        if concurrent_windows:
            print(f"\n发现 {len(concurrent_windows)} 个并发窗口:")
            for i, window in enumerate(concurrent_windows[:10], 1):  # 只显示前 10 个
                print(f"\n{i}. 间隔 {window['gap_ms']:.2f}ms")
                print(f"   {window['prev']['thread_name']}: {window['prev']['op_type']}")
                print(f"   {window['curr']['thread_name']}: {window['curr']['op_type']}")
        else:
            print("\n✅ 未发现显著并发窗口")


# Monkey patch MLX operations to track them
def install_metal_hooks():
    """安装 MLX Metal 操作 Hook"""
    try:
        import mlx.core as mx

        # Hook mx.eval
        original_eval = mx.eval
        def tracked_eval(*args, **kwargs):
            MetalOpTracker.log_op('mx.eval', threading.get_ident(),
                                  f"{len(args)} arrays")
            return original_eval(*args, **kwargs)
        mx.eval = tracked_eval

        # Hook mx.synchronize
        original_sync = mx.synchronize
        def tracked_sync(*args, **kwargs):
            stream_name = str(args[0]) if args else "default"
            MetalOpTracker.log_op('mx.synchronize', threading.get_ident(),
                                  stream_name)
            return original_sync(*args, **kwargs)
        mx.synchronize = tracked_sync

        print("✅ Metal Hook 已安装")
        return True

    except Exception as e:
        print(f"❌ Metal Hook 安装失败: {e}")
        return False


async def test_with_tracking():
    """运行测试并追踪 Metal 操作"""
    from omlx.engine.batched import BatchedEngine

    print("=" * 80)
    print("🔍 Metal 深度调试 - 2x64 并发测试")
    print("=" * 80)
    print("")

    model_path = Path.home() / "models" / "qwen3.5-35b-mlx"
    engine = BatchedEngine(
        model_name=str(model_path),
        trust_remote_code=True
    )
    await engine.start()

    try:
        prompts = [
            "Explain Python.",
            "What is TypeScript?"
        ]

        async def generate(prompt, rid):
            tokens = 0
            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=64,
                temperature=0.0
            ):
                if output.new_text:
                    tokens += 1
            return rid, tokens

        print("🚀 开始 2x64 并发测试（带 Metal 追踪）...")
        start = time.time()

        results = await asyncio.gather(
            generate(prompts[0], "R1"),
            generate(prompts[1], "R2"),
            return_exceptions=True
        )

        elapsed = time.time() - start

        success = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Request R{i+1} failed: {result}")
                success = False
            else:
                rid, tokens = result
                print(f"✅ {rid}: {tokens} tokens")

        print(f"\n总时间: {elapsed:.2f}s")

        return success

    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.stop()


async def main():
    # 安装 Hook
    if not install_metal_hooks():
        print("⚠️  无法安装 Hook，跳过追踪")
        return

    print("")

    # 运行测试
    success = await test_with_tracking()

    # 分析结果
    MetalOpTracker.analyze()

    print("\n" + "=" * 80)
    print("🎯 诊断结论")
    print("=" * 80)

    if success:
        print("✅ 测试通过（无 Metal 错误）")
        print("   建议：检查是否有高频并发窗口（< 10ms）")
    else:
        print("❌ 测试失败（Metal 错误）")
        print("   建议：检查并发窗口中的操作序列")


if __name__ == "__main__":
    asyncio.run(main())
