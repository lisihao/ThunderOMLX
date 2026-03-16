#!/usr/bin/env python3
"""
Add performance monitoring to ThunderOMLX engine to find pp8192 bottleneck.
"""
import sys
from pathlib import Path

# Find the main generation loop
engine_path = Path("src/omlx/engine/batched_engine.py")

if not engine_path.exists():
    print(f"❌ {engine_path} not found")
    sys.exit(1)

print("📍 找到的关键文件：")
print(f"  - {engine_path}")

# Read the file
content = engine_path.read_text()

# Check if monitoring is already added
if "PERF_MONITOR" in content:
    print("\n✅ Performance monitoring already added")
else:
    print("\n💡 建议手动添加性能监控代码到以下位置：")
    print("  - src/omlx/engine/batched_engine.py")
    print("  - 在 generate_step() 或类似函数中")
    print("")
    print("添加的代码示例：")
    print("""
import time

class PerfMonitor:
    def __init__(self):
        self.timings = {}
    
    def record(self, name, duration):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def report(self):
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            total = sum(times)
            print(f"  {name}: {avg*1000:.2f}ms avg, {total:.2f}s total ({len(times)} calls)")

# In generate loop:
perf = PerfMonitor()

# Before each step:
t0 = time.perf_counter()
# ... operation ...
perf.record("operation_name", time.perf_counter() - t0)

# At end:
perf.report()
""")

print("\n或者使用更简单的方法：")
print("使用 mlx-lm 的 benchmark 工具测试 oMLX 直接加载模型的性能")
