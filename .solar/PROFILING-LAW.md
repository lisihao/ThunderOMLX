# ThunderOMLX 铁律：性能分析框架使用规范

**建立于**: 2026-03-16
**来源**: Processing TPS 优化经验总结
**核心原则**: 基于现有工具扩展，不重复造轮子

---

## 铁律定义

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   🎯 性能分析铁律                                               │
│                                                                 │
│   1. 性能优化前必须先 Profile (MUST)                            │
│   2. 使用统一的 Profiling 框架 (MUST)                           │
│   3. 基于现有工具扩展，不重写 (MUST)                            │
│   4. 数据驱动决策，不凭感觉 (MUST)                              │
│                                                                 │
│   没有数据 = 没有优化依据                                       │
│   重复造轮子 = 浪费时间 + 不一致                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 为什么需要这个铁律

### 问题背景

在 Processing TPS 优化过程中发现：
- ❌ 每次分析都重新写 profiling 代码
- ❌ 不同地方的 timing 代码格式不一致
- ❌ 难以对比优化前后的效果
- ❌ 浪费时间重复实现相同功能

### 解决方案

建立**统一的 Profiling 框架**：
- ✅ 基于 `paged_ssd_cache.py` 的 stats 模式扩展
- ✅ 线程安全、可嵌套、自动统计
- ✅ 一次编写，到处使用
- ✅ 一致的输出格式

---

## 使用规范

### 1. 导入框架

```python
from omlx.profiling import get_global_profiler

profiler = get_global_profiler()
```

### 2. 添加 Timing

**推荐方式（Context Manager）**：

```python
# 单层 timing
with profiler.section("operation_name"):
    do_work()

# 嵌套 timing（自动处理层级关系）
with profiler.section("parent_operation"):
    with profiler.section("parent_operation.child_1"):
        do_work_1()

    with profiler.section("parent_operation.child_2"):
        do_work_2()
```

**手动方式（需要手动匹配）**：

```python
profiler.start("operation_name")
try:
    do_work()
finally:
    profiler.end("operation_name")
```

**直接记录（已知时间）**：

```python
profiler.record("operation_name", elapsed_ms=123.4)
```

### 3. 打印统计

```python
from omlx.profiling import print_profiling_stats

# 在 benchmark 或测试结束后
print_profiling_stats(top_n=30, min_percent=1.0)
```

### 4. 保存数据

```python
import json

stats = profiler.get_stats()

with open('/tmp/profiling_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

---

## 强制使用场景

### 场景 1: 性能优化前

**触发**：准备优化某个模块的性能

**必须**：
1. 先用 Profiling 框架分析现状
2. 识别真正的瓶颈（占比 > 5%）
3. 记录 baseline 数据
4. 基于数据制定优化方案

**禁止**：
- ❌ 凭感觉猜测瓶颈
- ❌ 没有 baseline 就开始优化
- ❌ 优化后不验证效果

**示例**：

```python
# 优化前
profiler = get_global_profiler()

with profiler.section("baseline.total"):
    run_workload()

baseline_stats = profiler.get_stats()
save_stats(baseline_stats, "baseline.json")

# 识别瓶颈
for name, stats in baseline_stats['top_operations']:
    if stats['percent'] > 5.0:
        print(f"瓶颈: {name} ({stats['percent']:.1f}%)")
```

### 场景 2: 性能优化后

**触发**：完成性能优化

**必须**：
1. 用相同的 Profiling 点测试
2. 对比优化前后数据
3. 量化提升百分比
4. 验证没有引入新瓶颈

**示例**：

```python
# 优化后
profiler.reset()  # 清空统计

with profiler.section("optimized.total"):
    run_workload()

optimized_stats = profiler.get_stats()

# 对比
baseline = load_stats("baseline.json")
improvement = (baseline['total_time_ms'] - optimized_stats['total_time_ms']) / baseline['total_time_ms']
print(f"性能提升: {improvement * 100:.1f}%")
```

### 场景 3: 新增性能关键代码

**触发**：编写性能关键路径的代码（Prefill、Generation、Cache 等）

**必须**：
1. 在关键函数中添加 profiling 点
2. 使用层级命名（如 `prefill.layer_0.qkv_proj`）
3. 文档中说明 profiling 点的含义

**示例**：

```python
class Scheduler:
    def _process_prompts(self, ...):
        """Process prefill requests"""

        # ✅ 必须添加 profiling
        with self._profiler.section("prefill.total"):
            with self._profiler.section("prefill.embedding"):
                embeddings = self.embed_tokens(tokens)

            with self._profiler.section("prefill.model_forward"):
                logits = self.model(embeddings)

            with self._profiler.section("prefill.cache_store"):
                self.cache.store(kv_cache)
```

### 场景 4: 调查性能回退

**触发**：发现性能回退（TPS 下降 > 2%）

**必须**：
1. 启用 profiling 重现问题
2. 对比最近的 baseline
3. 识别新增的耗时操作
4. 定位引入回退的 commit

**示例**：

```bash
# 1. 重现问题
export OMLX_ENABLE_PROFILING=true
python benchmark.py > /tmp/current.log

# 2. 对比 baseline
diff <(jq -S . baseline_stats.json) <(jq -S . current_stats.json)

# 3. 识别差异
python analyze_regression.py baseline.json current.json
```

---

## 集成清单

### Scheduler (必须)

**文件**: `src/omlx/scheduler.py`

**必须添加的 profiling 点**：

```python
# __init__
from omlx.profiling import get_global_profiler
self._profiler = get_global_profiler()

# _process_prompts (Prefill)
with self._profiler.section("prefill.total"):
    with self._profiler.section("prefill.embedding"):
        ...
    with self._profiler.section("prefill.model_forward"):
        ...
    with self._profiler.section("prefill.cache_store"):
        ...

# _process_running (Generation)
with self._profiler.section("generation.total"):
    with self._profiler.section("generation.model_forward"):
        ...
    with self._profiler.section("generation.sampling"):
        ...
```

### Cache Manager (已有 stats，需统一)

**文件**: `src/omlx/cache/paged_ssd_cache.py`

**当前**：已有 `_stats` 字典（queue_latency 等）

**迁移**：逐步迁移到统一 profiling 框架

```python
# 替代方案：集成现有 stats
from omlx.profiling import get_global_profiler

class PagedSSDCacheManager:
    def __init__(self, ...):
        self._profiler = get_global_profiler()

    def save_block(self, ...):
        with self._profiler.section("cache.save_block"):
            with self._profiler.section("cache.save_block.serialize"):
                ...
            with self._profiler.section("cache.save_block.write"):
                ...
```

### 模型代码 (可选，深度分析时)

**文件**: `src/omlx/models/*.py`

**何时添加**：需要 layer-level profiling 时

```python
from omlx.profiling import get_global_profiler

class Qwen3Attention:
    def __call__(self, x, mask, cache):
        profiler = get_global_profiler()

        with profiler.section("layer.qkv_proj"):
            q, k, v = self.qkv_proj(x)

        with profiler.section("layer.attention"):
            output = self.attention(q, k, v)

        return output
```

---

## 启用和禁用

### 环境变量控制

```bash
# 启用 profiling（开发/测试环境）
export OMLX_ENABLE_PROFILING=true

# 禁用 profiling（生产环境，默认）
export OMLX_ENABLE_PROFILING=false
# 或者不设置（默认禁用）
```

### 代码中检查

```python
profiler = get_global_profiler()

if profiler.enabled:
    logger.info("🔍 Performance profiling ENABLED")
else:
    logger.info("Performance profiling disabled")
```

### 性能开销

- **禁用时**：几乎 0 开销（只有一次 if 判断）
- **启用时**：每次 timing 约 100-200ns（可忽略）
- **内存**：每个操作约 100 bytes

---

## 输出格式

### 终端输出

```
================================================================================
Performance Profiling Results
================================================================================
Total Time: 7653.2 ms

Operation                                          Count    Avg (ms)   Total (ms)      %
--------------------------------------------------------------------------------
prefill.total                                          1    7653.2       7653.2  100.0%
prefill.model_forward                                  1    7400.5       7400.5   96.7%
prefill.model_forward.layer_0.qkv_proj                 1      40.2         40.2    0.5%
prefill.model_forward.layer_0.attention                1      60.3         60.3    0.8%
prefill.model_forward.layer_0.ffn                      1      25.1         25.1    0.3%
...
prefill.embedding                                      1     100.3        100.3    1.3%
prefill.cache_store                                    1     100.2        100.2    1.3%
prefill.synchronize                                    1      50.1         50.1    0.7%
================================================================================
```

### JSON 输出

```json
{
  "total_time_ms": 7653.2,
  "operations": {
    "prefill.total": {
      "count": 1,
      "total_ms": 7653.2,
      "avg_ms": 7653.2,
      "min_ms": 7653.2,
      "max_ms": 7653.2,
      "percent": 100.0
    },
    "prefill.model_forward": {
      "count": 1,
      "total_ms": 7400.5,
      "avg_ms": 7400.5,
      "min_ms": 7400.5,
      "max_ms": 7400.5,
      "percent": 96.7
    },
    ...
  },
  "top_operations": [
    ["prefill.total", {...}],
    ["prefill.model_forward", {...}],
    ...
  ]
}
```

---

## 命名规范

### 层级命名

使用 `.` 分隔层级：

```
顶层.子层级.具体操作
```

**示例**：
```
prefill.total
prefill.embedding
prefill.model_forward
prefill.model_forward.layer_0
prefill.model_forward.layer_0.qkv_proj
prefill.model_forward.layer_0.attention
prefill.model_forward.layer_0.ffn
prefill.cache_store
prefill.synchronize
```

### 命名约定

- **小写 + 下划线**：`model_forward`，不是 `modelForward`
- **动词开头**（操作）：`encode_tokens`，不是 `token_encoding`
- **名词开头**（模块）：`layer_0.qkv_proj`
- **避免缩写**：`attention`，不是 `attn`（除非是通用缩写）

---

## 分析方法

### 1. 识别瓶颈

```python
stats = profiler.get_stats()

bottlenecks = [
    (name, op_stats)
    for name, op_stats in stats['operations'].items()
    if op_stats['percent'] > 5.0  # 占比 > 5%
]

bottlenecks.sort(key=lambda x: x[1]['total_ms'], reverse=True)

for name, op_stats in bottlenecks:
    print(f"瓶颈: {name}")
    print(f"  时间: {op_stats['avg_ms']:.2f}ms ({op_stats['percent']:.1f}%)")
```

### 2. Before/After 对比

```python
import json

# 加载 baseline
with open('baseline_stats.json') as f:
    baseline = json.load(f)

# 加载 optimized
with open('optimized_stats.json') as f:
    optimized = json.load(f)

# 对比每个操作
for name in baseline['operations']:
    if name not in optimized['operations']:
        continue

    baseline_time = baseline['operations'][name]['avg_ms']
    optimized_time = optimized['operations'][name]['avg_ms']

    improvement = (baseline_time - optimized_time) / baseline_time * 100

    if abs(improvement) > 2.0:  # 变化 > 2%
        symbol = "🚀" if improvement > 0 else "⚠️"
        print(f"{symbol} {name}: {improvement:+.1f}%")
```

### 3. 生成优化建议

```python
def suggest_optimizations(stats):
    """基于 profiling 数据生成优化建议"""
    suggestions = []

    for name, op_stats in stats['top_operations'][:5]:
        if op_stats['percent'] < 5.0:
            break

        if 'qkv_proj' in name:
            suggestions.append({
                'operation': name,
                'issue': 'QKV projection 是瓶颈',
                'suggestions': [
                    'Fused QKV Projection',
                    '批量 bfloat16 eval'
                ]
            })

        elif 'attention' in name:
            suggestions.append({
                'operation': name,
                'issue': 'Attention 是瓶颈',
                'suggestions': [
                    'Chunked Prefill',
                    'FlashAttention-3 升级'
                ]
            })

        elif 'ffn' in name:
            suggestions.append({
                'operation': name,
                'issue': 'FFN 是瓶颈',
                'suggestions': [
                    'MoE Expert 并行化',
                    'Activation Fusion'
                ]
            })

    return suggestions
```

---

## 违反后果

```
违反 = 盲目优化 = 浪费时间在非瓶颈上
违反 = 无法量化效果 = 不知道是否真的优化了
违反 = 重复造轮子 = 不一致的代码风格
违反 = 性能回退难定位 = 调试时间倍增
```

---

## 参考文档

- **框架实现**: `src/omlx/profiling.py`
- **集成指南**: `.solar/profiling-framework-integration-guide.md`
- **使用示例**: `profile_prefill_with_instrumentation.py`
- **深度分析报告**: `.solar/prefill-performance-deep-analysis.md`

---

## 铁律总结

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   🎯 性能分析铁律                                               │
│                                                                 │
│   优化前：Profile → 识别瓶颈 → 数据驱动决策                     │
│   优化中：使用统一框架 → 层级 timing → 详细记录                 │
│   优化后：对比数据 → 量化提升 → 验证无回退                      │
│                                                                 │
│   工具：omlx.profiling.PerformanceProfiler                     │
│   启用：export OMLX_ENABLE_PROFILING=true                      │
│   输出：终端表格 + JSON 文件                                    │
│                                                                 │
│   没有 Profile = 没有优化权限                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Profiling Law v1.0*
*建立于: 2026-03-16*
*来源: Processing TPS 优化经验 + paged_ssd_cache stats 扩展*
*强制执行: 所有性能优化工作*
