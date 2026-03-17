# Profiling 框架集成指南

**日期**: 2026-03-16
**基于**: paged_ssd_cache.py 中现有的 stats 框架扩展

---

## 📦 新增文件

### 1. `src/omlx/profiling.py`

**通用性能分析框架**，提供：
- 线程安全的 timing 统计
- Context manager 接口
- 嵌套 timing 支持
- 自动百分比计算
- 最小开销（可禁用）

**使用方式**:

```python
from omlx.profiling import get_global_profiler

profiler = get_global_profiler()

# 方式1: Context manager（推荐）
with profiler.section("operation_name"):
    do_work()

# 方式2: 手动
profiler.start("operation_name")
do_work()
profiler.end("operation_name")

# 方式3: 直接记录
profiler.record("operation_name", elapsed_ms=123.4)
```

---

## 🔧 集成到 Scheduler（获取真实 Prefill 数据）

### Step 1: 在 scheduler.py 中启用 Profiling

**文件**: `src/omlx/scheduler.py`

**修改位置**: `__init__` 方法（约 line 1100）

```python
# 添加导入
from omlx.profiling import get_global_profiler

class Scheduler:
    def __init__(self, config: SchedulerConfig):
        # ... 现有代码 ...

        # 添加 profiler（只在启用时才有开销）
        self._profiler = get_global_profiler()
```

### Step 2: 注入 Prefill Timing

**文件**: `src/omlx/scheduler.py`

**修改位置**: `_process_prompts` 方法（约 line 331-720）

```python
def _process_prompts(self, seq_group_metadata: List[SequenceGroupMetadata]):
    """Process new prompts (prefill phase)."""

    # 开始 Prefill timing
    with self._profiler.section("prefill.total"):
        # ===== Token Embedding =====
        with self._profiler.section("prefill.embedding"):
            # 现有的 embedding 代码
            ...

        # ===== Layer-by-layer Forward =====
        # 在模型 forward 调用周围添加 timing
        with self._profiler.section("prefill.model_forward"):
            # 现有的模型 forward 代码
            ...

        # ===== Cache Operations =====
        with self._profiler.section("prefill.cache_ops"):
            # 现有的 cache 操作代码
            ...

        # ===== Synchronize =====
        with self._profiler.section("prefill.synchronize"):
            mx.synchronize()
```

### Step 3: 更详细的 Layer-level Timing（可选）

如果需要更细粒度的分析，可以在模型代码中添加：

**文件**: `src/omlx/models/qwen3.py`（或其他模型文件）

```python
from omlx.profiling import get_global_profiler

class Qwen3Attention:
    def __call__(self, x, mask, cache):
        profiler = get_global_profiler()

        # QKV Projection
        with profiler.section("layer.qkv_proj"):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            mx.eval(q, k, v)

        # Attention
        with profiler.section("layer.attention"):
            output = self.attention(q, k, v, mask, cache)
            mx.eval(output)

        return output

class Qwen3MLP:
    def __call__(self, x):
        profiler = get_global_profiler()

        with profiler.section("layer.ffn"):
            # MoE 或标准 FFN
            output = self.forward(x)
            mx.eval(output)

        return output
```

### Step 4: 打印统计（测试时）

**文件**: 在 benchmark 或测试脚本中

```python
from omlx.profiling import print_profiling_stats, get_global_profiler

# 执行 benchmark
run_benchmark()

# 打印统计
print_profiling_stats(top_n=30, min_percent=1.0)

# 或者获取详细数据
profiler = get_global_profiler()
stats = profiler.get_stats()

# 保存到文件
import json
with open('/tmp/profiling_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

---

## 🎯 快速验证（不修改 scheduler）

如果不想立即修改 scheduler.py，可以先用示例脚本验证框架：

```bash
# 1. 运行示例（当前是 mock 数据）
python3 profile_prefill_with_instrumentation.py \
  --length 8192 \
  --trials 3

# 2. 查看输出
# 会显示各个阶段的 timing 统计
```

**预期输出**:
```
================================================================================
Performance Profiling Results
================================================================================
Total Time: 12345.6 ms

Operation                                          Count    Avg (ms)   Total (ms)      %
--------------------------------------------------------------------------------
3.prefill_trial_0                                      1    4115.2       4115.2   33.3%
3.prefill_trial_0.layer_0.qkv_proj                     1      15.3         15.3    0.1%
3.prefill_trial_0.layer_0.attention                    1      25.7         25.7    0.2%
...
```

---

## 🔍 真实数据采集计划

### Phase 1: 最小侵入式集成（1 小时）

只在 scheduler.py 的顶层添加 timing：

```python
# scheduler.py: _process_prompts 方法
with self._profiler.section("prefill.total"):
    # 整个 prefill 流程
    ...
```

**收益**：
- 验证框架工作正常
- 获取端到端 Prefill 时间
- 无需修改模型代码

### Phase 2: 分层 Timing（2-3 小时）

在 scheduler.py 中添加更详细的 timing：

```python
with self._profiler.section("prefill.embedding"):
    ...

with self._profiler.section("prefill.model_forward"):
    ...

with self._profiler.section("prefill.cache_ops"):
    ...
```

**收益**：
- 识别 Prefill 内部的瓶颈（embedding vs forward vs cache）
- 指导优化方向

### Phase 3: Layer-level Timing（1 天）

修改模型代码，添加 layer-level timing：

```python
# models/qwen3.py
with profiler.section(f"layer_{layer_idx}.qkv_proj"):
    ...

with profiler.section(f"layer_{layer_idx}.attention"):
    ...

with profiler.section(f"layer_{layer_idx}.ffn"):
    ...
```

**收益**：
- 精确定位瓶颈（QKV vs Attention vs FFN）
- 量化各组件的时间占比
- 验证优化效果

---

## 📊 数据分析流程

### 1. 运行带 Profiling 的 Benchmark

```bash
# 启用 profiling
export OMLX_ENABLE_PROFILING=true

# 运行 benchmark
python3 benchmark_chunked_prefill.py \
  --length 8192 \
  --trials 5 \
  2>&1 | tee /tmp/profiling_run.log
```

### 2. 保存统计数据

```python
# 在 benchmark 脚本末尾添加
from omlx.profiling import get_global_profiler
import json

profiler = get_global_profiler()
stats = profiler.get_stats()

with open('/tmp/prefill_profiling_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

### 3. 分析数据

```python
import json

with open('/tmp/prefill_profiling_stats.json') as f:
    stats = json.load(f)

# 找出瓶颈（占比 > 5%）
bottlenecks = [
    (name, op_stats)
    for name, op_stats in stats['operations'].items()
    if op_stats['percent'] > 5.0
]

# 按时间排序
bottlenecks.sort(key=lambda x: x[1]['total_ms'], reverse=True)

for name, op_stats in bottlenecks:
    print(f"{name}: {op_stats['avg_ms']:.2f}ms ({op_stats['percent']:.1f}%)")
```

---

## 🚀 优先级推荐

**立即执行（今天）**:
1. ✅ 验证 profiling 框架正常工作（运行示例脚本）
2. ✅ Phase 1 集成（scheduler.py 顶层 timing）

**本周执行**:
3. ✅ Phase 2 集成（分层 timing）
4. ✅ 运行真实 benchmark，收集数据

**下周执行**:
5. ✅ Phase 3 集成（layer-level timing，如果需要）
6. ✅ 基于数据优化瓶颈

---

## 📝 代码示例（完整集成）

### scheduler.py 集成示例

```python
# src/omlx/scheduler.py

from omlx.profiling import get_global_profiler

class Scheduler:
    def __init__(self, config: SchedulerConfig):
        # ... 现有代码 ...

        # 添加 profiler
        self._profiler = get_global_profiler()

        # 在日志中提示 profiling 状态
        if self._profiler.enabled:
            logger.info("🔍 Performance profiling ENABLED")

    def _process_prompts(self, seq_group_metadata: List[SequenceGroupMetadata]):
        """Process new prompts (prefill phase)."""

        # 整体 prefill timing
        with self._profiler.section("prefill"):
            # === Prepare inputs ===
            with self._profiler.section("prefill.prepare_inputs"):
                # ... 现有代码 ...
                pass

            # === Model forward ===
            with self._profiler.section("prefill.model_forward"):
                # ... 现有的模型 forward 调用 ...
                pass

            # === Cache operations ===
            with self._profiler.section("prefill.cache_store"):
                # ... 现有的 cache 操作 ...
                pass

            # === Synchronize ===
            with self._profiler.section("prefill.synchronize"):
                mx.synchronize()

    def get_profiling_stats(self) -> Dict:
        """获取 profiling 统计（供 admin API 使用）"""
        return self._profiler.get_stats()
```

### Admin API 集成（可选）

```python
# src/omlx/admin/api.py

@app.get("/admin/profiling")
async def get_profiling_stats():
    """获取性能分析统计"""
    from omlx.profiling import get_global_profiler

    profiler = get_global_profiler()
    stats = profiler.get_stats(top_n=50)

    return {
        "enabled": profiler.enabled,
        "stats": stats
    }

@app.post("/admin/profiling/reset")
async def reset_profiling_stats():
    """重置性能分析统计"""
    from omlx.profiling import reset_global_profiler

    reset_global_profiler()

    return {"message": "Profiling stats reset"}
```

---

## ⚠️ 注意事项

1. **性能开销**:
   - Profiling 禁用时开销几乎为 0
   - 启用时每次 timing 约 100-200ns（可忽略）
   - 生产环境默认禁用

2. **线程安全**:
   - 框架已处理线程安全
   - 可以在多线程环境中使用

3. **嵌套 Timing**:
   - 支持嵌套，自动计算百分比
   - 避免重复计算（只计算顶层操作）

4. **内存占用**:
   - 每个操作约 100 bytes
   - 1000 个操作约 100KB
   - 可定期 reset

---

## 📈 预期成果

完成集成后，可以获得：

1. **端到端 Prefill 时间分解**:
   ```
   prefill.total: 7653ms (100%)
   ├─ prefill.embedding: 100ms (1.3%)
   ├─ prefill.model_forward: 7400ms (96.7%)
   ├─ prefill.cache_store: 100ms (1.3%)
   └─ prefill.synchronize: 50ms (0.7%)
   ```

2. **Layer-level 时间分解** (Phase 3):
   ```
   layer_0.qkv_proj: 40ms
   layer_0.attention: 60ms
   layer_0.ffn: 25ms
   layer_1.qkv_proj: 40ms
   ...
   ```

3. **瓶颈识别**:
   - 量化各组件时间占比
   - 排序优先级
   - 生成优化建议

4. **优化效果验证**:
   - Before/After 对比
   - 量化提升百分比
   - 识别新瓶颈

---

*Integration Guide v1.0*
*基于: paged_ssd_cache.py stats 框架*
*创建时间: 2026-03-16*
