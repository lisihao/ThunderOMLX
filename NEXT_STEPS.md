# ThunderOMLX 性能优化 - 下一步行动计划

**日期**: 2026-03-15
**当前状态**: Generation 性能分析已完成
**关键发现**: ThunderOMLX 调度层优秀（0.5% 开销），但 Benchmark 比直接测试慢 18.5%

---

## 📊 当前性能状态

| 测试方式 | TPS | TPOT | vs Native MLX |
|----------|-----|------|---------------|
| **Native MLX Baseline** | 80.1 tok/s | 12.48 ms | 100% |
| **ThunderOMLX 直接测试** | 79.8 tok/s | 12.52 ms | 99.6% ✅ |
| **ThunderOMLX Benchmark** | 65.1 tok/s | 15.47 ms | 81.3% ⚠️ |

**核心问题**: 为什么 Benchmark 比直接测试慢 14.7 tok/s (18.5%)？

---

## 🎯 优先级排序

### P0 - 立即行动（高影响，待确认）

#### 1. 确认 Benchmark vs 直接测试差异根因

**目标**: 找出 18.5% 性能差距的真正原因

**方法**:
```bash
# Step 1: 在 Benchmark 中添加相同的性能分析
编辑 src/omlx/admin/benchmark.py
添加与 test_profiling.py 相同的性能打印

# Step 2: 运行 Benchmark 并收集数据
运行 Admin Panel benchmark
对比 step time 分布

# Step 3: 逐步对比
- 单请求 vs 并发请求
- 有/无 ContextPilot
- HTTP/SSE vs 直接调用
- 长上下文 vs 短上下文
```

**预期发现**:
- 如果是 API 层：step time 相似，但 E2E latency 高
- 如果是并发：batch_gen time 变高
- 如果是 ContextPilot：schedule_waiting 或 process_responses 变高
- 如果是 KV cache：batch_gen time 在长上下文下变高

**成功标准**:
- 定位到具体的性能瓶颈（API/并发/ContextPilot/KV cache）
- 量化各部分开销（X ms from API, Y ms from ContextPilot...）

---

### P1 - 短期优化（已知问题）

#### 2. 修复内存泄漏

**问题**: Benchmark 运行后内存不释放（19GB → 38GB）

**位置**: `src/omlx/admin/benchmark.py` - run_benchmark()

**修复**:
```python
async def run_benchmark(run: BenchmarkRun, engine_pool: Any) -> None:
    # ... 运行测试 ...

    # 修复：显式清理内存
    import mlx.core as mx
    import gc

    mx.clear_cache()  # 清理 MLX Metal 缓存
    gc.collect()      # 强制 Python GC
    await asyncio.sleep(1)  # 等待内存释放
```

**验证**:
- 运行两次 benchmark，检查内存是否仍翻倍
- 确认第二次运行性能不受影响

**影响**: ~3% 性能提升 (62.8 → 65.1 tok/s)

---

#### 3. 优化 API 层（如果确认是瓶颈）

**前提**: 完成 P0.1，确认 API 层有显著开销

**可能优化**:
- 减少序列化/反序列化次数
- 使用更快的序列化库（orjson）
- 减少中间拷贝
- 考虑 binary protocol (msgpack)

**目标**: 减少 1-2 ms/tok 开销

---

### P2 - 中期优化（性能提升）

#### 4. 优化 ContextPilot（如果有开销）

**前提**: Profile 显示 ContextPilot 有可测量的开销

**方法**:
- 只在必要时判断（prompt 边界）
- 缓存判断结果
- 优化判断逻辑

**目标**: 减少 0.5-1 ms/tok 开销

---

#### 5. 优化长上下文 KV Cache 加载

**目标**: 优化 pp8192 vs pp1024 的性能差距（16%）

**可能方案**:
1. **批量预加载相邻 blocks**
   - 当前：按需加载单个 block
   - 优化：预测并预加载下一个 block

2. **优化 block 索引结构**
   - 当前：可能是 list 查找
   - 优化：使用 hash map 或更快的数据结构

3. **缓存热点 block 索引**
   - 避免重复查找相同 block 的位置

4. **测试 Flash Attention**
   - 如果 MLX 支持，启用 Flash Attention
   - 可能减少 attention 计算时间

**验证**:
- 对比优化前后的 pp8192 vs pp1024 性能
- 目标：将 pp8192 从 65.1 提升到 70+ tok/s

---

### P3 - 长期优化（边际收益）

#### 6. 微优化 ThunderOMLX 调度层

**当前状态**: 已经很优秀（0.5% 开销）

**可能优化**:
- 延迟更新索引
- 批量处理 cache 操作
- 减少函数调用开销

**目标**: 减少 0.01-0.02 ms/tok 开销
**投入产出比**: 低（优化空间很小）

---

## 📅 执行时间线

### 第 1 周：根因分析

- [ ] Day 1-2: 在 Benchmark 中添加性能分析，收集数据
- [ ] Day 3-4: 对比分析，定位瓶颈
- [ ] Day 5: 修复内存泄漏，验证效果
- [ ] Day 6-7: 制定优化方案

### 第 2 周：优化实施

- [ ] Day 8-10: 实施优化（根据 P0 的发现）
- [ ] Day 11-12: 测试验证
- [ ] Day 13-14: 性能回归测试，文档更新

### 第 3 周：长期优化

- [ ] Day 15-18: P2 优化（KV cache, ContextPilot）
- [ ] Day 19-21: 性能测试，对比分析

---

## 🎯 性能目标

### 近期目标（1-2 周）

**Benchmark pp8192/tg128**:
- 当前：65.1 tok/s (15.47 ms/tok)
- 目标：70+ tok/s (< 14.3 ms/tok)
- 提升：7.5% (+4.9 tok/s)

**Benchmark pp1024/tg128**:
- 当前：78.5 tok/s (12.84 ms/tok)
- 目标：82+ tok/s (< 12.2 ms/tok)
- 提升：4.5% (+3.5 tok/s)

### 中期目标（1 个月）

**Benchmark pp8192/tg128**:
- 目标：75+ tok/s (< 13.3 ms/tok)
- vs Native MLX: 93.6%
- 提升：15% (+9.9 tok/s)

### 长期目标（3 个月）

**Benchmark pp8192/tg128**:
- 目标：78+ tok/s (< 12.8 ms/tok)
- vs Native MLX: 97.4%
- 提升：20% (+12.9 tok/s)

---

## 🔧 技术实现细节

### 如何在 Benchmark 中添加性能分析

**文件**: `src/omlx/admin/benchmark.py`

**方法 1**: 复用现有的 scheduler profiling
```python
# Benchmark 已经使用了 scheduler.step()
# 现有的性能分析代码会自动生效
# 只需要收集日志输出

async def run_benchmark(...):
    # ... 现有代码 ...

    # 运行测试时，scheduler.step() 会自动打印性能数据
    # 收集并解析这些数据
```

**方法 2**: 在 Benchmark 层添加额外分析
```python
async def _run_single_test(...):
    start_time = time.perf_counter()

    # ... 运行测试 ...

    end_time = time.perf_counter()

    # 收集并打印详细的性能分解
    print(f"API overhead: {api_time:.2f}ms")
    print(f"Engine overhead: {engine_time:.2f}ms")
```

---

## 📊 验证标准

### 每次优化后必须验证

1. **性能回归测试**:
   ```bash
   # pp8192/tg128
   python3 run_admin_benchmark.py

   # pp1024/tg128
   python3 run_admin_benchmark.py --pp 1024

   # 对比优化前后数据
   ```

2. **功能正确性**:
   - 运行完整测试套件
   - 验证输出质量
   - 检查内存使用

3. **稳定性**:
   - 连续运行 10 次，检查波动
   - 长时间运行（1000+ tokens）
   - 并发测试

---

## 💡 关键假设与风险

### 假设

1. **Benchmark 慢的主要原因是 API 层开销**
   - 如果不是，需要重新分析

2. **优化 API 层可以带来显著提升**
   - 如果提升 < 5%，可能不值得优化

3. **KV cache 优化可以缩小长上下文差距**
   - 需要验证是否真的是 cache 加载慢

### 风险

1. **优化可能引入 bug**
   - 缓解：充分测试

2. **性能提升可能不如预期**
   - 缓解：先 profile，后优化

3. **优化可能影响功能**
   - 缓解：保持向后兼容

---

## 📚 参考资料

### 已完成的分析

1. `PERFORMANCE_BOTTLENECK_ANALYSIS.md` - 初始分析
2. `PROFILING_RESULTS.md` - 详细性能数据
3. `PROFILING_SUMMARY.md` - 执行总结
4. `PERFORMANCE_VISUALIZATION.md` - 可视化报告

### 技术文档

1. MLX-LM BatchGenerator: `mlx_lm/generate.py`
2. ThunderOMLX Scheduler: `src/omlx/scheduler.py`
3. Admin Benchmark: `src/omlx/admin/benchmark.py`

---

## ✅ 决策检查点

### 在开始优化前，回答以下问题：

- [ ] 我们确认了瓶颈在哪里吗？（P0.1）
- [ ] 优化的投入产出比是多少？
- [ ] 有没有更简单的方法达到目标？
- [ ] 优化后如何验证效果？
- [ ] 如果优化失败，有回滚方案吗？

### 在优化完成后，验证以下内容：

- [ ] 性能是否提升？提升了多少？
- [ ] 功能是否正常？有没有引入 bug？
- [ ] 性能是否稳定？波动范围多大？
- [ ] 内存使用是否正常？有没有泄漏？
- [ ] 代码是否可维护？有没有增加复杂度？

---

*行动计划生成时间: 2026-03-15 21:15*
*负责人: Claude Sonnet 4.5*
*审核人: 监护人昊哥*
