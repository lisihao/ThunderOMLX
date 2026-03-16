# Task #14 Phase 4: Prefill 深度优化计划

**日期**: 2026-03-15
**状态**: 🔄 进行中
**目标**: 从 704 tok/s 提升到 880 tok/s (21% gap)

---

## 执行摘要

**Phase 1-3 回顾**：
- ✅ Chunked Prefill 实现
- ✅ Warmup 优化
- ❌ 锁优化（失败，-2.6%，已回滚）

**当前性能**：
- TTFT: 11631.3ms
- Prefill TPS: 704.3 tok/s
- Generation TPS: ~77 tok/s

**性能差距**：
- vs MLX baseline (880 tok/s): **-20%**
- 需要优化: **~2.2s**

---

## 专家会审结论

### 稳健派 (Gemini 2.5 Pro) - 系统化方案

| 优化方向 | 预期收益 | 风险 | 优先级 |
|----------|----------|------|--------|
| **Profiling 定位 1.5s 开销** | 0.5-1.2s | 低 | P0 ⭐ |
| **模型量化 (4-bit)** | 2.0-3.0s | 中 | P1 |
| **异步 Cache I/O** | 0.8-1.5s | 中 | P1 |
| **Chunk Size 调优** | 0.4-0.8s | 低 | P0 ⭐ |

### 探索派 (Gemini 3 Pro) - 创新思路

| 优化方向 | 预期收益 | 风险 | 置信度 |
|----------|----------|------|--------|
| **动态 Token 剪枝** | ~2.0s | 高 | 0.4 |
| **极限 Kernel Fusion** | 2.0-2.5s | 高 | 0.6 |
| **语义级 KV Cache 检索** | 巨大 | 低 | 0.8 ⭐ |

---

## Phase 4 执行计划

### **Stage 1: 快速验证（1-2 天）** 🔄 **进行中**

#### 1.1 深度 Profiling ⚡ **执行中**

**目标**：定位 1.5s "其他开销" 的具体来源

**方法**：
- Python 侧：cProfile 生成 tottime/cumulative 报告
- GPU 侧：（待定）Xcode Instruments Metal System Trace

**脚本**：`profile_prefill_detailed.py`
**任务 ID**：bcc724a（后台运行）

**输出**：
- `/tmp/prefill_profile_cumulative.txt`（按总时间排序）
- `/tmp/prefill_profile_tottime.txt`（按自身时间排序）

**预期发现**：
- Tensor 拷贝开销？
- Python 循环开销？
- GC 开销？
- 意外的函数调用？

**成功标准**：
- 识别出至少 0.5s 的可优化开销
- 有明确的优化方向

---

#### 1.2 Chunk Size 扫描 ⏳ **就绪**

**目标**：找到最优 chunk size（当前默认 512）

**方法**：
- 测试 chunk_size: 256, 384, 512, 768, 1024, 2048
- 测量 TTFT 和 Prefill TPS
- 绘制性能曲线

**脚本**：`scan_chunk_size.py`

**预期**：
- 找到比 512 更优的配置
- 收益预估：0.4-0.8s

**成功标准**：
- 相比 512，TPS 提升 > 5%

---

### **Stage 2: 中等风险优化（3-5 天）** ⏳ **待启动**

#### 2.1 异步 Cache I/O

**理由**：
- 当前 Cache 开销 2s，大部分是 SSD I/O
- 可与 GPU 计算重叠

**实施**：
```python
# 创建 I/O 线程池
executor = ThreadPoolExecutor(max_workers=2)

# 在计算 chunk N 时，预取 chunk N+1 的 KV Cache
def prefill_chunk(chunk_id):
    # 启动下一个 chunk 的预取
    future = executor.submit(cache.prefetch, chunk_id + 1)

    # 计算当前 chunk
    output = model.forward(chunk_id)

    # 等待预取完成（如果需要）
    future.result()
```

**预期收益**：0.8-1.5s

---

#### 2.2 语义级 Prompt Caching

**前置条件**：分析真实请求的前缀重复率

**步骤**：
1. 统计请求前缀重复率
2. 如果 > 40%，实现 Radix Tree 精确匹配
3. 对于重复前缀，直接复用 KV Cache

**预期收益**：
- 对于重复前缀：Forward 时间趋近 0
- 对于新请求：无影响

---

### **Stage 3: 激进优化（需深入评估）** ⏳ **研究中**

#### 3.1 模型量化 (4-bit)

**预期收益**：2.0-3.0s（最大）
**风险**：模型精度可能下降

**决策流程**：
1. 量化模型：`mlx_lm.convert --quantize -q-bits 4`
2. 性能测试：对比 TTFT/TPS
3. 精度评估：Perplexity, MMLU
4. 决策：
   - 精度损失 < 2% → 采用
   - 精度损失 > 5% → 放弃

---

#### 3.2 极限 Kernel Fusion

**前置条件**：Profiling 显示内存带宽是瓶颈

**步骤**：
1. Nsight Systems 分析：`nsys profile python benchmark.py`
2. 识别 Memory-bound 算子
3. 评估 Triton/CUDA 融合可行性
4. 风险评估：寄存器溢出可能性

**预期收益**：2.0-2.5s
**风险**：极高（寄存器压力导致性能下降）

---

## 里程碑与成功标准

### Milestone 1: Stage 1 完成（1-2 天）

**交付物**：
- ✅ Profiling 报告（识别 1.5s 开销）
- ✅ 最优 Chunk Size（配置文件）

**成功标准**：
- 识别至少 1.0s 的可优化空间
- Chunk Size 调优后 TPS > 720 tok/s

---

### Milestone 2: Stage 2 完成（3-5 天）

**交付物**：
- ✅ 异步 Cache I/O 实现
- ✅ Prompt Caching 可行性报告

**成功标准**：
- Prefill TPS > 800 tok/s
- Cache 时间 < 1.0s

---

### Milestone 3: 最终目标（7-10 天）

**成功标准**：
- **Prefill TPS ≥ 880 tok/s** ⭐
- TTFT < 10.0s
- 无精度损失（或 < 2%）

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 1.5s 开销无法优化 | 中 | 中 | 转向其他方向（量化、异步 I/O） |
| Chunk Size 调优无收益 | 低 | 低 | 损失 1 天时间，影响有限 |
| 异步 I/O 引入 bug | 中 | 高 | 充分测试，保留回滚路径 |
| 模型量化精度损失大 | 中 | 高 | 先评估再决策，可放弃 |

---

## 当前进度

**日期**: 2026-03-15 23:55
**状态**: Stage 1.1 进行中

- ✅ 专家会审完成
- 🔄 Profiling 运行中（任务 bcc724a）
- ✅ Chunk Size 扫描脚本就绪
- ⏳ 等待 Profiling 结果

**下一步**：
1. 分析 Profiling 报告
2. 运行 Chunk Size 扫描
3. 基于结果决定 Stage 2 优先级

---

*负责人: Solar (战略家模式)*
*审核: 稳健派 (Gemini 2.5 Pro) + 探索派 (Gemini 3 Pro)*
