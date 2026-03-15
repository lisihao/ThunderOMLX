# Metal System Trace Profiling 操作指南

> **目标**: 明确 MLX 生成 1 token 需要 223ms 的具体瓶颈分布

---

## 准备工作

### 1. 确保 Xcode 已安装

```bash
xcode-select -p  # 应该显示 Xcode 路径
```

如果未安装，从 App Store 安装 Xcode。

### 2. 确保测试脚本就绪

```bash
cd /Users/lisihao/ThunderOMLX
ls -la test_metal_profiling.py  # 应该存在
```

---

## 执行 Profiling

### Step 1: 打开 Instruments

```bash
open -a "Instruments"
```

或者从 Xcode 菜单: `Xcode → Open Developer Tool → Instruments`

### Step 2: 选择 "Metal System Trace" 模板

1. 在 Instruments 启动窗口中
2. 搜索 "Metal System Trace"
3. 双击打开

### Step 3: 配置目标

1. 点击顶部的 "Choose Target" 按钮
2. 选择 "Python" (或 "python3")
3. 在 "Arguments" 中填写:
   ```
   /Users/lisihao/ThunderOMLX/test_metal_profiling.py
   ```
4. 在 "Environment" 中添加:
   ```
   PYTHONPATH=/Users/lisihao/ThunderOMLX/src
   ```

### Step 4: 开始录制

1. 点击红色的 "Record" 按钮
2. 等待脚本启动（会有 warm up 阶段）
3. 看到 "🔴 开始录制！" 提示时，profiling 正式开始
4. 等待脚本运行完成（20 次生成，约 5 秒）
5. 看到 "⚠️ 现在可以停止 Instruments 录制" 时，点击 "Stop" 按钮

### Step 5: 保存录制文件

1. File → Save
2. 保存为: `mlx_profiling_$(date +%Y%m%d_%H%M%S).trace`
3. 位置: `/Users/lisihao/ThunderOMLX/profiling/`

---

## 分析数据

### 关键指标

#### 1. GPU Utilization（GPU 利用率）

**查看位置**: Timeline → GPU → Utilization

**关键问题**:
- [ ] GPU 利用率是否长时间低于 50%？
- [ ] 是否有频繁的"空闲"时段？
- [ ] 空闲时段是否对应 CPU 活动？

**推测瓶颈**:
- 如果 GPU 频繁空闲 → **CPU-GPU 同步问题**或**Python 调度开销**
- 如果 GPU 持续满载 → **GPU 计算瓶颈**或**内存带宽瓶颈**

#### 2. Kernel Duration & Count（内核时长和数量）

**查看位置**: Timeline → Metal → Kernels

**关键问题**:
- [ ] 平均每次生成调用了多少个 Metal 内核？
- [ ] 内核平均时长是多少？
- [ ] 是否有大量 <1ms 的短时内核？

**推测瓶颈**:
- 如果内核数量 >50 → **内核启动开销过大**，需要内核融合
- 如果大量 <1ms 内核 → **启动开销 > 计算时间**，严重低效

#### 3. Memory Bandwidth（内存带宽）

**查看位置**: Timeline → Metal → Memory

**关键问题**:
- [ ] 内存带宽利用率是多少？
- [ ] 是否有大量小数据传输？
- [ ] CPU-GPU 数据传输频率？

**推测瓶颈**:
- 如果带宽利用率低 → **访问模式不友好**，需要优化 KV Cache 布局
- 如果频繁小数据传输 → **CPU-GPU 往返过多**，需要 GPU 采样器

#### 4. CPU Activity（CPU 活动）

**查看位置**: Timeline → Threads

**关键问题**:
- [ ] Python 线程的 CPU 使用率？
- [ ] 是否有大量 Python 调用栈？
- [ ] Python 和 Metal 调用的比例？

**推测瓶颈**:
- 如果 Python CPU 占用 >30% → **Python 解释器开销**，需要 C++ 化
- 如果调用栈很深 → **框架调度开销**

---

## 数据导出

### 导出 GPU 利用率数据

1. 选择 "GPU Utilization" track
2. File → Export → As CSV
3. 保存为: `gpu_utilization.csv`

### 导出 Kernel 数据

1. 选择 "Metal Kernels" track
2. File → Export → As CSV
3. 保存为: `kernel_stats.csv`

### 截图关键视图

截取以下视图的截图：
1. 完整 Timeline 概览
2. GPU Utilization 详细视图
3. Metal Kernels 列表
4. Memory Bandwidth 视图

保存到: `/Users/lisihao/ThunderOMLX/profiling/screenshots/`

---

## 验证审判官的推测

基于 profiling 数据，验证以下推测：

| 瓶颈类别 | 审判官推测 | 验证方法 | 实际测量 |
|---------|-----------|---------|---------|
| **Python/框架调度** | 70-100ms (35%) | CPU Activity + Python 调用栈 | ________ms |
| **数据迁移（CPU↔GPU）** | 20-40ms (15%) | Memory Transfer 事件 | ________ms |
| **内存带宽瓶颈** | 30-50ms (18%) | Memory Bandwidth 利用率 | ________ms |
| **GPU内核启动** | 10-20ms (6%) | Kernel Count × 平均启动开销 | ________ms |
| **编译/图构建** | 10-30ms (10%) | 首次 vs 后续生成时间差 | ________ms |

---

## 生成报告

基于 profiling 数据，生成报告：

### 报告模板

```markdown
# MLX 生成性能 Profiling 报告

**日期**: 2026-03-14
**测试环境**: M4 Pro, Qwen 3.5 35B (Q5_K_M), MLX
**测试条件**: FULL SKIP, 生成 1 token, 20 次平均

## 性能概览

- 平均时间: XXX ms
- GPU 利用率: XX%
- 内核调用次数: XX
- 内存带宽利用率: XX%

## 瓶颈分析

### 1. [最大瓶颈名称] (XX ms, XX%)
- **发现**: ...
- **证据**: ...
- **优化方案**: ...

### 2. [第二大瓶颈] (XX ms, XX%)
- **发现**: ...
- **证据**: ...
- **优化方案**: ...

## 优化优先级

1. **立即优化**: [方案名称] - 预期收益 XX ms
2. **次优先**: [方案名称] - 预期收益 XX ms
3. **可选**: [方案名称] - 预期收益 XX ms

## 附录

- profiling 文件: mlx_profiling_YYYYMMDD_HHMMSS.trace
- 截图: profiling/screenshots/
- 数据导出: profiling/*.csv
```

保存为: `/Users/lisihao/ThunderOMLX/docs/MLX_PROFILING_REPORT.md`

---

## 下一步

基于 profiling 报告的结论：

1. **如果 Python/框架调度是主要瓶颈** → 优先执行 Task #6（调整评估策略）
2. **如果数据迁移是主要瓶颈** → 优先执行 Task #7（采样器 GPU 化）
3. **如果内存带宽是主要瓶颈** → 评估 Task #8（KV Cache 优化）可行性
4. **如果 GPU 内核启动是主要瓶颈** → 评估 Task #9（内核融合）可行性

---

**准备就绪！现在可以开始 profiling 了。**
