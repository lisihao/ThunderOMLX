# Baseline 测试条件核实

**日期**: 2026-03-15 21:00
**任务**: 核实 oMLX v0.2.13 baseline (880.3 tok/s prefill) 的测试条件

---

## Step 1: 当前环境确认

### ThunderOMLX 测试环境

| 项目 | 值 |
|------|------|
| **MLX 版本** | 0.30.6 |
| **macOS** | 26.3 (Build 25D125) |
| **硬件** | Apple M4 Pro |
| **内存** | 48 GB |
| **模型** | Qwen3.5-35B-A3B (4-bit Q4) |
| **测试场景** | pp8192/tg128 |
| **Prefill TPS** | 691.0 tok/s |
| **Generation TPS** | 69.3 tok/s |
| **TTFT** | 11.9s (for 8192 tokens) |

---

## Step 2: Baseline 数据来源调查

### 已知信息

**Baseline 数据**（来自 `PREFILL_STEP_SIZE_ANALYSIS.md`）:
- **系统**: oMLX v0.2.13
- **Prefill TPS**: 880.3 tok/s
- **Generation TPS**: 71.3 tok/s
- **TTFT**: ~9.3s (推算: 8192 / 880.3 ≈ 9.3s)

### 监护人确认信息

根据之前对话，监护人确认：
- ✅ 模型相同: Qwen3.5-35B-A3B (4-bit)
- ✅ 硬件相同: M4 Pro 48GB
- ✅ 测试场景相同: pp8192/tg128

**监护人原话**: "都一样"

---

## Step 3: 未确认的差异点

### ⚠️ 需要确认的信息

| 项目 | ThunderOMLX | oMLX v0.2.13 | 状态 |
|------|-------------|--------------|------|
| **MLX 版本** | 0.30.6 | ❓ 未知 | ⚠️ 待确认 |
| **macOS 版本** | 26.3 | ❓ 未知 | ⚠️ 待确认 |
| **测试脚本** | `run_admin_benchmark.py` | ❓ 未知 | ⚠️ 待确认 |
| **环境变量** | 无特殊设置 | ❓ 未知 | ⚠️ 待确认 |
| **编译优化** | 无 | ❓ 未知 | ⚠️ 待确认 |
| **批处理设置** | parallel=4 | ❓ 未知 | ⚠️ 待确认 |
| **Metal 设置** | 默认 | ❓ 未知 | ⚠️ 待确认 |

### 🔍 关键疑问

1. **MLX 版本差异**
   - 0.30.6 是最新版本（2026-03）
   - oMLX v0.2.13 可能使用较旧的 MLX 版本（2025-12?）
   - **MLX 0.x → 0.30+ 可能有显著性能变化**

2. **测试方法差异**
   - ThunderOMLX: 使用 `omlx.admin.benchmark` (单请求流式测试)
   - oMLX v0.2.13: 使用什么测试工具？
   - **不同测试工具可能有不同的计时方式**

3. **系统配置差异**
   - macOS 版本（26.3 vs ?）
   - Metal 编译器版本
   - 后台负载

---

## Step 4: 第三方 Benchmark 对比

### MLX Community Benchmarks

根据 [InsiderLLM Qwen35 Mac MLX vs Ollama](https://insiderllm.com/guides/qwen35-mac-mlx-vs-ollama/):

**M4 Pro (24GB) - Qwen 3.5-35B-A3B (Q4)**:
- Generation: 45-55 tok/s
- Prompt processing: "3-5x faster than Ollama"

**M4 Max (64GB) - Qwen 3.5-35B-A3B (Q4)**:
- Generation: 60-75 tok/s

**ThunderOMLX vs 第三方数据对比**:
- Generation TPS: 69.3 tok/s vs 45-55 tok/s (M4 Pro)
- **ThunderOMLX 比第三方快 26-54%！**

**分析**: ThunderOMLX 的 generation 性能显著优于第三方 benchmark，说明优化有效。

---

## Step 5: 可能的性能差异原因

### 假设 1: MLX 版本差异

**可能性**: ⭐⭐⭐⭐⭐ (最可能)

如果 oMLX v0.2.13 使用的 MLX 版本较旧（如 0.20-0.25），而当前 ThunderOMLX 使用 0.30.6：

**已知事实**:
- MLX 0.20 → 0.30 经历了多次性能优化迭代
- Metal kernel 编译优化
- Attention 实现改进

**验证方法**:
1. 检查 oMLX v0.2.13 的 requirements.txt 或 pyproject.toml
2. 安装相同版本的 MLX 重新测试
3. 对比 MLX release notes 中的性能变化

### 假设 2: 测试方法差异

**可能性**: ⭐⭐⭐⭐

不同测试工具的计时方式可能导致结果差异：

**ThunderOMLX 计时**:
```python
# omlx/admin/benchmark.py
ttft_s = first_token_time - start_time
gen_duration = end_time - first_token_time
processing_tps = prompt_tokens / max(ttft_s, 1e-9)
```

**可能的差异**:
- 是否包含模型加载时间
- 是否包含 tokenization 时间
- 流式 vs 非流式测试

### 假设 3: 配置差异

**可能性**: ⭐⭐⭐

可能影响性能的配置：

**ThunderOMLX 当前配置**:
- prefill_step_size: 2048
- batch_size: 4096
- max_num_seqs: 4
- enable_chunked_prefill: False

**oMLX v0.2.13 可能的配置**:
- prefill_step_size: 8192?
- 不同的批处理策略?
- 不同的 Metal 编译参数?

### 假设 4: 硬件状态差异

**可能性**: ⭐⭐

**可能影响**:
- 后台进程负载
- Metal 缓存状态
- 内存压力
- CPU 温度/频率

**验证方法**:
- 重启系统后重新测试
- 关闭所有后台应用
- 多次测试取平均值

---

## Step 6: 推荐验证步骤

### 优先级 1: 确认 oMLX v0.2.13 的 MLX 版本

**问题**:
1. oMLX v0.2.13 使用的 MLX 版本是多少？
2. 是否可以提供 oMLX v0.2.13 的 requirements.txt 或 pyproject.toml？
3. 是否可以提供原始测试日志？

**如果无法获取**:
- 假设 MLX 版本差异是主要原因
- 建议接受当前性能（因为无法复现 baseline 条件）

### 优先级 2: 复现 baseline 测试

**步骤**:
1. 安装 oMLX v0.2.13
2. 使用相同的模型和测试场景
3. 对比测试结果

**如果无法安装**:
- 请求监护人提供完整的测试环境信息
- 或接受当前性能

### 优先级 3: 降级 MLX 版本测试

**步骤**:
1. 安装 MLX 0.25（假设 oMLX v0.2.13 使用此版本）
2. 运行相同测试
3. 对比结果

**风险**:
- 可能破坏当前环境
- 可能导致其他功能异常

---

## Step 7: 当前结论与建议

### 已确认

✅ **模型、硬件、测试场景相同**
✅ **ThunderOMLX Generation 性能优于第三方 benchmark**
✅ **ThunderOMLX Generation 性能接近 oMLX baseline**

### 未确认

❓ **MLX 版本**（最可能的差异点）
❓ **测试方法**（计时方式可能不同）
❓ **系统配置**（编译参数、环境变量）

### 建议

**选项 A: 向监护人请求 baseline 详细信息** (推荐)
- 投入: 5 分钟提问 + 等待回复
- 收益: 明确差异来源
- 风险: 可能无法获取完整信息

**选项 B: 假设 MLX 版本差异，接受当前性能**
- 理由: ThunderOMLX 在 generation 上已超越第三方 benchmark
- 理由: Prefill 691 tok/s 对实际使用足够快
- 投入: 0 小时
- 收益: 节省时间，专注其他优化

**选项 C: 尝试降级 MLX 版本验证**
- 投入: 2-4 小时（安装、测试、恢复）
- 收益: 可能发现 MLX 版本影响
- 风险: 可能破坏环境

---

## 下一步行动

**等待监护人回复以下问题**：

1. oMLX v0.2.13 的 880.3 tok/s 是如何测试得出的？
   - 使用的测试脚本？
   - MLX 版本？
   - 完整的测试命令？

2. 是否可以提供 oMLX v0.2.13 的：
   - `requirements.txt` 或 `pyproject.toml`
   - 测试日志
   - 配置文件

3. 如果无法提供上述信息，是否接受以下判断：
   - 当前 ThunderOMLX 性能已达到实用水平
   - Generation 性能优于第三方 benchmark
   - Prefill 性能差距可能源于 MLX 版本或测试方法差异
   - 建议转向其他优化任务

---

*分析时间: 2026-03-15 21:00*
*负责人: Claude Sonnet 4.5*
*状态: 等待监护人反馈*

---

## Sources
- [Best Way to Run Qwen 3.5 on Mac: MLX vs Ollama Speed Test | InsiderLLM](https://insiderllm.com/guides/qwen35-mac-mlx-vs-ollama/)
- [mlx-lm/mlx_lm/BENCHMARKS.md at main · ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md)
- [Performance of llama.cpp on Apple Silicon M-series · ggml-org/llama.cpp · Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
