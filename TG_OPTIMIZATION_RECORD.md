# TG 性能记录 - 85.8 tok/s

> **日期**: 2026-03-16
> **性能**: 85.8 tok/s (vs 社区基线 60-80 tok/s)
> **注意**: 此性能**不是** Phase 1-4 优化带来的，而是基础配置 + 测试环境优势

---

## 测试环境

### 硬件
- **芯片**: M4 Pro
- **GPU**: 38 cores
- **内存**: 充足（无 swap）

### 模型
- **路径**: `~/models/qwen3.5-35b-mlx`
- **量化**: FP16/BF16 (未确认，需检查)
- **上下文**: 支持 128K

### 软件
- **引擎**: `omlx.engine.batched.BatchedEngine`
- **MLX**: 最新版本（需记录具体版本）
- **Python**: 3.x

---

## BatchedEngine 配置

### 默认参数（未修改）
```python
engine = BatchedEngine(
    model_name=str(model_path),
    trust_remote_code=True
    # 其他参数使用默认值
)
```

**关键默认值** (需从代码确认):
- `max_batch_size`: ?
- `max_context_length`: ?
- `use_kv_cache`: True (推测)

---

## 测试参数

### TG 测试配置
```python
# Prompt
prompt = "Explain Python and JavaScript differences."  # ~50 tokens

# 生成参数
max_tokens = 512
temperature = 0.7

# 并发
parallel_requests = 1  # 单请求
```

### 测试结果
```
Token 生成: 512 tokens
生成时间: ~5.96s
TG TPS: 85.8 tok/s
```

---

## 关键发现

### 为什么 85.8 tok/s？

1. **单请求测试** ✅
   - 无并发负载
   - GPU 资源独占
   - 社区测试可能有并发（parallel_slots=4）

2. **M4 Pro 硬件** ✅
   - 更强的 GPU 性能
   - 社区可能用 M1/M2/M3

3. **可能的量化优势** ❓
   - 如果是 FP16，比 Q4_K_M 快
   - 需确认社区用的量化级别

4. **Phase 1-4 无关** ❌
   - Phase 1-4 优化 PP（cache save）
   - TG 阶段不涉及 cache save
   - TG 测试代码未使用 Phase 1-4 改动

---

## 复现步骤

```bash
# 1. 检查模型量化级别
file ~/models/qwen3.5-35b-mlx/model.safetensors
# 或
ls -lh ~/models/qwen3.5-35b-mlx/

# 2. 运行 TG 测试
cd /Users/lisihao/ThunderOMLX
python3 test_tg_no_warmup.py

# 3. 验证结果
# 预期: 85.8 tok/s ± 5%
```

---

## 与 Phase 1-4 优化的关系

**Phase 1-4 改动对 TG 的影响: NONE**

- Phase 1: 异步 tensor 提取 → 只影响 PP cache save
- Phase 2: wait_for_writes() → 只影响 PP cleanup
- Phase 3: 队列延迟监控 → 诊断工具，不影响性能
- Phase 4: 批量 Metal 操作 → 未实现

**TG 性能完全来自基础配置，不依赖 Phase 1-4**

---

## 建议

### 保留
✅ BatchedEngine 默认配置（已经很好）
✅ 当前测试环境（M4 Pro, 单请求）
✅ 测试脚本（test_tg_no_warmup.py, test_tg_separate.py）

### 回滚
❌ Phase 1-4 所有改动（导致不稳定，且不影响 TG）
✅ 只保留 bug 修复（cache reconstruction fix）

### 后续
- 记录 MLX 版本、模型量化级别
- 测试并发负载下的 TG 性能
- 对比社区相同硬件的测试结果

---

## 附录：社区基线数据

| 指标 | 范围 | 平均 |
|------|------|------|
| PP (8K) | 637-693 tok/s | ~665 tok/s |
| TG | 36.5-71.3 tok/s | ~65 tok/s |

**我们的结果**:
- TG: 85.8 tok/s (+32% vs 平均)

**差异原因**: 测试条件 + 硬件差异，NOT 优化
