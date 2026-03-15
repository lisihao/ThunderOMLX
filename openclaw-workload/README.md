# OpenClaw 负载数据集

基于 OpenClaw 真实使用模式生成的模拟生产环境负载数据，用于测试 ThunderOMLX Adaptive Cache Optimizer 的 Phase 3 高级策略。

---

## 📦 数据集内容

### 文件列表
```
openclaw-workload/
├── README.md                      # 本文件
├── metadata.json                  # 负载元数据
└── openclaw-workload-7d.jsonl     # 7 天负载数据（2209 条记录）
```

### 数据规模
- **时间跨度**: 7 天（2026-03-07 至 2026-03-14）
- **总请求数**: 2,209 次推理
- **Agent 类型**: 5 个（researcher, coder, analyst, pm, tester）
- **数据格式**: JSONL（每行一个 JSON 对象）

---

## 📊 数据格式

### JSONL 记录结构
```json
{
  "agent_id": "researcher-agent",
  "timestamp": "2026-03-07T15:42:48.596865",
  "system_prompt_length": 1245,
  "user_query_length": 85,
  "total_prompt_length": 1330,
  "cache_hit_ratio": 0.912,
  "skip_logic_type": "APPROXIMATE",
  "block_size": 128,
  "padding_tokens": 34,
  "padding_overhead": 2.56,
  "prefill_time_ms": 724.5,
  "decode_time_ms": 298.3,
  "total_time_ms": 1022.8,
  "config_version": "1.0.0-initial"
}
```

### 字段说明
| 字段 | 类型 | 说明 |
|------|------|------|
| `agent_id` | string | Agent 标识符（5 种） |
| `timestamp` | string | ISO 8601 格式时间戳 |
| `system_prompt_length` | int | 系统 prompt 长度（tokens） |
| `user_query_length` | int | 用户输入长度（tokens） |
| `total_prompt_length` | int | 总 prompt 长度 |
| `cache_hit_ratio` | float | 缓存命中率（0-1） |
| `skip_logic_type` | string | 跳过逻辑类型（APPROXIMATE/NONE） |
| `block_size` | int | 当前 block size 配置 |
| `padding_tokens` | int | Padding tokens 数量 |
| `padding_overhead` | float | Padding 开销百分比 |
| `prefill_time_ms` | float | Prefill 时间（毫秒） |
| `decode_time_ms` | float | Decode 时间（毫秒） |
| `total_time_ms` | float | 总推理时间（毫秒） |
| `config_version` | string | 配置版本 |

---

## 🤖 Agent 特征

### Agent 配置

| Agent | Prompt 范围 | Cache Hit | Block Size | 使用模式 | 频率 |
|-------|-------------|-----------|-----------|---------|------|
| **researcher-agent** | 850-1700 | 85-95% | 128 (次优) | 工作时间 | 24.9% |
| **coder-agent** | 620-1000 | 75-88% | 96 (次优) | 重度工作时间 | 36.0% |
| **analyst-agent** | 430-850 | 65-80% | 64 (合理) | 全天 | 22.4% |
| **pm-agent** | 310-580 | 80-92% | 64 (合理) | 工作时间 | 8.5% |
| **tester-agent** | 515-900 | 70-85% | 80 (略大) | 工作时间 | 8.2% |

### 时间分布
- **工作时间** (9-18h): 高峰（1.2-1.5× 基线流量）
- **夜间** (0-8h): 低谷（0.1-0.2× 基线流量）
- **晚间** (19-23h): 中等（0.5-0.8× 基线流量）

---

## 🚀 使用方法

### 1. 生成新的负载数据
```bash
cd /Users/lisihao/ThunderOMLX
python3 scripts/generate_openclaw_workload.py
```

可选参数（修改脚本）：
- `num_days`: 模拟天数（默认 7）
- `requests_per_day`: 每天平均请求数（默认 500）
- `output_format`: 输出格式（jsonl 或 csv）

### 2. 运行 Phase 3 测试
```bash
python3 scripts/test_phase3_with_openclaw_workload.py
```

测试包括：
- ✅ Phase 3-A: 多维度分析
- ✅ Phase 3-B: A/B 测试框架（验证）
- ✅ Phase 3-C: 自动回滚（验证）
- ✅ Phase 3-D: 多 Agent 协同优化
- ✅ Phase 3-E: 时间序列分析

### 3. 查看测试报告
```bash
cat docs/phase3-openclaw-workload-test-report.md
```

---

## 📈 测试结果摘要

基于 7 天负载数据的测试结果：

### 关键发现
- **需要优化的 Agent**: 4/5（80%）
- **Fragmentation Score**: 40/100（中度碎片化）
- **协调优化净收益**: 15.8（✅ 建议应用）

### 优化效果
| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 平均 Padding | 5.7% | 1.5% | **-4.2%** |
| KV Cache 复用 | 基线 | +20.0% | **+20.0%** |
| Prefill 加速 | 基线 | - | **~24%** |

---

## 🔧 自定义负载

### 修改 Agent 配置
编辑 `scripts/generate_openclaw_workload.py`：

```python
self.agent_profiles = {
    "custom-agent": {
        "system_prompt_range": (500, 1000),
        "user_query_range": (10, 100),
        "cache_hit_range": (0.7, 0.9),
        "skip_logic_types": ["APPROXIMATE", "NONE"],
        "skip_logic_weights": [0.7, 0.3],
        "block_size": 64,
        "usage_pattern": "working_hours",
        "request_frequency": 0.2,  # 20% 请求
    }
}
```

### 调整时间分布
修改 `_get_hourly_traffic_multiplier()` 方法：

```python
def _get_hourly_traffic_multiplier(self, hour: int, pattern: str) -> float:
    if pattern == "custom_pattern":
        if 0 <= hour <= 6:
            return 2.0  # 凌晨高峰
        else:
            return 0.5
    # ...
```

---

## 📚 相关文档

- [Phase 3 测试报告](../docs/phase3-openclaw-workload-test-report.md)
- [Adaptive Cache Optimizer 文档](../docs/adaptive_cache_optimizer.md)
- [Phase 3 设计文档](../docs/phase3-advanced-strategies.md)

---

## 🐛 故障排查

### 问题：数据导入失败
```
TypeError: log_inference() got an unexpected keyword argument 'timestamp'
```

**解决**: 使用 SQL INSERT 而不是 `log_inference()` 方法，参考 `test_phase3_with_openclaw_workload.py` 中的 `import_workload_to_db()` 实现。

### 问题：时间序列分析返回 None
**原因**: 数据量不足或时间窗口内无数据
**解决**:
1. 增加 `num_days` 或 `requests_per_day`
2. 检查 `analyze_time_series()` 的 `min_samples` 参数

---

**生成日期**: 2026-03-14
**版本**: 1.0.0
**维护者**: ThunderOMLX Team
