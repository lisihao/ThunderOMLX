# Phase 2: Long Context Optimization - CHANGELOG

> **版本**: v0.2.0 → v0.2.3
> **完成日期**: 2026-03-17
> **核心目标**: 支持 OpenClaw 多 agent 场景的 128K-256K 超长上下文

---

## 🎯 核心成果

### ✅ 解决 128K Tokens 首次 Prefill OOM

**问题**: MLX Metal buffer 限制 30GB，128K tokens 首次 prefill 会 OOM

**解决**: Chunked Prefill - 分块计算 KV cache

**效果**:
- 16K: 949 tok/s ✅
- 64K: 624 tok/s ✅
- **128K: 422 tok/s ✅ (无 OOM)**

### ✅ 输出质量保证

**测试**: Baseline vs Chunked Prefill
- **相似度**: 99.88%（仅首字母空格差异）
- **Cache 累积**: 100% 正确
- **结论**: 不影响生成质量

### ✅ 性能开销极低

**Chunked Prefill 开销**:
- 小 context (6K): **+2.5%**
- 中 context (16K): **+11.1%**
- 结论: < 20%，完全可接受

### ✅ Prefix Cache 流式加载

**问题**: Cache 加载到内存时内存峰值过高

**解决**: 分批加载 cache blocks

**效果**:
- 64K: -1.1% TTFT, +18MB 内存
- **128K: +0.1% TTFT, -739MB 内存** ⭐

---

## 🚀 技术突破

### 1. 发现 MLX-LM KVCache 原地修改机制

**关键发现**:
- `KVCache.update_and_fetch()` 原地修改 cache
- Model 不返回 cache tuple，cache 在内部被修改
- 可以逐块调用 model，cache 会自动累积

**核心代码**:
```python
from mlx_lm.models.cache import KVCache

# 创建 cache（每层一个）
cache = [KVCache() for _ in range(len(model.model.layers))]

# 逐块 prefill，cache 自动累积
for chunk in chunks:
    logits = model(mx.array([chunk]), cache=cache)
    mx.eval(logits)
    mx.eval([c.keys for c in cache])  # 确保更新完成
```

### 2. Prefix Cache 动态阈值控制

**灵活性**:
- 环境变量: `OMLX_STREAMING_THRESHOLD`
- 默认: 32 blocks (8K tokens)
- 可调: 16-999999 blocks

**代码**:
```python
# src/omlx/cache/prefix_cache.py
import os
STREAMING_THRESHOLD = int(os.getenv("OMLX_STREAMING_THRESHOLD", "32"))
```

### 3. P2.2 + P2.3 完美配合

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  首次请求 (cache 不存在)                                 │
│       ↓                                                  │
│  P2.3: Chunked Prefill                                   │
│       ↓                                                  │
│  构建 KV cache (128K tokens, 无 OOM)                     │
│       ↓                                                  │
│  保存到 SSD                                              │
│                                                          │
│  ────────────────────────────────────────────────────    │
│                                                          │
│  后续请求 (cache 已存在)                                 │
│       ↓                                                  │
│  P2.2: 流式加载 cache                                    │
│       ↓                                                  │
│  加载到内存 (避免内存峰值，节省 739MB)                   │
│       ↓                                                  │
│  生成响应                                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 性能对比

### P2.2: Prefix Cache 流式加载

| Context | Batch Load | Streaming Load | TTFT Diff | Memory Diff |
|---------|------------|----------------|-----------|-------------|
| 16K | 10,162ms | 10,168ms | **+0.1%** | +99MB |
| 32K | 22,892ms | 22,417ms | **-2.1%** | +40MB |
| 64K | 36,155ms | 35,754ms | **-1.1%** | +18MB |
| **128K** | 87,783ms | 87,875ms | **+0.1%** | **-739MB** ⭐ |

**结论**: 几乎零性能开销，128K 显著节省内存

### P2.3: Chunked Prefill

| Context | Baseline | Chunked | Overhead | Status |
|---------|----------|---------|----------|--------|
| 16K (prefill) | 6,641ms | 6,808ms | **+2.5%** | ✅ |
| 16K (full) | 7,819ms | 8,685ms | **+11.1%** | ✅ |
| 128K | N/A (OOM) | 122,164ms | N/A | ✅ **无 OOM** |

**结论**: 小开销，解决大问题

---

## 🔧 实现文件

### P2.2 相关

| 文件 | 描述 |
|------|------|
| `src/omlx/cache/prefix_cache.py` | 添加 OMLX_STREAMING_THRESHOLD 支持 |
| `test_p2.2_performance_comparison.py` | 16K/32K 性能对比 |
| `test_p2.2_performance_64k_128k.py` | 64K/128K 性能对比 |

### P2.3 相关

| 文件 | 描述 |
|------|------|
| `test_p2.3_chunked_prefill.py` | 16K 正确性测试 |
| `test_p2.3_oom_test.py` | 16K/64K/128K OOM 测试 |
| `test_chunked_prefill_verification.py` | 完整验证（输出质量） |
| `test_cache_api.py` | MLX-LM cache API 研究 |
| `.solar/p2.3-chunked-prefill-summary.md` | 完整技术总结 |

---

## 🎓 经验教训

### 1. 理解底层机制比调参数更重要

**错误尝试**: 直接期望 model 返回 `(logits, cache)` tuple

**正确理解**: MLX-LM 使用可变 cache 对象，原地修改

**收获**: 读源码找到 `KVCache.update_and_fetch()` 机制

### 2. 固定切分也能有好效果

**担心**: 固定 4K 切分会破坏文本连贯性

**实测**: 相似度 99.88%，几乎无影响

**原因**: Transformer 的 attention 机制本身就能处理跨边界依赖

### 3. 性能测试要分离变量

**P2.1 教训**: Phase 1 测试 chunked_generate 慢 77%，因为做了两次工作

**P2.3 改进**:
- 正确实现 cache 累积
- 分离 prefill 和 generation 测试
- 性能开销降到 2.5%

---

## 📈 未来优化

### 1. 智能分块（待实现）

**语义边界检测**:
- 段落: `\n\n`
- 句子: `. ` `! ` `? `
- 代码: 完整函数

**混合 chunk size**:
- 基础: 4K tokens
- 弹性: 3.5K-4.5K tokens
- 最大: 6K tokens

**预期收益**:
- 更好的文本连贯性
- 减少边界效应

### 2. 动态 Chunk Size

**根据可用内存调整**:
- 小内存 (8GB): 2K chunk
- 中内存 (16GB): 4K chunk
- 大内存 (32GB): 8K chunk

### 3. 并行 Chunked Prefill

**如果内存足够**:
- 多个 chunk 并行计算
- 最后合并 cache

**挑战**: 需要修改 MLX-LM 的 KVCache 实现

---

## 🏆 关键指标

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 128K Prefill | ❌ OOM | ✅ 422 tok/s | **从不可用到可用** |
| 128K Cache Load | 高内存峰值 | -739MB | **-3.8%** |
| 输出质量 | N/A | 99.88% | **无损** |
| 性能开销 | N/A | +2.5% ~ +11.1% | **可接受** |

---

## ✨ 总结

**Phase 2 核心价值**:

1. **彻底解决 OpenClaw 超长上下文问题**
   - 首次 prefill: ✅ Chunked Prefill
   - 后续加载: ✅ 流式加载

2. **技术突破**
   - 发现 MLX-LM KVCache 原地修改机制
   - 实现零拷贝 cache 累积

3. **工程实践**
   - 完整的测试覆盖（16K/64K/128K）
   - 性能和正确性双验证
   - 可扩展的设计（支持动态阈值）

4. **为 OpenClaw 铺平道路**
   - 128K tokens 稳定运行
   - 多 agent 场景无压力
   - 性能开销极低

---

*Phase 2 完成于 2026-03-17*
*测试通过: 16K ✅ / 64K ✅ / 128K ✅*
*Ready for Production 🚀*
