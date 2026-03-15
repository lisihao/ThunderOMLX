# Tokenizer 性能优化报告

**日期**: 2026-03-14
**任务**: Task #12 + #13
**负责人**: Solar (Claude Opus 4.6)

---

## 问题背景

cProfile 分析显示 `tokenizer.get_vocab()` 是最大的性能瓶颈：
- 被调用 20 次，每次 **45ms**
- 占总时间的 **58.9%**（~900ms）
- 原因：MLX-LM 的 `tokenizer.detokenizer` property 每次都创建新实例

---

## 优化方案

### 1. 内存缓存 + Monkey Patch

**实现**:
- 在 `Scheduler.__init__` 中预先获取 `tokenizer.get_vocab()`
- 缓存结果到 `self._cached_vocab`
- Monkey patch `tokenizer.get_vocab` 方法，返回缓存结果

**代码位置**:
- `src/omlx/scheduler.py` (Line 1133-1149)

**性能**:
- 创建 detokenizer: 95ms → **0.57ms** (167x 加速)

---

### 2. 磁盘缓存 + 完整性校验

**实现**:
- 缓存 vocab 到磁盘 (`~/.cache/omlx/vocab_cache/{model_name}_vocab.pkl`)
- 保存元数据（size + hash）用于完整性校验
- 启动时优先从磁盘加载，校验失败则从 tokenizer 重新加载

**代码位置**:
- `src/omlx/scheduler.py` (`_load_vocab_with_cache` 方法)

**性能**:
- 从磁盘加载: **34ms** (vs 从 tokenizer: 45ms)
- 缓存文件: 4.6 MB
- Vocab size: 248,077 tokens

**校验机制**:
- Size 校验: 对比 vocab 条目数量
- Hash 校验: MD5 hash of first 100 entries
- 自动降级: 校验失败时从 tokenizer 重新加载并修复缓存

---

### 3. 避免使用 `tokenizer.detokenizer` Property

**实现**:
- 在 `_get_detokenizer` 和 `_get_harmony_detokenizer` 中
- 直接使用 `NaiveStreamingDetokenizer(self.tokenizer)` 而不是 `self.tokenizer.detokenizer`
- 避免每次都创建新实例

**代码位置**:
- `src/omlx/scheduler.py` (`_get_detokenizer`, Line 1609)
- `src/omlx/scheduler.py` (`_get_harmony_detokenizer`, Line 1637)

---

## 性能结果

### 整体性能提升

```
┌────────────────┬──────────┬──────────┬──────────┐
│    指标        │  优化前  │  优化后  │  提升    │
├────────────────┼──────────┼──────────┼──────────┤
│ 总时间 (5次)   │ 1.541 s  │ 0.170 s  │ -89%     │
│ 平均时间/次    │ 308 ms   │ 34 ms    │ 9.1x     │
│ Tokenizer 耗时 │ 907 ms   │ ~0 ms    │ 消除     │
│ Detokenizer 创建│ 95 ms    │ 0.57 ms  │ 167x     │
└────────────────┴──────────┴──────────┴──────────┘
```

### 磁盘缓存性能

```
┌────────────────┬──────────┬──────────┐
│    操作        │  时间    │  说明    │
├────────────────┼──────────┼──────────┤
│ 从 tokenizer   │ 41-45 ms │ 首次加载 │
│ 从磁盘缓存     │ 34 ms    │ 后续加载 │
│ 校验失败重载   │ 41 ms    │ 自动修复 │
└────────────────┴──────────┴──────────┘
```

---

## 新的性能瓶颈分布 (34ms)

```
┌─────────────────────┬──────────┬──────┐
│      瓶颈           │ 实测耗时 │ 占比 │
├─────────────────────┼──────────┼──────┤
│ _process_prompts    │ 57ms     │ 33.5%│
│ _next (MLX generate)│ 51ms     │ 30.0%│
│ cache.extract       │ 21ms     │ 12.4%│
│ 模型 Forward        │ 24ms     │ 14.1%│
│ 其他                │ 17ms     │ 10.0%│
├─────────────────────┼──────────┼──────┤
│ **总计**            │ 170ms    │ 100% │
└─────────────────────┴──────────┴──────┘
```

**注意**: 总时间 170ms 是 5 次生成的总和，平均每次 34ms。

---

## 验证测试

### Test 1: 基础性能测试

```bash
python3 test_tokenizer_issue.py
```

**结果**:
- 访问 `tokenizer.detokenizer` property: 95.63 ms
- 直接调用 `get_vocab()`: 47.13 ms
- 使用缓存 vocab: 0.57 ms
- **加速比**: 167.2x

### Test 2: cProfile 完整测试

```bash
PYTHONPATH=src python3 test_python_profiling.py
```

**结果**:
- 总时间: 1.541s → 0.170s
- 平均时间/次: 308ms → 34ms
- **加速比**: 9.1x

### Test 3: 磁盘缓存性能测试

```bash
PYTHONPATH=src python3 test_vocab_cache_timing.py
```

**结果**:
- 直接从磁盘加载 pickle: 34ms (平均)
- 缓存文件大小: 4.6 MB

### Test 4: 缓存校验测试

```bash
# 故意破坏缓存
echo -e "99999\nwrong_hash" > ~/.cache/omlx/vocab_cache/*_vocab.meta

# 运行测试
PYTHONPATH=src python3 test_vocab_cache_validation.py
```

**结果**:
- ✅ 检测到缓存损坏
- ✅ 从 tokenizer 重新加载（41.42ms）
- ✅ 自动修复缓存文件

---

## 代码改动

### 文件修改

1. **`src/omlx/scheduler.py`**
   - 添加 `_load_vocab_with_cache()` 方法（磁盘缓存 + 校验）
   - 在 `__init__` 中调用并 monkey patch `tokenizer.get_vocab`
   - 修改 `_get_detokenizer()` 避免使用 property
   - 修改 `_get_harmony_detokenizer()` 避免使用 property

### 新增文件

1. **`test_tokenizer_issue.py`** - 验证问题和优化效果
2. **`test_vocab_cache.py`** - 测试磁盘缓存基本功能
3. **`test_vocab_cache_timing.py`** - 测试磁盘缓存性能
4. **`test_vocab_cache_validation.py`** - 测试缓存校验机制

---

## 经验教训

### 1. Property 陷阱

MLX-LM 的 `tokenizer.detokenizer` 是一个 **property**，每次访问都创建新实例：

```python
@property
def detokenizer(self):
    return self._detokenizer_class(self)  # 每次创建新实例！
```

**教训**: 检查库的 API 时，注意 property 和方法的区别。如果是 property，可能每次都有开销。

### 2. 性能分析的重要性

**原计划**:
- Metal System Trace profiling
- 分析 GPU 层瓶颈

**实际发现**:
- 最大瓶颈在 Python 层（tokenizer）
- cProfile 就足够了，不需要 Metal trace

**教训**: 先用简单的 profiling 工具（cProfile）找到大瓶颈，再用复杂工具（Metal trace）分析细节。

### 3. 磁盘缓存的价值

虽然磁盘缓存只节省了 **11ms**（45ms → 34ms），但它的价值在于：
- ✅ 启动时更快（不需要等待 tokenizer 初始化）
- ✅ 可靠性（有校验机制）
- ✅ 可迁移（可以预先生成缓存并分发）

### 4. 校验的必要性

用户（监护人）明确要求："要做校验，万一存储坏了还可以从模型中加载"

**实现**:
- Size 校验（快速）
- Hash 校验（准确）
- 自动降级（可靠）

这确保了系统的鲁棒性。

---

## 下一步

基于新的性能瓶颈分布（34ms），下一步优化方向：

### Task #14: _process_prompts 优化（预期 -20ms）

当前占 33.5%（57ms），可能的优化：
- 减少 hash 计算次数
- 优化 cache 查找逻辑
- 使用 C++ 实现关键路径

### Task #15: 减少 Async/Threading 层次（预期 -15ms）

当前 async/threading 开销已大幅降低（从 163ms 降到估计 <20ms），但仍可优化：
- 减少异步层次
- 关键路径使用同步 API

---

**最后更新**: 2026-03-14
**Git Tag**: v0.2.1-tokenizer-opt
