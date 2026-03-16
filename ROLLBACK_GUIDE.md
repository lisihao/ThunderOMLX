# Phase 1-4 回滚指南

> **目标**: 回滚所有 Phase 1-4 改动，只保留 bug 修复
> **原因**: Phase 1-4 导致系统不稳定（128K 上下文 → 只能处理短 prompt）
> **日期**: 2026-03-16

---

## 改动清单

### Phase 1: 异步 Tensor 提取
**文件**: `src/omlx/cache/paged_ssd_cache.py`
- 行 1554: 传递 arrays 而非 tensors_raw
- 行 1249: writer 线程接收 arrays
- 行 1265: writer 线程提取 bytes

### Phase 2: wait_for_writes() 同步
**文件**: `src/omlx/cache/paged_ssd_cache.py`
- 行 2529-2576: wait_for_writes() 方法

**文件**: `src/omlx/scheduler.py`
- 行 3731-3740: 调用 wait_for_writes()

### Phase 3: 队列延迟监控
**文件**: `src/omlx/cache/paged_ssd_cache.py`
- 行 1252-1259: 队列延迟测量

### Bug 修复 (需保留)
**文件**: `src/omlx/cache/paged_ssd_cache.py`
- 行 2010-2030: load_blocks_batch tensors_raw None check

**文件**: `src/omlx/cache/prefix_cache.py`
- 行 1967-1969: debug traceback (可选，可移除)

---

## 方案 1: Git 回滚 + 手动应用 bug 修复（推荐）

### 步骤 1: 备份当前修改
```bash
cd /Users/lisihao/ThunderOMLX
git diff > phase1-4-backup-$(date +%Y%m%d-%H%M%S).patch
```

### 步骤 2: 回滚到原始版本
```bash
git checkout HEAD -- src/omlx/cache/paged_ssd_cache.py
git checkout HEAD -- src/omlx/cache/prefix_cache.py
git checkout HEAD -- src/omlx/scheduler.py
```

### 步骤 3: 手动应用 bug 修复

**编辑 `src/omlx/cache/paged_ssd_cache.py`**，找到 `load_blocks_batch` 方法（约 2007 行）：

**原始代码**:
```python
# Phase 1: Check hot cache and index (fast, no I/O)
for block_hash in block_hashes:
    # Try hot cache first
    entry = self._hot_cache_get(block_hash)
    if entry is not None:
        arrays = self._arrays_from_tensors_raw(entry['tensors_raw'])
        cache_data = self._reconstruct_cache_data(
            arrays, entry['file_metadata'],
            entry['num_layers'], entry['layer_cache_types'],
        )
        results[block_hash] = cache_data
        ...
        continue
```

**修改为**:
```python
# Phase 1: Check hot cache and index (fast, no I/O)
for block_hash in block_hashes:
    # Try hot cache first
    entry = self._hot_cache_get(block_hash)
    if entry is not None and entry.get('tensors_raw') is not None:
        # Hot cache hit with tensors_raw (hot cache enabled mode)
        arrays = self._arrays_from_tensors_raw(entry['tensors_raw'])
        cache_data = self._reconstruct_cache_data(
            arrays, entry['file_metadata'],
            entry['num_layers'], entry['layer_cache_types'],
        )
        results[block_hash] = cache_data
        ...
        continue
    elif entry is not None and entry.get('tensors_raw') is None:
        # Hot cache entry exists but tensors_raw is None (SSD mode, pending write)
        # Fall through to disk load path below
        pass
```

**关键点**: 添加 `and entry.get('tensors_raw') is not None` 检查

### 步骤 4: 验证回滚
```bash
# 查看修改
git diff src/omlx/cache/paged_ssd_cache.py

# 应该只看到 bug 修复（tensors_raw None check）
```

### 步骤 5: 测试
```bash
# 测试 TG 性能
python3 test_tg_no_warmup.py
# 预期: 85.8 tok/s ± 5%

# 测试 PP 性能（应该能处理更长 prompt）
python3 test_pp_no_warmup.py

# 测试 8K 上下文
python3 test_pp_8k.py
```

### 步骤 6: 提交（如果测试通过）
```bash
git add src/omlx/cache/paged_ssd_cache.py
git commit -m "Rollback Phase 1-4, keep cache reconstruction bug fix"
```

---

## 方案 2: 使用回滚脚本（自动化）

```bash
chmod +x rollback_phase1-4.sh
./rollback_phase1-4.sh
```

**注意**: 脚本可能因行号变化失败，推荐方案 1（手动）

---

## 验证清单

回滚后，系统应该：

### 稳定性
- [ ] 能处理 8K context（不崩溃）
- [ ] 能处理 128K context（原始能力）
- [ ] 没有 Metal 并发错误（或至少减少）

### 性能
- [ ] TG: ~85.8 tok/s（应该保持）
- [ ] PP: 600-800 tok/s（社区基线）
- [ ] 无队列延迟警告（已移除监控）

### Bug 修复
- [ ] 没有 "Failed to reconstruct cache: 'NoneType' object has no attribute 'items'" 错误
- [ ] cache 能正常 load/save

---

## 回滚影响分析

### 移除的功能
- ❌ 异步 tensor 提取（Phase 1）
- ❌ wait_for_writes() 同步（Phase 2）
- ❌ 队列延迟监控（Phase 3）
- ❌ 批量 Metal 操作（Phase 4，未实现）

### PP 性能影响
**理论损失**:
- Phase 1: -2.8% (-19 tok/s)
- Phase 2: -2.1% (-15 tok/s)
- Phase 3: -1.0% (-7 tok/s)
- Phase 4: -0.3% (-2 tok/s)
- **总计**: 理论损失 -6.3%（692.7 → ~650 tok/s）

**实际影响**: 可能小于理论值，因为：
- Phase 1-4 在高负载时可能导致 Metal 错误
- 稳定性 > 小幅性能提升
- 能处理 128K 上下文更重要

### TG 性能影响
**无影响**: TG 85.8 tok/s 完全来自基础配置，不依赖 Phase 1-4

---

## 后续计划

### 短期
1. 验证回滚后的稳定性
2. 测试 128K 上下文支持
3. 记录基线性能数据

### 中期
1. 研究 Metal 并发错误根因
2. 寻找其他 PP 优化方向（不触发 Metal 错误）
3. 对比社区相同硬件的性能

### 长期
1. 等待 MLX 框架更新（可能修复 Metal 并发问题）
2. 考虑其他 cache 策略（避免频繁 save）
3. 优化 cache 数据结构（减少序列化开销）

---

## 总结

**回滚原因**: Phase 1-4 导致不稳定（128K → 短 prompt）
**保留内容**: cache reconstruction bug fix
**预期结果**: 恢复 128K 上下文支持，TG 性能保持 85.8 tok/s
**理论损失**: PP 性能 -6.3%（但可能避免 Metal 错误）
**最终目标**: 稳定 > 小幅性能提升
