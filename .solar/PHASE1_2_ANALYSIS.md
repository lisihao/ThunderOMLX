# Phase 1+2 优化分析报告

## 实施总结

### Phase 1: 异步 Tensor 提取
**修改**: `paged_ssd_cache.py` (+29/-20)
**原理**: 将 `_extract_tensor_bytes()` 移到后台线程

**优化点**:
- 推理线程：`mx.eval()` + `mx.synchronize()` + `queue.put_nowait()`
- 后台线程：提取 bytes + 计算 checksum

**理论节省**:
- tensor 提取：~100ms (64 layers × 1.5ms)
- checksum 计算：~50ms
- **总计**：~150ms **每次 save_block 调用**

### Phase 2: 异步 save_block 调用
**修改**: 新建 `cache_save_executor.py` (148 行) + `scheduler.py` (+112/-18)
**原理**: 单线程执行器异步执行 `store_cache()`

**优化点**:
- 推理线程：`submit_save()` 立即返回（<1ms）
- 后台线程：执行完整 `store_cache()` 流程

**理论节省**:
- store_cache 阻塞时间：~1500ms **每次 cleanup_finished 调用**

## 关键问题：save_block 调用频率

根据 ThunderOMLX 架构，`save_block()` 不是每个 token 都调用，而是：
- **在请求完成时**调用 `_cleanup_finished()`
- `cleanup_finished()` → `store_cache()` → `save_block()` (多次，每个 block 一次)

所以优化效果**不是每个 token 都体现**，而是：
- **请求完成时**减少阻塞时间
- **长请求**受益更大（更多 blocks 需要保存）

## 正确的性能分析方法

需要实际运行推理测试，测量：
1. **端到端 Processing TPS**（多请求场景）
2. **cleanup_finished 执行时间**（请求完成时）
3. **后台队列统计**（dropped/errors）

## 当前状态

✅ **代码修改完成**：
- Phase 1: paged_ssd_cache.py
- Phase 2: cache_save_executor.py + scheduler.py

✅ **功能验证通过**：
- 语法检查通过
- block_table fallback 验证
- Metal 线程安全

⚠️ **性能测试缺失**：
- 需要实际推理场景测试
- 需要多请求并发测试
- 需要长上下文测试

## 建议

1. **暂停 Phase 3/4**，先验证 Phase 1+2 效果
2. **运行实际推理测试**（需要模型加载）
3. **如果效果不理想**，分析根因并调整
4. **如果效果理想**，再考虑 Phase 3/4

---
分析时间: 2026-03-16
