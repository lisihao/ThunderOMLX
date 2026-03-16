#!/bin/bash
# 回滚 Phase 1-4 改动，保留 bug 修复
# 日期: 2026-03-16

set -e

echo "======================================================================"
echo "回滚 Phase 1-4 改动"
echo "======================================================================"
echo ""
echo "⚠️  WARNING: 此脚本将回滚所有 Phase 1-4 改动"
echo "✅ 保留: cache reconstruction bug fix (tensors_raw None check)"
echo "❌ 移除: 异步 tensor 提取、wait_for_writes()、队列延迟监控"
echo ""
read -p "继续? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 备份当前修改
echo ""
echo "1️⃣ 备份当前修改..."
git diff > phase1-4-changes-backup-$(date +%Y%m%d-%H%M%S).patch
echo "✅ 备份完成: phase1-4-changes-backup-*.patch"

# 回滚到原始版本
echo ""
echo "2️⃣ 回滚文件到原始版本..."
git checkout HEAD -- src/omlx/cache/paged_ssd_cache.py
git checkout HEAD -- src/omlx/cache/prefix_cache.py
git checkout HEAD -- src/omlx/scheduler.py
echo "✅ 文件已回滚"

# 重新应用 bug 修复（cache reconstruction fix）
echo ""
echo "3️⃣ 重新应用 bug 修复..."

# Fix 1: paged_ssd_cache.py - load_blocks_batch tensors_raw None check
echo "应用 Fix 1: load_blocks_batch tensors_raw None check..."
cat > /tmp/fix1.patch << 'EOF'
--- a/src/omlx/cache/paged_ssd_cache.py
+++ b/src/omlx/cache/paged_ssd_cache.py
@@ -2007,7 +2007,18 @@ class PagedSSDCache:
         # Phase 1: Check hot cache and index (fast, no I/O)
         for block_hash in block_hashes:
             # Try hot cache first
             entry = self._hot_cache_get(block_hash)
-            if entry is not None:
+            if entry is not None and entry.get('tensors_raw') is not None:
+                # Hot cache hit with tensors_raw (hot cache enabled mode)
                 arrays = self._arrays_from_tensors_raw(entry['tensors_raw'])
                 cache_data = self._reconstruct_cache_data(
                     arrays, entry['file_metadata'],
@@ -2023,6 +2034,10 @@ class PagedSSDCache:
                     self._stats["loads"] += 1
                     self._stats["hits"] += 1
                 continue
+            elif entry is not None and entry.get('tensors_raw') is None:
+                # Hot cache entry exists but tensors_raw is None (SSD mode, pending write)
+                # Fall through to disk load path below
+                pass

             # Check index for SSD blocks
             metadata = self._index.get(block_hash)
EOF

patch -p1 < /tmp/fix1.patch || echo "⚠️  Fix 1 可能已应用或冲突"

echo "✅ Bug 修复已应用"

# 验证修改
echo ""
echo "4️⃣ 验证回滚结果..."
echo "修改的文件:"
git diff --name-only

echo ""
echo "======================================================================"
echo "✅ 回滚完成"
echo "======================================================================"
echo ""
echo "已移除的改动:"
echo "  ❌ Phase 1: 异步 tensor 提取"
echo "  ❌ Phase 2: wait_for_writes() 同步"
echo "  ❌ Phase 3: 队列延迟监控"
echo "  ❌ Phase 4: 批量 Metal 操作（未实现）"
echo ""
echo "保留的修复:"
echo "  ✅ cache reconstruction bug fix (tensors_raw None check)"
echo ""
echo "下一步:"
echo "  1. 测试 TG 性能: python3 test_tg_no_warmup.py"
echo "  2. 测试 PP 性能: python3 test_pp_no_warmup.py"
echo "  3. 验证 128K 上下文支持"
echo "  4. 如果稳定，提交: git add -A && git commit -m 'Rollback Phase 1-4, keep bug fixes'"
echo ""
