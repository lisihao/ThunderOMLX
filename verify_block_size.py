#!/usr/bin/env python3
"""验证 block_size 配置修改是否生效"""

import sys
sys.path.insert(0, '/Users/lisihao/ThunderOMLX/src')

from omlx.scheduler import SchedulerConfig

# 创建默认配置
config = SchedulerConfig()

print("=" * 60)
print("ThunderOMLX Block Size 验证")
print("=" * 60)
print(f"✓ paged_cache_block_size: {config.paged_cache_block_size}")
print(f"✓ disable_block_size_enlargement: {config.disable_block_size_enlargement}")
print(f"✓ enable_prompt_padding: {config.enable_prompt_padding}")
print(f"✓ max_padding_tokens: {config.max_padding_tokens}")
print("=" * 60)

# 验证结果
if config.paged_cache_block_size == 256:
    print("✅ SUCCESS: block_size 已修改为 256")
    print("\n预期效果:")
    print("  - 140-token prompt → 0.55 blocks (vs 之前 2.18 blocks)")
    print("  - 碎片化降低 75%")
    print("  - 缓存命中率预计提升到 50%+")
    print("  - FULL SKIP 触发率预计提升到 30%+")
elif config.paged_cache_block_size == 64:
    print("❌ FAILED: block_size 仍然是 64")
    print("   请检查修改是否保存到文件")
else:
    print(f"⚠️  UNEXPECTED: block_size = {config.paged_cache_block_size}")

print("=" * 60)
