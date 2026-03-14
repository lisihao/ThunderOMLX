# ThunderLLAMA 优化特性移植实现计划

> **目标**: 将 ThunderLLAMA 的核心优化能力移植到 oMLX，实现 4-5x 性能提升

---

## 📊 执行摘要

### 当前状态
- **oMLX Generation TPS**: 119.3 tok/s
- **ThunderLLAMA baseline**: 687.6 tok/s
- **性能差距**: 5.8x

### 移植目标
- **Generation TPS**: 500-650 tok/s (4-5x 提升)
- **Prefill TPS**: 200-300 tok/s (5-7x 提升)
- **TTFT**: 400-600ms (6-9x 提升)

### 实现周期
- **P0 特性**: 3-4 天
- **验证测试**: 1 天
- **总计**: 4-5 天

---

## 🎯 移植特性优先级

### P0 (必须) - 3-4 天

| 特性 | 预期加速 | 工期 | 复杂度 |
|------|---------|------|--------|
| **1. Full Skip Logic** | 27x (100% 命中) | 1.5 天 | ⭐⭐⭐ |
| **2. Approximate Skip** | 5-10x (95%+ 命中) | 1 天 | ⭐⭐ |
| **3. Hybrid Hashing** | 50x 哈希速度 + 3-7x 重叠检测 | 1 天 | ⭐⭐ |
| **4. SSD Compression** | 2-4x I/O 加速 | 0.5 天 | ⭐ |

### P1 (重要) - 2-3 天

| 特性 | 预期加速 | 工期 |
|------|---------|------|
| **5. Smart Prefetch** | 4x L3 加速 | 1 天 |
| **6. Checksum Validation** | 数据完整性 | 0.5 天 |
| **7. Adaptive Chunk Prefill** | 减少碎片化 | 1 天 |

### P2 (可选) - 1-2 天

| 特性 | 效果 |
|------|------|
| **8. 访问频率追踪** | 智能缓存管理 |
| **9. 块级 LRU 优化** | 提升命中率 |

---

## 📋 P0-1: Full Skip Logic 实现

> **核心价值**: 100% 缓存命中时跳过全部 prefill 计算，27x 加速

### 技术架构

```
┌─────────────────────────────────────────────────────────┐
│              Full Skip Logic 数据流                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Request → prefix_cache.match_cache()                  │
│                       │                                 │
│                       ▼                                 │
│            检查缓存命中率                                │
│                       │                                 │
│         ┌─────────────┴─────────────┐                   │
│         ▼                           ▼                   │
│    100% 命中                     < 100%                 │
│         │                           │                   │
│         ▼                           ▼                   │
│  set skip_prefill=True       set skip_prefill=False    │
│         │                           │                   │
│         ▼                           ▼                   │
│  scheduler 跳过 prefill      scheduler 正常 prefill     │
│         │                           │                   │
│         ▼                           ▼                   │
│  直接进入 decode               prefill → decode         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 文件修改清单

#### 1. `src/omlx/cache/prefix_cache.py` (核心修改)

**位置**: `BlockAwarePrefixCache` 类

**新增方法**:
```python
def match_cache_with_skip_logic(
    self,
    token_ids: List[int]
) -> Dict[str, Any]:
    """
    缓存匹配 + Skip Logic 决策

    Returns:
        {
            'block_table': BlockTable,
            'remaining_tokens': List[int],
            'can_skip_prefill': bool,
            'cache_hit_ratio': float,
            'skip_reason': str  # 'full', 'none'
        }
    """
    # Step 1: 现有逻辑 - 找到匹配的 blocks
    block_hash_list = self._compute_block_hashes(token_ids)
    matched_blocks, remaining_hashes = self._find_best_prefix_match(
        block_hash_list
    )

    # Step 2: 计算命中率
    total_blocks = len(block_hash_list)
    matched_count = len(matched_blocks)
    cache_hit_ratio = matched_count / total_blocks if total_blocks > 0 else 0.0

    # Step 3: Full Skip Logic 决策
    can_skip_prefill = False
    skip_reason = 'none'
    remaining_tokens = []

    if cache_hit_ratio == 1.0:
        # ✅ 100% 命中 → Full Skip
        can_skip_prefill = True
        skip_reason = 'full'
        remaining_tokens = []

        logger.info(
            f"✨ FULL SKIP: 100% cache hit "
            f"({matched_count}/{total_blocks} blocks)"
        )
    else:
        # ❌ 部分命中 → 计算 remaining tokens
        matched_tokens = matched_count * self.block_size
        remaining_tokens = token_ids[matched_tokens:]

        logger.debug(
            f"Partial cache hit: {cache_hit_ratio*100:.1f}% "
            f"({matched_count}/{total_blocks} blocks), "
            f"remaining: {len(remaining_tokens)} tokens"
        )

    # Step 4: 构造 block_table
    block_table = BlockTable(blocks=matched_blocks) if matched_blocks else None

    return {
        'block_table': block_table,
        'remaining_tokens': remaining_tokens,
        'can_skip_prefill': can_skip_prefill,
        'cache_hit_ratio': cache_hit_ratio,
        'skip_reason': skip_reason
    }
```

**修改现有方法**:
```python
def match(self, token_ids: List[int]) -> Tuple[Optional[BlockTable], List[int]]:
    """
    保持向后兼容的同时，内部调用新方法
    """
    result = self.match_cache_with_skip_logic(token_ids)
    return result['block_table'], result['remaining_tokens']
```

#### 2. `src/omlx/scheduler.py` (调度逻辑修改)

**位置**: `SchedulerPrefill.schedule()` 方法

**现有代码** (line ~2255):
```python
def schedule(self, request: Request) -> Optional[ScheduledRequest]:
    # ... 现有代码 ...

    # 缓存匹配
    block_table, remaining = self.prefix_cache.match(request.prompt_token_ids)

    # 决定是否重算
    if len(remaining) == 0 and request.cached_tokens > 0:
        # ❌ 问题: Stateful cache 无法 trim，fallback 到重算全部
        if self._cache_list_needs_boundary_snapshot(request.prompt_cache):
            request.prompt_cache = None
            remaining = request.prompt_token_ids
```

**新代码** (修改后):
```python
def schedule(self, request: Request) -> Optional[ScheduledRequest]:
    # ... 现有代码 ...

    # ✅ 新增: 使用 Skip Logic 进行缓存匹配
    cache_result = self.prefix_cache.match_cache_with_skip_logic(
        request.prompt_token_ids
    )

    block_table = cache_result['block_table']
    remaining = cache_result['remaining_tokens']
    can_skip = cache_result['can_skip_prefill']

    # ✅ Full Skip Logic
    if can_skip:
        logger.info(
            f"🎯 FULL SKIP enabled for request {request.request_id} "
            f"(hit ratio: {cache_result['cache_hit_ratio']*100:.1f}%)"
        )

        # 标记跳过 prefill
        request.skip_prefill = True
        request.prompt_cache = block_table
        request.remaining_tokens = []

        # 直接进入 decode 阶段
        # (BatchedEngine 会检查 skip_prefill 标记)
        return ScheduledRequest(
            request=request,
            block_table=block_table,
            num_tokens_to_prefill=0,  # ✅ 跳过 prefill
            skip_compute=True
        )
    else:
        # 正常 prefill 流程
        request.skip_prefill = False
        request.remaining_tokens = remaining

        # ... 现有 prefill 调度逻辑 ...
```

#### 3. `src/omlx/batched_engine.py` (引擎适配)

**位置**: `BatchedEngine` 类

**新增: Request 类属性**:
```python
@dataclass
class Request:
    # ... 现有属性 ...

    # ✅ 新增
    skip_prefill: bool = False  # 是否跳过 prefill 计算
```

**修改: `stream_generate()` 方法**:
```python
async def stream_generate(
    self,
    prompt: str,
    ...
) -> AsyncGenerator[Dict[str, Any], None]:
    # ... tokenize ...

    # ✅ 检查是否可以 skip prefill
    cache_result = self.cache_manager.match_cache_with_skip_logic(tokens)

    if cache_result['can_skip_prefill']:
        # ✅ Full Skip 路径
        logger.info("🚀 FULL SKIP: Skipping prefill computation")

        # 恢复缓存的 KV Cache
        kv_cache = self._restore_kv_cache_from_blocks(
            cache_result['block_table']
        )

        # 直接开始 decode（从最后一个 token）
        last_token = tokens[-1]

        # ✅ 跳过 prefill，直接 decode
        async for output in self._decode_loop(
            prompt_tokens=tokens,
            last_token=last_token,
            kv_cache=kv_cache,
            max_tokens=max_tokens,
            ...
        ):
            yield output
    else:
        # 正常 prefill + decode 流程
        async for output in self._prefill_and_decode(
            tokens=tokens,
            remaining=cache_result['remaining_tokens'],
            ...
        ):
            yield output
```

**新增辅助方法**:
```python
def _restore_kv_cache_from_blocks(
    self,
    block_table: BlockTable
) -> KVCache:
    """
    从 block_table 恢复完整的 KV Cache

    Args:
        block_table: 缓存的 block table

    Returns:
        完整的 KVCache 对象，可直接用于 decode
    """
    # 从 PagedCacheManager 恢复 KV tensors
    kv_cache_data = []

    for block_id in block_table.block_ids:
        # 读取 block 的 KV 数据
        kv_tensor = self.cache_manager.hot_tier.read_block(block_id)
        kv_cache_data.append(kv_tensor)

    # 合并为连续的 KV Cache
    kv_cache = self._concatenate_kv_cache(kv_cache_data)

    return kv_cache

def _concatenate_kv_cache(self, kv_tensors: List[mx.array]) -> KVCache:
    """
    将多个 block 的 KV tensor 合并为连续的 KV Cache
    """
    # 实现 tensor 合并逻辑
    # (取决于 mlx-lm 的 KVCache 格式)
    ...
```

### 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|----------|
| **Stateful 缓存兼容性** | 🔴 高 | 先支持 PagedCache，后续逐步支持 Rotating/Arrays |
| **KV Cache 恢复正确性** | 🟡 中 | 充分测试，对比 skip vs non-skip 输出一致性 |
| **mlx-lm API 变化** | 🟡 中 | 版本锁定 + 兼容性测试 |
| **性能回归** | 🟢 低 | 基准测试验证，失败则回滚 |

### 验证标准

#### 功能验证
- [ ] 100% 缓存命中时触发 Full Skip
- [ ] Full Skip 输出与正常 prefill 输出一致
- [ ] 部分命中时正确回退到正常 prefill
- [ ] 支持所有 cache 类型（优先 PagedCache）

#### 性能验证
- [ ] Agent scenario (4 并发) Generation TPS > 500 tok/s
- [ ] 100% 命中场景 TTFT < 500ms
- [ ] 无性能回退（< 5%）

---

## 📋 P0-2: Approximate Skip 实现

> **核心价值**: 95%+ 命中时零填充 + 跳过，5-10x 加速

### 技术架构

```
Approximate Skip 决策树:

  缓存命中率 >= 95%?
         │
    ┌────┴────┐
    YES       NO
    │         │
    ▼         ▼
95-99% 命中   正常 prefill
    │
    ▼
识别缺失的 blocks
    │
    ▼
零填充缺失的 KV
    │
    ▼
设置 skip_prefill=True
    │
    ▼
跳过 prefill 计算
```

### 文件修改清单

#### 1. `src/omlx/cache/prefix_cache.py` (扩展 Skip Logic)

**修改方法**: `match_cache_with_skip_logic()`

```python
def match_cache_with_skip_logic(
    self,
    token_ids: List[int]
) -> Dict[str, Any]:
    # ... Step 1-2: 现有逻辑 (计算命中率) ...

    # ✅ 新增: Approximate Skip Logic
    APPROX_SKIP_THRESHOLD = 0.95  # 可配置

    if cache_hit_ratio >= APPROX_SKIP_THRESHOLD and cache_hit_ratio < 1.0:
        # ✅ 95-99% 命中 → Approximate Skip
        missing_blocks = total_blocks - matched_count

        logger.info(
            f"⚡ APPROXIMATE SKIP: {cache_hit_ratio*100:.1f}% hit "
            f"({matched_count}/{total_blocks} blocks), "
            f"zero-filling {missing_blocks} blocks"
        )

        can_skip_prefill = True
        skip_reason = 'approximate'

        # 记录缺失的 block 位置
        missing_block_indices = [
            i for i in range(total_blocks)
            if i >= matched_count
        ]

        return {
            'block_table': block_table,
            'remaining_tokens': [],  # ✅ 零填充，无需重算
            'can_skip_prefill': True,
            'cache_hit_ratio': cache_hit_ratio,
            'skip_reason': 'approximate',
            'missing_block_indices': missing_block_indices
        }

    # ... 现有 Full Skip 和 Normal 逻辑 ...
```

#### 2. `src/omlx/batched_engine.py` (零填充实现)

**新增方法**:
```python
def _restore_kv_cache_with_zero_fill(
    self,
    block_table: BlockTable,
    missing_block_indices: List[int],
    total_blocks: int
) -> KVCache:
    """
    恢复 KV Cache，对缺失的 blocks 零填充

    Args:
        block_table: 已缓存的 blocks
        missing_block_indices: 缺失的 block 索引列表
        total_blocks: 总 block 数量

    Returns:
        完整的 KVCache（包含零填充部分）
    """
    kv_cache_data = []

    for i in range(total_blocks):
        if i < len(block_table.block_ids):
            # ✅ 命中: 从缓存读取
            block_id = block_table.block_ids[i]
            kv_tensor = self.cache_manager.hot_tier.read_block(block_id)
        else:
            # ❌ 未命中: 零填充
            # 创建形状匹配的零 tensor
            # 假设 block_size = 256, n_layers = 32, d_model = 4096
            zero_kv = mx.zeros(
                (self.config.block_size, self.config.n_layers, self.config.d_model),
                dtype=mx.float16
            )
            kv_tensor = zero_kv

            logger.debug(f"Zero-filling block {i}")

        kv_cache_data.append(kv_tensor)

    # 合并为连续的 KV Cache
    kv_cache = self._concatenate_kv_cache(kv_cache_data)

    return kv_cache
```

**修改**: `stream_generate()` 方法

```python
async def stream_generate(...):
    cache_result = self.cache_manager.match_cache_with_skip_logic(tokens)

    if cache_result['can_skip_prefill']:
        skip_reason = cache_result['skip_reason']

        if skip_reason == 'full':
            # ✅ Full Skip
            kv_cache = self._restore_kv_cache_from_blocks(
                cache_result['block_table']
            )
        elif skip_reason == 'approximate':
            # ✅ Approximate Skip (零填充)
            kv_cache = self._restore_kv_cache_with_zero_fill(
                cache_result['block_table'],
                cache_result['missing_block_indices'],
                len(tokens) // self.config.block_size + 1
            )

        # 跳过 prefill，直接 decode
        async for output in self._decode_loop(...):
            yield output
    else:
        # 正常流程
        ...
```

### 质量影响评估

| 命中率 | 零填充比例 | 质量影响 | 使用场景 |
|--------|-----------|---------|----------|
| 99% | 1% | < 0.5% | 推荐 |
| 97% | 3% | ~1% | 可接受 |
| 95% | 5% | ~2% | 临界值 |
| < 95% | > 5% | > 3% | 不建议 |

### 配置参数

```python
# src/omlx/config.py

class CacheConfig:
    # Approximate Skip 阈值
    APPROX_SKIP_THRESHOLD: float = 0.95  # 95%

    # 是否启用 Approximate Skip
    ENABLE_APPROXIMATE_SKIP: bool = True

    # 质量监控: 如果输出质量下降超过此阈值，禁用 Approximate Skip
    MAX_QUALITY_DEGRADATION: float = 0.02  # 2%
```

---

## 📋 P0-3: Hybrid Hashing 实现

> **核心价值**: xxHash64 快 50x，双重哈希支持前缀重叠检测 (3-7x)

### 当前问题

**oMLX 现状** (`src/omlx/cache/prefix_cache.py`):
```python
import hashlib

def compute_block_hash(token_ids: List[int]) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(bytes(str(tuple(token_ids)), "utf-8"))
    return hasher.digest()  # ~200 MB/s
```

**性能瓶颈**:
- SHA256: ~200 MB/s
- xxHash64: ~10 GB/s
- **50x 速度差距**

### 实现方案

#### 1. 安装 xxHash

```bash
# requirements.txt
xxhash>=3.0.0
```

#### 2. 修改 `src/omlx/cache/prefix_cache.py`

**替换哈希函数**:
```python
import xxhash

class BlockAwarePrefixCache:
    def __init__(self, ...):
        # ... 现有初始化 ...

        # ✅ 新增: 使用 xxHash64
        self.use_xxhash = True

    def _compute_single_block_hash(
        self,
        token_ids: List[int],
        block_position: int = 0  # ✅ 新增: 位置信息
    ) -> int:
        """
        计算单个 block 的 hybrid hash

        Args:
            token_ids: block 的 token 列表
            block_position: block 在序列中的位置

        Returns:
            64-bit hash value
        """
        if self.use_xxhash:
            # ✅ xxHash64 (快)
            # Content hash
            token_bytes = bytes(token_ids)
            content_hash = xxhash.xxh64(token_bytes).intdigest()

            # Position hash (支持前缀重叠检测)
            position_bytes = block_position.to_bytes(4, byteorder='little')
            position_hash = xxhash.xxh64(position_bytes).intdigest()

            # Hybrid hash (XOR 组合)
            hybrid_hash = content_hash ^ position_hash

            return hybrid_hash
        else:
            # 保留旧的 SHA256 (慢，向后兼容)
            hasher = hashlib.sha256()
            hasher.update(bytes(str(tuple(token_ids)), "utf-8"))
            return int.from_bytes(hasher.digest()[:8], byteorder='big')

    def _compute_block_hashes(
        self,
        all_token_ids: List[int]
    ) -> List[int]:
        """
        计算所有 blocks 的 hashes

        Returns:
            List of 64-bit hash values
        """
        hashes = []
        num_blocks = (len(all_token_ids) + self.block_size - 1) // self.block_size

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(all_token_ids))
            block_tokens = all_token_ids[start:end]

            # ✅ 传入 position 信息
            block_hash = self._compute_single_block_hash(
                block_tokens,
                block_position=block_idx
            )
            hashes.append(block_hash)

        return hashes
```

#### 3. 修改 Hash Table 存储

**现有代码** (`src/omlx/cache/paged_cache.py`):
```python
# 当前: 使用 bytes 作为 key
self._hash_to_blocks: Dict[bytes, List[BlockId]] = {}
```

**新代码**:
```python
# ✅ 改用 int (xxHash64 返回 64-bit int)
self._hash_to_blocks: Dict[int, List[BlockId]] = {}

def insert_block(self, block_hash: int, block_id: BlockId):
    """
    插入 block 到 hash table

    Args:
        block_hash: 64-bit int hash
        block_id: block ID
    """
    if block_hash not in self._hash_to_blocks:
        self._hash_to_blocks[block_hash] = []

    self._hash_to_blocks[block_hash].append(block_id)

def lookup_block(self, block_hash: int) -> Optional[List[BlockId]]:
    """
    查找 block

    Returns:
        匹配的 block IDs（可能有多个，hash 冲突）
    """
    return self._hash_to_blocks.get(block_hash)
```

### 前缀重叠检测 (Prefix Overlap Detection)

**使用场景**:
```
Request 1: "Explain Python list comprehensions in detail"
Request 2: "Explain Python list comprehensions and generators"

前缀重叠: "Explain Python list comprehensions"
```

**Position-aware Hashing 优势**:
- Content-only hash: 相同内容在不同位置的 hash 相同
- **问题**: 无法区分位置，导致误匹配
- **Position-aware hash**: 相同内容在不同位置的 hash 不同
- **优势**: 精确匹配位置，3-7x 前缀重叠场景加速

### 性能对比

| 哈希算法 | 速度 | 功能 |
|----------|------|------|
| SHA256 (current) | ~200 MB/s | Content-only |
| xxHash64 Content | ~10 GB/s | Content-only |
| xxHash64 Hybrid | ~10 GB/s | Content + Position |

**预期提升**:
- 哈希速度: 50x
- 前缀重叠场景: 3-7x

---

## 📋 P0-4: SSD Compression 实现

> **核心价值**: 2-4x I/O 加速 + 2-4x 存储节省

### 当前问题

**oMLX 现状** (`src/omlx/cache/paged_ssd_cache.py`):
- 直接写入原始 safetensors
- 无压缩
- I/O 吞吐受限于 SSD 写入速度

**ThunderLLAMA 优势**:
- zlib compression (level 1)
- 实测: 3.9x 压缩比
- I/O 时间: 1.2s → 0.3s (4x)

### 实现方案

#### 1. 安装依赖

```bash
# requirements.txt
# (zlib 是 Python 标准库，无需安装)
```

#### 2. 修改 `src/omlx/cache/paged_ssd_cache.py`

**新增配置**:
```python
import zlib

class PagedSSDCacheManager:
    def __init__(
        self,
        cache_dir: str,
        max_size_bytes: int,
        enable_compression: bool = True,  # ✅ 新增
        compression_level: int = 1,        # ✅ 新增 (1=fastest)
        ...
    ):
        self.enable_compression = enable_compression
        self.compression_level = compression_level

        # 统计信息
        self.compression_stats = {
            'total_compressed_bytes': 0,
            'total_uncompressed_bytes': 0,
            'compression_ratio': 1.0
        }
```

**修改写入方法**:
```python
def write_block(
    self,
    block_id: BlockId,
    kv_tensor: mx.array
) -> bool:
    """
    写入 block 到 SSD (带压缩)

    Args:
        block_id: block ID
        kv_tensor: KV cache tensor

    Returns:
        True if success
    """
    try:
        # Step 1: Serialize tensor to bytes
        tensor_bytes = self._serialize_tensor(kv_tensor)
        uncompressed_size = len(tensor_bytes)

        # Step 2: Compress (如果启用)
        if self.enable_compression:
            compressed_bytes = zlib.compress(
                tensor_bytes,
                level=self.compression_level
            )
            compressed_size = len(compressed_bytes)

            # 更新统计
            self.compression_stats['total_uncompressed_bytes'] += uncompressed_size
            self.compression_stats['total_compressed_bytes'] += compressed_size
            self.compression_stats['compression_ratio'] = (
                self.compression_stats['total_uncompressed_bytes'] /
                max(self.compression_stats['total_compressed_bytes'], 1)
            )

            logger.debug(
                f"Compressed block {block_id}: "
                f"{uncompressed_size} → {compressed_size} bytes "
                f"({compressed_size/uncompressed_size*100:.1f}%)"
            )

            data_to_write = compressed_bytes
        else:
            data_to_write = tensor_bytes

        # Step 3: Write to disk
        file_path = self._get_block_file_path(block_id)
        with open(file_path, 'wb') as f:
            f.write(data_to_write)

        return True
    except Exception as e:
        logger.error(f"Failed to write block {block_id}: {e}")
        return False
```

**修改读取方法**:
```python
def read_block(
    self,
    block_id: BlockId
) -> Optional[mx.array]:
    """
    从 SSD 读取 block (带解压)

    Returns:
        KV tensor or None if not found
    """
    try:
        file_path = self._get_block_file_path(block_id)

        if not os.path.exists(file_path):
            return None

        # Step 1: Read from disk
        with open(file_path, 'rb') as f:
            data_bytes = f.read()

        # Step 2: Decompress (如果启用)
        if self.enable_compression:
            decompressed_bytes = zlib.decompress(data_bytes)
            logger.debug(
                f"Decompressed block {block_id}: "
                f"{len(data_bytes)} → {len(decompressed_bytes)} bytes"
            )
            tensor_bytes = decompressed_bytes
        else:
            tensor_bytes = data_bytes

        # Step 3: Deserialize to tensor
        kv_tensor = self._deserialize_tensor(tensor_bytes)

        return kv_tensor
    except Exception as e:
        logger.error(f"Failed to read block {block_id}: {e}")
        return None
```

**新增辅助方法**:
```python
def _serialize_tensor(self, tensor: mx.array) -> bytes:
    """
    序列化 mlx array 为 bytes
    """
    # 使用 mlx 的序列化方法
    # (具体实现取决于 mlx API)
    import io
    buffer = io.BytesIO()
    mx.save(buffer, tensor)
    return buffer.getvalue()

def _deserialize_tensor(self, data: bytes) -> mx.array:
    """
    反序列化 bytes 为 mlx array
    """
    import io
    buffer = io.BytesIO(data)
    return mx.load(buffer)

def get_compression_stats(self) -> Dict[str, Any]:
    """
    获取压缩统计信息
    """
    return {
        'enabled': self.enable_compression,
        'level': self.compression_level,
        'total_compressed_mb': self.compression_stats['total_compressed_bytes'] / 1024**2,
        'total_uncompressed_mb': self.compression_stats['total_uncompressed_bytes'] / 1024**2,
        'compression_ratio': self.compression_stats['compression_ratio'],
        'space_saved_mb': (
            self.compression_stats['total_uncompressed_bytes'] -
            self.compression_stats['total_compressed_bytes']
        ) / 1024**2
    }
```

### 配置参数

```python
# src/omlx/config.py

class CacheConfig:
    # SSD 压缩配置
    ENABLE_SSD_COMPRESSION: bool = True
    SSD_COMPRESSION_LEVEL: int = 1  # 1=fastest, 9=best compression

    # 压缩阈值 (小于此大小的 block 不压缩)
    MIN_BLOCK_SIZE_FOR_COMPRESSION: int = 1024  # 1KB
```

### 性能预测

| 指标 | 无压缩 | 有压缩 (level 1) | 提升 |
|------|--------|-----------------|------|
| **写入速度** | ~500 MB/s | ~1800 MB/s | 3.6x |
| **读取速度** | ~600 MB/s | ~2200 MB/s | 3.7x |
| **存储占用** | 2.4 GB | 620 MB | 3.9x |
| **CPU 开销** | 0% | ~5% | 可接受 |

---

## 🧪 测试验证计划

### 功能测试

#### Test 1: Full Skip Logic
```python
def test_full_skip_logic():
    """测试 100% 缓存命中时的 Full Skip"""

    # Setup
    engine = BatchedEngine(...)
    prompt = "Explain Python list comprehensions in detail"

    # First run (cold cache)
    result1 = await engine.stream_generate(prompt, max_tokens=128)
    assert result1['cache_hit_ratio'] == 0.0
    assert result1['skip_prefill'] == False
    ttft1 = result1['ttft_ms']

    # Second run (hot cache, 100% hit)
    result2 = await engine.stream_generate(prompt, max_tokens=128)
    assert result2['cache_hit_ratio'] == 1.0
    assert result2['skip_prefill'] == True
    assert result2['skip_reason'] == 'full'
    ttft2 = result2['ttft_ms']

    # Verify speedup
    assert ttft2 < ttft1 / 10, "Full Skip should give > 10x TTFT speedup"

    # Verify output consistency
    assert result1['text'] == result2['text'], "Output should be identical"
```

#### Test 2: Approximate Skip
```python
def test_approximate_skip():
    """测试 95%+ 缓存命中时的 Approximate Skip"""

    engine = BatchedEngine(...)

    # First run
    prompt1 = "Explain Python " + "a" * 1000  # 填充到 95% 的长度
    result1 = await engine.stream_generate(prompt1, max_tokens=128)

    # Second run (95% overlap)
    prompt2 = "Explain Python " + "a" * 1000 + "b" * 50  # 额外 5%
    result2 = await engine.stream_generate(prompt2, max_tokens=128)

    # Verify Approximate Skip triggered
    assert result2['cache_hit_ratio'] >= 0.95
    assert result2['cache_hit_ratio'] < 1.0
    assert result2['skip_prefill'] == True
    assert result2['skip_reason'] == 'approximate'

    # Verify quality (should be close to normal prefill)
    result3 = await engine.stream_generate(prompt2, max_tokens=128, disable_cache=True)
    similarity = compute_similarity(result2['text'], result3['text'])
    assert similarity >= 0.98, "Approximate Skip quality should be > 98%"
```

#### Test 3: Hybrid Hashing
```python
def test_hybrid_hashing_speed():
    """测试 xxHash64 vs SHA256 速度"""

    import time

    tokens = list(range(256))  # 1 block

    # SHA256
    start = time.perf_counter()
    for _ in range(10000):
        compute_block_hash_sha256(tokens)
    sha256_time = time.perf_counter() - start

    # xxHash64
    start = time.perf_counter()
    for _ in range(10000):
        compute_block_hash_xxhash64(tokens)
    xxhash_time = time.perf_counter() - start

    speedup = sha256_time / xxhash_time
    assert speedup > 20, f"xxHash64 should be > 20x faster (got {speedup:.1f}x)"
```

#### Test 4: SSD Compression
```python
def test_ssd_compression():
    """测试 SSD 压缩功能"""

    ssd_cache = PagedSSDCacheManager(
        cache_dir="/tmp/test_cache",
        enable_compression=True,
        compression_level=1
    )

    # 创建测试 tensor
    kv_tensor = mx.random.normal((256, 32, 4096))  # block_size=256, layers=32, d_model=4096

    # 写入
    block_id = "test_block_1"
    success = ssd_cache.write_block(block_id, kv_tensor)
    assert success

    # 读取
    restored_tensor = ssd_cache.read_block(block_id)
    assert restored_tensor is not None

    # 验证一致性
    assert mx.allclose(kv_tensor, restored_tensor), "Compressed data should match original"

    # 验证压缩率
    stats = ssd_cache.get_compression_stats()
    assert stats['compression_ratio'] > 2.0, "Should achieve > 2x compression"
```

### 性能测试

#### Benchmark 1: Agent Scenario (4 并发)
```python
async def benchmark_agent_scenario():
    """
    运行 Agent scenario benchmark

    Expected results (after migration):
    - Generation TPS: > 500 tok/s (current: 119.3)
    - Cache hit rate: > 95%
    - Skip rate: > 90%
    """
    results = await run_concurrent_requests(
        client=client,
        model_id="qwen3.5-35b-mlx",
        num_concurrent=4
    )

    assert results['gen_tps'] > 500, \
        f"Generation TPS should be > 500 (got {results['gen_tps']:.1f})"

    assert results['cache_hit_rate'] > 0.95, \
        f"Cache hit rate should be > 95% (got {results['cache_hit_rate']*100:.1f}%)"

    print(f"✅ Benchmark passed: {results['gen_tps']:.1f} tok/s")
```

#### Benchmark 2: 性能对比
```bash
# Before migration
python benchmark_omlx.py
# Expected: ~119 tok/s

# After migration
python benchmark_omlx.py
# Target: > 500 tok/s

# Improvement: 4-5x
```

---

## 📅 实施时间表

### Week 1 (3-4 天)

| 日期 | 任务 | 负责人 | 产出 |
|------|------|--------|------|
| Day 1 | P0-1: Full Skip Logic 实现 | 建设者 (GLM-5) | `prefix_cache.py`, `scheduler.py`, `batched_engine.py` 修改完成 |
| Day 2 | P0-1: Full Skip Logic 测试 + P0-2: Approximate Skip 开始 | 建设者 + 测试者 | 测试通过，Approximate Skip 完成 50% |
| Day 3 | P0-2: Approximate Skip 完成 + P0-3: Hybrid Hashing | 建设者 | Approximate Skip 测试通过，xxHash64 集成完成 |
| Day 4 | P0-3: Hybrid Hashing 测试 + P0-4: SSD Compression | 建设者 | 全部 P0 特性完成 |

### Week 2 (1 天)

| 日期 | 任务 | 负责人 | 产出 |
|------|------|--------|------|
| Day 5 | 集成测试 + 性能验证 | 测试者 + 审查者 | Benchmark 报告 |

---

## 🚨 风险管理

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 | 应急预案 |
|------|------|------|----------|----------|
| **mlx-lm KV Cache 格式不兼容** | 中 | 高 | 深入研究 mlx-lm 源码，编写适配层 | 回退到 Partial Skip (只优化部分场景) |
| **Stateful 缓存无法支持 Skip** | 高 | 中 | 优先支持 PagedCache，记录限制 | 仅 PagedCache 启用 Skip Logic |
| **零填充质量下降过大** | 低 | 中 | 充分测试，动态调整阈值 | 降低 APPROX_SKIP_THRESHOLD 到 0.97 |
| **xxHash64 在 Apple Silicon 上性能不佳** | 低 | 低 | 使用 Python xxhash 库（已优化） | 保留 SHA256 作为 fallback |
| **SSD 压缩 CPU 开销过大** | 低 | 低 | 使用 level 1 (fastest) | 提供配置开关，允许禁用 |

### 项目风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **时间估计不足** | 中 | 中 | 预留 20% buffer，优先实现 P0-1 |
| **依赖库版本冲突** | 低 | 低 | 版本锁定 + 虚拟环境隔离 |
| **性能提升未达预期** | 中 | 高 | 基于 ThunderLLAMA 实测数据，保守估计 |

---

## 📊 成功标准

### 功能标准
- [ ] Full Skip Logic: 100% 命中时触发，输出一致性 100%
- [ ] Approximate Skip: 95%+ 命中时触发，输出一致性 > 98%
- [ ] Hybrid Hashing: 哈希速度 > 20x (vs SHA256)
- [ ] SSD Compression: 压缩比 > 2x，I/O 加速 > 2x

### 性能标准
- [ ] **Generation TPS**: > 500 tok/s (当前 119.3 tok/s)
- [ ] **Prefill TPS**: > 200 tok/s (当前 40.1 tok/s)
- [ ] **TTFT (100% 命中)**: < 500ms (当前 3772ms)
- [ ] **Cache hit rate**: > 95% (Agent scenario)
- [ ] **Skip rate**: > 90%

### 质量标准
- [ ] 测试覆盖率 > 80%
- [ ] 无性能回退 (< 5%)
- [ ] 代码审查通过
- [ ] 文档完整（API + 配置 + 故障排查）

---

## 📚 参考资料

### 源代码分析
- [CACHE_COMPARISON.md](./CACHE_COMPARISON.md) - oMLX vs ThunderLLAMA 缓存架构对比
- [THUNDERLLAMA_SKIP_LOGIC_ANALYSIS.md](./THUNDERLLAMA_SKIP_LOGIC_ANALYSIS.md) - ThunderLLAMA Skip Logic 深度分析
- [~/ThunderLLAMA/src/llama-context.cpp](~/ThunderLLAMA/src/llama-context.cpp) - ThunderLLAMA 源码
- [src/omlx/cache/](./src/omlx/cache/) - oMLX 缓存实现

### 性能数据
- [benchmark_results.json](./benchmark_results.json) - oMLX 基准测试结果
- [~/ThunderLLAMA/OPTIMIZATION_FEATURES.md](~/ThunderLLAMA/OPTIMIZATION_FEATURES.md) - ThunderLLAMA 优化特性列表

---

*实施计划版本: v1.0*
*创建日期: 2026-03-13*
*预计完成: 2026-03-18*
