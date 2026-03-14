# P1-6: Checksum Validation 实现设计

> **目标**: 为 SSD 缓存添加数据完整性校验，防止缓存损坏
> **预计工期**: 0.5 天

---

## 📊 核心原理

### ThunderLLAMA 实现分析

ThunderLLAMA 在 L3 (SSD) 缓存中使用 XXH64 checksum 验证数据完整性：

```cpp
// thunder-lmcache-storage.cpp

// 1. 写入时计算 checksum
struct BlockHeader {
    uint64_t magic;         // 魔数
    uint64_t checksum;      // XXH64 checksum
    size_t k_size;          // key tensor 大小
    size_t v_size;          // value tensor 大小
};

// 保存块
void save_block(block_id, k_data, v_data) {
    // 计算 checksum
    uint64_t checksum = XXH64(k_data, k_size, 0);
    checksum ^= XXH64(v_data, v_size, 0);  // XOR 组合

    // 写入 header
    BlockHeader header = {
        .magic = 0x4C4D43414348,  // "LMCACHE"
        .checksum = checksum,
        .k_size = k_size,
        .v_size = v_size
    };

    fwrite(&header, sizeof(header), 1, file);
    fwrite(k_data, k_size, 1, file);
    fwrite(v_data, v_size, 1, file);
}

// 2. 读取时验证 checksum
bool load_block(block_id) {
    // 读取 header
    BlockHeader header;
    fread(&header, sizeof(header), 1, file);

    // 验证魔数
    if (header.magic != 0x4C4D43414348) {
        return false;  // 损坏
    }

    // 读取数据
    fread(k_data, header.k_size, 1, file);
    fread(v_data, header.v_size, 1, file);

    // 验证 checksum
    uint64_t actual = XXH64(k_data, header.k_size, 0);
    actual ^= XXH64(v_data, header.v_size, 0);

    if (actual != header.checksum) {
        fprintf(stderr, "❌ Checksum mismatch! Expected %lx, got %lx\n",
                header.checksum, actual);
        return false;  // 损坏
    }

    return true;  // 验证通过
}
```

### 关键特性

1. **XXH64 哈希**: 快速（~10 GB/s），适合大数据块
2. **XOR 组合**: 多个 tensor 的 checksum 通过 XOR 组合
3. **魔数验证**: 快速判断文件格式正确性
4. **损坏自动丢弃**: 验证失败时自动删除损坏文件

---

## 🏗️ oMLX 实现方案

### 当前状态

oMLX 目前使用 safetensors 格式存储缓存块，**无 checksum 验证**：

```python
# src/omlx/cache/paged_ssd_cache.py

def save_block(block_hash, cache_data, ...):
    # 序列化为 safetensors
    mx.save_safetensors(file_path, arrays, metadata)

    # ❌ 无 checksum 计算
    # ❌ 无数据完整性验证

def load_block(block_hash):
    # 加载 safetensors
    arrays, metadata = mx.load(file_path)

    # ❌ 无 checksum 验证
    # ❌ 无损坏检测
```

### 实现策略

**方案选择**: 在 safetensors metadata 中嵌入 checksum

**优势**:
- ✅ 不破坏现有 safetensors 格式
- ✅ 兼容已有缓存文件
- ✅ 利用 safetensors 的 metadata 字段

**架构**:

```
┌─────────────────────────────────────────────────────────┐
│              Safetensors File Format                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Header (8 bytes)                                       │
│  ├─ Metadata Length                                     │
│                                                         │
│  Metadata (JSON)                                        │
│  ├─ layer_0_k: {...}                                    │
│  ├─ layer_0_v: {...}                                    │
│  ├─ "omlx_checksum": "xxhash64"  ← ✅ 新增            │
│  ├─ "omlx_checksum_value": "1a2b3c4d..."  ← ✅ 新增   │
│  └─ "omlx_checksum_algo": "xxh64"  ← ✅ 新增          │
│                                                         │
│  Tensor Data                                            │
│  ├─ layer_0_k (raw bytes)                               │
│  ├─ layer_0_v (raw bytes)                               │
│  └─ ...                                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 实现清单

### 1. 添加 xxHash 依赖

**文件**: `requirements.txt`

```txt
xxhash>=3.0.0  # P0-3 Hybrid Hashing 已添加
```

### 2. Checksum 计算工具

**文件**: `src/omlx/cache/checksum.py` (新建)

```python
"""
缓存块 checksum 计算和验证工具。

P1-6: Checksum Validation
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    xxhash = None

logger = logging.getLogger(__name__)


class ChecksumCalculator:
    """
    缓存块 checksum 计算器。

    使用 xxHash64 算法，快速且高质量。
    """

    ALGO_XXH64 = "xxh64"

    def __init__(self, algorithm: str = ALGO_XXH64):
        """
        初始化 checksum 计算器。

        Args:
            algorithm: 哈希算法（默认 xxh64）
        """
        if not HAS_XXHASH and algorithm == self.ALGO_XXH64:
            raise ImportError("xxhash library not found, install with: pip install xxhash")

        self.algorithm = algorithm

    def compute_tensors_checksum(
        self,
        tensors_raw: Dict[str, tuple]
    ) -> str:
        """
        计算多个 tensor 的组合 checksum。

        Args:
            tensors_raw: {name: (raw_bytes, dtype_str, shape)} 字典

        Returns:
            Checksum 字符串（十六进制）
        """
        if self.algorithm == self.ALGO_XXH64:
            return self._compute_xxh64(tensors_raw)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _compute_xxh64(
        self,
        tensors_raw: Dict[str, tuple]
    ) -> str:
        """
        使用 XXH64 计算 checksum。

        策略：对每个 tensor 计算 XXH64，然后 XOR 组合。
        """
        combined_hash = 0

        # 按名称排序确保一致性
        for name in sorted(tensors_raw.keys()):
            raw_bytes, dtype_str, shape = tensors_raw[name]

            # 计算此 tensor 的 XXH64
            tensor_hash = xxhash.xxh64(raw_bytes).intdigest()

            # XOR 组合
            combined_hash ^= tensor_hash

        # 转换为十六进制字符串
        return f"{combined_hash:016x}"

    def verify_checksum(
        self,
        tensors_raw: Dict[str, tuple],
        expected_checksum: str
    ) -> bool:
        """
        验证 checksum 是否匹配。

        Args:
            tensors_raw: {name: (raw_bytes, dtype_str, shape)} 字典
            expected_checksum: 预期的 checksum（十六进制）

        Returns:
            True if matches, False otherwise
        """
        actual_checksum = self.compute_tensors_checksum(tensors_raw)

        if actual_checksum != expected_checksum:
            logger.warning(
                f"❌ Checksum mismatch! Expected {expected_checksum}, "
                f"got {actual_checksum}"
            )
            return False

        logger.debug(f"✅ Checksum verified: {actual_checksum}")
        return True


def add_checksum_to_metadata(
    metadata: Dict[str, str],
    tensors_raw: Dict[str, tuple]
) -> Dict[str, str]:
    """
    添加 checksum 到 safetensors metadata。

    Args:
        metadata: 现有 metadata 字典
        tensors_raw: tensor 原始数据

    Returns:
        更新后的 metadata
    """
    calculator = ChecksumCalculator()

    # 计算 checksum
    checksum_value = calculator.compute_tensors_checksum(tensors_raw)

    # 添加到 metadata
    metadata_with_checksum = metadata.copy()
    metadata_with_checksum["omlx_checksum"] = "enabled"
    metadata_with_checksum["omlx_checksum_value"] = checksum_value
    metadata_with_checksum["omlx_checksum_algo"] = calculator.ALGO_XXH64

    return metadata_with_checksum


def verify_checksum_from_metadata(
    metadata: Dict[str, str],
    tensors_raw: Dict[str, tuple]
) -> bool:
    """
    从 safetensors metadata 中验证 checksum。

    Args:
        metadata: Safetensors metadata
        tensors_raw: Tensor 原始数据

    Returns:
        True if checksum valid or not present, False if mismatch
    """
    # 检查是否启用 checksum
    if "omlx_checksum" not in metadata or metadata["omlx_checksum"] != "enabled":
        logger.debug("Checksum not enabled for this block")
        return True  # 无 checksum，认为通过

    # 获取预期 checksum
    expected_checksum = metadata.get("omlx_checksum_value")
    if not expected_checksum:
        logger.warning("Checksum enabled but value missing")
        return True  # 元数据损坏，但不阻止加载

    # 验证
    calculator = ChecksumCalculator()
    return calculator.verify_checksum(tensors_raw, expected_checksum)
```

---

### 3. 集成到 PagedSSDCacheManager

**文件**: `src/omlx/cache/paged_ssd_cache.py` (修改)

#### 修改 __init__

```python
def __init__(
    self,
    ...
    enable_checksum: bool = True,  # ✅ 新增
):
    """
    Args:
        ...
        enable_checksum: Enable checksum validation (P1-6).
            Default True (data integrity protection).
    """
    ...
    self.enable_checksum = enable_checksum

    if enable_checksum:
        logger.info("✅ P1-6 Checksum Validation enabled (xxh64)")
    else:
        logger.info("Checksum validation disabled")
```

#### 修改 save_block (添加 checksum)

```python
def save_block(...):
    ...

    # 提取 tensor 原始字节
    tensors_raw = {}
    for name, arr in arrays.items():
        tensors_raw[name] = _extract_tensor_bytes(arr)

    # ✅ P1-6: 添加 checksum 到 metadata
    if self.enable_checksum:
        from .checksum import add_checksum_to_metadata
        metadata = add_checksum_to_metadata(metadata, tensors_raw)

    # 写入队列
    self._write_queue.put_nowait(
        (block_hash, tensors_raw, metadata, file_path)
    )
    ...
```

#### 修改 load_block (验证 checksum)

```python
def load_block(block_hash):
    ...

    # 加载 safetensors
    arrays, file_metadata = mx.load(str(file_path), return_metadata=True)

    # 提取原始字节（用于 checksum 验证）
    if self.enable_checksum:
        tensors_raw = {}
        for name, arr in arrays.items():
            mx.eval(arr)
            tensors_raw[name] = _extract_tensor_bytes(arr)

        # ✅ P1-6: 验证 checksum
        from .checksum import verify_checksum_from_metadata

        if not verify_checksum_from_metadata(file_metadata, tensors_raw):
            # ❌ Checksum 验证失败
            logger.error(
                f"❌ Checksum validation failed for block {block_hash.hex()[:16]}..."
            )
            self._stats["checksum_failures"] += 1

            # 删除损坏的缓存文件
            self._index.remove(block_hash)
            try:
                file_path.unlink()
                logger.info(f"Deleted corrupted cache file: {file_path}")
            except Exception:
                pass

            return None  # 验证失败，返回 None

        self._stats["checksum_verifications"] += 1

    # 重建 cache_data
    cache_data = self._reconstruct_cache_data(...)
    ...
```

#### 新增统计字段

```python
def __init__(...):
    ...
    self._stats = {
        ...
        "checksum_verifications": 0,  # ✅ 新增
        "checksum_failures": 0,        # ✅ 新增
    }
```

---

## 🧪 测试验证

### 功能测试

**文件**: `tests/test_checksum_validation.py` (新建)

```python
import pytest
from omlx.cache.checksum import (
    ChecksumCalculator,
    add_checksum_to_metadata,
    verify_checksum_from_metadata
)


def test_checksum_calculator():
    """测试 checksum 计算器"""
    calculator = ChecksumCalculator()

    # 模拟 tensor 数据
    tensors_raw = {
        "layer_0_k": (b"key_data_0" * 100, "F16", [10, 10]),
        "layer_0_v": (b"value_data_0" * 100, "F16", [10, 10]),
    }

    # 计算 checksum
    checksum1 = calculator.compute_tensors_checksum(tensors_raw)

    # 相同数据应该得到相同 checksum
    checksum2 = calculator.compute_tensors_checksum(tensors_raw)
    assert checksum1 == checksum2

    # 不同数据应该得到不同 checksum
    tensors_raw_modified = {
        "layer_0_k": (b"key_data_MODIFIED" * 100, "F16", [10, 10]),
        "layer_0_v": (b"value_data_0" * 100, "F16", [10, 10]),
    }
    checksum3 = calculator.compute_tensors_checksum(tensors_raw_modified)
    assert checksum1 != checksum3


def test_add_checksum_to_metadata():
    """测试添加 checksum 到 metadata"""
    metadata = {
        "num_layers": "32",
        "block_size": "1024",
    }

    tensors_raw = {
        "layer_0_k": (b"key_data" * 100, "F16", [10, 10]),
        "layer_0_v": (b"value_data" * 100, "F16", [10, 10]),
    }

    # 添加 checksum
    metadata_with_checksum = add_checksum_to_metadata(metadata, tensors_raw)

    # 验证 metadata 包含 checksum 字段
    assert "omlx_checksum" in metadata_with_checksum
    assert metadata_with_checksum["omlx_checksum"] == "enabled"
    assert "omlx_checksum_value" in metadata_with_checksum
    assert "omlx_checksum_algo" in metadata_with_checksum
    assert metadata_with_checksum["omlx_checksum_algo"] == "xxh64"


def test_verify_checksum_from_metadata():
    """测试从 metadata 验证 checksum"""
    tensors_raw = {
        "layer_0_k": (b"key_data" * 100, "F16", [10, 10]),
        "layer_0_v": (b"value_data" * 100, "F16", [10, 10]),
    }

    # 创建带 checksum 的 metadata
    metadata = {"num_layers": "32"}
    metadata_with_checksum = add_checksum_to_metadata(metadata, tensors_raw)

    # ✅ 验证应该通过（相同数据）
    assert verify_checksum_from_metadata(metadata_with_checksum, tensors_raw) is True

    # ❌ 验证应该失败（数据被篡改）
    tensors_raw_corrupted = {
        "layer_0_k": (b"CORRUPTED" * 100, "F16", [10, 10]),
        "layer_0_v": (b"value_data" * 100, "F16", [10, 10]),
    }
    assert verify_checksum_from_metadata(metadata_with_checksum, tensors_raw_corrupted) is False


def test_checksum_backward_compatibility():
    """测试向后兼容（旧缓存文件无 checksum）"""
    tensors_raw = {
        "layer_0_k": (b"key_data" * 100, "F16", [10, 10]),
        "layer_0_v": (b"value_data" * 100, "F16", [10, 10]),
    }

    # 旧 metadata（无 checksum）
    old_metadata = {"num_layers": "32"}

    # 验证应该通过（向后兼容）
    assert verify_checksum_from_metadata(old_metadata, tensors_raw) is True
```

---

## 📊 预期效果

### 功能验证

| 场景 | 预期行为 |
|------|----------|
| **正常保存/加载** | ✅ Checksum 自动计算和验证 |
| **数据损坏** | ❌ 验证失败，自动删除损坏文件 |
| **向后兼容** | ✅ 旧缓存文件（无 checksum）正常加载 |
| **性能影响** | < 1% (XXH64 ~10 GB/s) |

### 统计信息

```python
stats = manager.get_stats()
print(stats)

# 输出:
# {
#     "saves": 100,
#     "loads": 150,
#     "hits": 120,
#     "checksum_verifications": 120,  # ✅ 新增
#     "checksum_failures": 2,         # ✅ 新增
#     ...
# }
```

---

## 🎯 成功标准

### 功能标准

- [ ] Checksum 正确计算
- [ ] 验证逻辑正确
- [ ] 损坏文件自动删除
- [ ] 向后兼容旧缓存

### 性能标准

- [ ] Checksum 计算开销 < 1%
- [ ] 不影响主流程性能

### 质量标准

- [ ] 测试覆盖率 > 80%
- [ ] 所有测试通过
- [ ] 代码审查通过

---

**设计版本**: v1.0
**创建日期**: 2026-03-13
**预计工期**: 0.5 天
