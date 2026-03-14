# SPDX-License-Identifier: Apache-2.0
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
