#!/usr/bin/env python3
"""
P1-6: Checksum Validation 功能测试
"""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
