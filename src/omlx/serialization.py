"""
ThunderOMLX 张量序列化模块
基于 MLX 原生 API + Checksum 验证
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import mlx.core as mx
import numpy as np

from omlx.thunder_config import SerializationConfig


@dataclass
class TensorMetadata:
    """张量元数据"""
    shape: tuple[int, ...]
    dtype: str
    checksum: str
    compression: Literal["none", "zlib", "lz4"]
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "checksum": self.checksum,
            "compression": self.compression,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorMetadata":
        return cls(
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            checksum=data["checksum"],
            compression=data["compression"],
            version=data.get("version", "1.0"),
        )


class TensorSerializer:
    """张量序列化器

    文件格式：
    - <filename>.meta.json - 元数据
    - <filename>.data - 二进制数据（MLX .npy 或 .npz 格式）
    """

    def __init__(self, config: SerializationConfig):
        self.config = config

    def _compute_checksum(self, data: bytes) -> str:
        """计算数据校验和"""
        if self.config.checksum_algorithm == "xxh64":
            try:
                import xxhash
                return xxhash.xxh64(data).hexdigest()
            except ImportError:
                # Fallback to hashlib if xxhash not available
                import hashlib
                return hashlib.sha256(data).hexdigest()[:16]  # 截断为类似长度
        elif self.config.checksum_algorithm == "md5":
            import hashlib
            return hashlib.md5(data).hexdigest()
        elif self.config.checksum_algorithm == "sha256":
            import hashlib
            return hashlib.sha256(data).hexdigest()
        else:
            raise ValueError(f"不支持的校验算法: {self.config.checksum_algorithm}")

    def save(self, tensor: mx.array, file_path: Path) -> TensorMetadata:
        """保存张量到文件

        Args:
            tensor: MLX 数组
            file_path: 文件路径（不含扩展名）

        Returns:
            张量元数据
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = file_path.with_suffix(".meta.json")

        # 1. 保存张量数据
        # 先 evaluate tensor 以确保数据已计算
        mx.eval(tensor)

        if self.config.compression == "zlib":
            # 使用 MLX 的压缩保存（自动添加 .npz 后缀）
            base_path = str(file_path)
            mx.savez_compressed(base_path, tensor=tensor)
            data_path = Path(base_path + ".npz")
        elif self.config.compression == "lz4":
            # 使用 lz4 压缩（自定义格式）
            import lz4.frame

            # 转换为 numpy array
            np_array = np.array(tensor)

            # 序列化为 bytes
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            uncompressed_bytes = buffer.getvalue()

            # lz4 压缩
            compressed_bytes = lz4.frame.compress(
                uncompressed_bytes,
                compression_level=self.config.compression_level
            )

            # 保存压缩数据
            data_path = file_path.with_suffix(".lz4")
            with open(data_path, "wb") as f:
                f.write(compressed_bytes)
        elif self.config.compression == "none":
            # 使用 MLX 的原生保存（自动添加 .npy 后缀）
            base_path = str(file_path)
            mx.save(base_path, tensor)
            data_path = Path(base_path + ".npy")
        else:
            raise ValueError(f"不支持的压缩方式: {self.config.compression}")

        # 2. 计算 Checksum
        if self.config.enable_checksum:
            with open(data_path, "rb") as f:
                data_bytes = f.read()
            checksum = self._compute_checksum(data_bytes)
        else:
            checksum = ""

        # 3. 保存元数据
        metadata = TensorMetadata(
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            checksum=checksum,
            compression=self.config.compression,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return metadata

    def load(self, file_path: Path, verify_checksum: bool = True) -> mx.array:
        """从文件加载张量

        Args:
            file_path: 文件路径（不含扩展名）
            verify_checksum: 是否验证校验和

        Returns:
            MLX 数组

        Raises:
            ValueError: 校验和不匹配
            FileNotFoundError: 文件不存在
        """
        file_path = Path(file_path)
        meta_path = file_path.with_suffix(".meta.json")

        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

        # 1. 加载元数据
        with open(meta_path, encoding="utf-8") as f:
            metadata = TensorMetadata.from_dict(json.load(f))

        # 2. 确定数据文件路径
        if metadata.compression == "zlib":
            data_path = Path(str(file_path) + ".npz")
        elif metadata.compression == "lz4":
            data_path = Path(str(file_path) + ".lz4")
        else:
            data_path = Path(str(file_path) + ".npy")

        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        # 3. 验证 Checksum
        if verify_checksum and metadata.checksum and self.config.enable_checksum:
            with open(data_path, "rb") as f:
                data_bytes = f.read()

            actual_checksum = self._compute_checksum(data_bytes)
            if actual_checksum != metadata.checksum:
                raise ValueError(
                    f"校验和不匹配: 期望 {metadata.checksum}，实际 {actual_checksum}"
                )

        # 4. 加载张量数据
        if metadata.compression == "lz4":
            # lz4 解压
            import lz4.frame

            with open(data_path, "rb") as f:
                compressed_bytes = f.read()

            # 解压
            uncompressed_bytes = lz4.frame.decompress(compressed_bytes)

            # 反序列化 numpy array
            buffer = io.BytesIO(uncompressed_bytes)
            np_array = np.load(buffer)

            # 转换为 MLX array
            tensor = mx.array(np_array)
        elif metadata.compression == "zlib":
            # MLX savez_compressed 保存为 .npz 格式
            loaded = mx.load(str(data_path))
            if isinstance(loaded, dict):
                tensor = loaded["tensor"]  # 我们保存时使用的键
            else:
                tensor = loaded
        else:
            tensor = mx.load(str(data_path))

        return tensor

    def get_metadata(self, file_path: Path) -> Optional[TensorMetadata]:
        """获取张量元数据（不加载数据）

        Args:
            file_path: 文件路径（不含扩展名）

        Returns:
            元数据，如果文件不存在则返回 None
        """
        meta_path = Path(file_path).with_suffix(".meta.json")
        if not meta_path.exists():
            return None

        with open(meta_path, encoding="utf-8") as f:
            return TensorMetadata.from_dict(json.load(f))
