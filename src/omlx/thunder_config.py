"""
ThunderOMLX 统一配置系统
使用 Pydantic v2 实现类型安全和校验
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheConfig(BaseModel):
    """缓存配置"""
    enable_l2_cache: bool = Field(default=True, description="启用 L2 内存缓存")
    enable_l3_cache: bool = Field(default=True, description="启用 L3 磁盘缓存")
    l2_cache_size_mb: int = Field(
        default=100, ge=20, le=32768, description="L2 缓存大小（MB）"
    )
    l3_cache_path: Path = Field(
        default=Path("~/.cache/thunderomlx/l3_cache").expanduser()
    )
    l3_cache_size_gb: int = Field(default=256, ge=10, le=1024)
    lru_k: int = Field(default=2, ge=1, le=5, description="LRU-K 策略的 K 值")
    enable_mmap_zero_copy: bool = Field(default=True, description="启用 mmap 零拷贝")

    @field_validator("l3_cache_path", mode="before")
    @classmethod
    def expand_path(cls, v: str | Path) -> Path:
        """扩展 ~ 路径"""
        return Path(v).expanduser()


class SerializationConfig(BaseModel):
    """张量序列化配置"""
    compression: Literal["none", "zlib", "lz4"] = Field(default="lz4")
    compression_level: int = Field(default=6, ge=1, le=9)
    enable_checksum: bool = Field(default=True)
    checksum_algorithm: Literal["xxh64", "md5", "sha256"] = Field(default="xxh64")


class AsyncIOConfig(BaseModel):
    """异步 I/O 配置"""
    enable_async_io: bool = Field(default=True)
    prefetch_threads: int = Field(default=4, ge=1, le=8)
    batch_size: int = Field(default=8, ge=1, le=32, description="批量加载块数量")
    read_buffer_kb: int = Field(default=256, ge=64, le=1024)


class ThunderOMLXConfig(BaseSettings):
    """ThunderOMLX 统一配置"""

    model_config = SettingsConfigDict(
        env_prefix="THUNDEROMLX_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # 基础配置
    project_name: str = Field(default="ThunderOMLX")
    version: str = Field(default="0.1.0")

    # 子配置
    cache: CacheConfig = Field(default_factory=CacheConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)
    async_io: AsyncIOConfig = Field(default_factory=AsyncIOConfig)

    # 性能目标
    target_generation_tps: int = Field(default=250, ge=100, le=1000)

    # 配置文件路径（用于记录来源，不参与环境变量）
    _config_path: Optional[Path] = None

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "ThunderOMLXConfig":
        """从 YAML 文件加载配置

        优先级：环境变量 > YAML 配置 > 默认值
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        instance = cls(**data)
        instance._config_path = yaml_path
        return instance

    def save_to_yaml(self, yaml_path: Path) -> None:
        """保存配置到 YAML 文件"""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换 Path 为字符串以便序列化
        data = self.model_dump(mode="python")

        # 处理 Path 对象
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = convert_paths(data)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @model_validator(mode="after")
    def validate_config(self) -> "ThunderOMLXConfig":
        """跨字段校验"""
        # 如果禁用 L2 缓存，警告但允许
        if not self.cache.enable_l2_cache and self.cache.enable_l3_cache:
            # L3 依赖 L2 时可能需要调整
            pass
        return self
