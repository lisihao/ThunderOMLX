"""
ThunderOMLX 配置加载器
单例模式 + 热重载支持
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from omlx.thunder_config import ThunderOMLXConfig

logger = logging.getLogger(__name__)

# 全局配置实例
_config_instance: Optional[ThunderOMLXConfig] = None
_config_path: Optional[Path] = None


def get_default_config_path() -> Path:
    """获取默认配置文件路径

    查找顺序:
    1. 环境变量 THUNDEROMLX_CONFIG_PATH
    2. 当前工作目录/thunderomlx.yaml
    3. 项目根目录/thunderomlx.yaml（向上查找）
    """
    # 1. 环境变量
    env_path = os.environ.get("THUNDEROMLX_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    # 2. 当前目录
    cwd_path = Path.cwd() / "thunderomlx.yaml"
    if cwd_path.exists():
        return cwd_path

    # 3. 向上查找项目根目录
    current = Path.cwd()
    for _ in range(5):  # 最多向上 5 层
        candidate = current / "thunderomlx.yaml"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent

    # 默认返回当前目录（即使不存在，后续会创建）
    return cwd_path


def load_thunder_config(
    config_path: Optional[Path] = None,
    force_reload: bool = False,
) -> ThunderOMLXConfig:
    """加载 ThunderOMLX 配置（单例模式）

    Args:
        config_path: 配置文件路径，为 None 时自动查找
        force_reload: 强制重新加载（忽略缓存）

    Returns:
        ThunderOMLXConfig 实例
    """
    global _config_instance, _config_path

    if _config_instance is not None and not force_reload:
        return _config_instance

    # 确定配置路径
    if config_path is None:
        config_path = get_default_config_path()

    _config_path = config_path

    if config_path.exists():
        logger.info(f"加载配置文件: {config_path}")
        _config_instance = ThunderOMLXConfig.load_from_yaml(config_path)
    else:
        logger.warning(f"配置文件不存在，使用默认配置: {config_path}")
        _config_instance = ThunderOMLXConfig()
        # 创建默认配置文件
        _config_instance.save_to_yaml(config_path)
        logger.info(f"已创建默认配置文件: {config_path}")

    return _config_instance


def reload_thunder_config(config_path: Optional[Path] = None) -> ThunderOMLXConfig:
    """重新加载配置（热重载）

    Args:
        config_path: 新的配置文件路径（可选）

    Returns:
        重新加载的 ThunderOMLXConfig 实例
    """
    global _config_instance

    _config_instance = None
    logger.info("重新加载配置...")

    return load_thunder_config(config_path, force_reload=True)


def get_config() -> ThunderOMLXConfig:
    """获取当前配置实例（不触发加载）

    如果配置未加载，会触发加载
    """
    if _config_instance is None:
        return load_thunder_config()
    return _config_instance


def get_config_path() -> Optional[Path]:
    """获取当前使用的配置文件路径"""
    return _config_path
