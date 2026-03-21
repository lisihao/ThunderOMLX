"""Memory Monitor - 内存监控器

监控系统内存使用，自动驱逐小模型防止 OOM：
- 内存超限 → 按 LRU 卸载小模型
- 紧急驱逐 → 卸载所有 On-Demand 模型

来源：ClawGate 多模型架构设计（2026-03-11）
"""

import asyncio
import logging
from typing import Dict, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from omlx.cloud.lifecycle_manager import ModelLifecycleManager

logger = logging.getLogger("omlx.cloud.memory_monitor")


class MemoryMonitor:
    """内存监控器"""

    def __init__(
        self,
        lifecycle_manager: ModelLifecycleManager,
        threshold_gb: float = 42.0,
        check_interval_sec: int = 60,
        enabled: bool = True,
    ):
        """
        初始化内存监控器

        Args:
            lifecycle_manager: 模型生命周期管理器
            threshold_gb: 内存阈值（GB），超过后触发驱逐
            check_interval_sec: 检查间隔（秒）
            enabled: 是否启用
        """
        self.manager = lifecycle_manager
        self.threshold_bytes = int(threshold_gb * 1024**3)
        self.check_interval = check_interval_sec
        self.enabled = enabled and PSUTIL_AVAILABLE

        self._monitor_task: Optional[asyncio.Task] = None

        if self.enabled:
            logger.info(
                f"MemoryMonitor: 启用内存监控 (阈值 {threshold_gb}GB, "
                f"间隔 {check_interval_sec}s)"
            )
        elif not PSUTIL_AVAILABLE:
            logger.warning("MemoryMonitor: psutil 未安装，内存监控禁用")
        else:
            logger.info("MemoryMonitor: 内存监控禁用")

    async def start(self):
        """启动监控循环"""
        if not self.enabled:
            return

        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """停止监控循环"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """监控循环"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MemoryMonitor: 监控循环错误: {e}")

    async def _check_memory(self):
        """检查内存使用"""
        if not PSUTIL_AVAILABLE:
            return

        mem = psutil.virtual_memory()

        # 记录当前内存使用
        logger.debug(
            f"MemoryMonitor: 内存使用 {mem.used / 1024**3:.1f}GB / "
            f"{mem.total / 1024**3:.1f}GB ({mem.percent:.1f}%)"
        )

        # 超过阈值 → 触发驱逐
        if mem.used > self.threshold_bytes:
            logger.warning(
                f"MemoryMonitor: 内存超限! "
                f"{mem.used / 1024**3:.1f}GB > {self.threshold_bytes / 1024**3:.1f}GB"
            )
            await self._evict_models()

    async def _evict_models(self):
        """驱逐模型（LRU 策略）"""
        # 获取所有 On-Demand 模型，按空闲时间排序（LRU）
        on_demand_models = [
            (name, instance)
            for name, instance in self.manager.instances.items()
            if instance.config.mode == "on_demand"
        ]

        if not on_demand_models:
            logger.warning("MemoryMonitor: 无可驱逐模型（所有模型都是 Always-On）")
            return

        # 按空闲时间排序（最久未使用的优先驱逐）
        on_demand_models.sort(key=lambda x: x[1].idle_time(), reverse=True)

        # 驱逐第一个（最久未使用的）
        name, instance = on_demand_models[0]
        logger.info(
            f"MemoryMonitor: 驱逐模型 {name} "
            f"(空闲 {instance.idle_time():.0f}s, mode {instance.config.mode})"
        )
        await self.manager.unload_model(name)

        # 再次检查内存
        mem = psutil.virtual_memory()
        if mem.used > self.threshold_bytes:
            logger.warning(
                f"MemoryMonitor: 内存仍超限 "
                f"({mem.used / 1024**3:.1f}GB > {self.threshold_bytes / 1024**3:.1f}GB)，"
                "继续驱逐..."
            )
            # 递归驱逐（最多驱逐所有 On-Demand 模型）
            if len(on_demand_models) > 1:
                await self._evict_models()

    async def emergency_evict_all(self):
        """紧急驱逐所有 On-Demand 模型"""
        logger.warning("MemoryMonitor: 紧急驱逐所有 On-Demand 模型")

        on_demand_models = [
            name
            for name, instance in self.manager.instances.items()
            if instance.config.mode == "on_demand"
        ]

        for name in on_demand_models:
            await self.manager.unload_model(name)

        logger.info(f"MemoryMonitor: 已驱逐 {len(on_demand_models)} 个模型")

    def get_memory_stats(self) -> Dict:
        """获取内存统计信息

        Returns:
            内存统计字典
        """
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil 未安装"}

        mem = psutil.virtual_memory()

        return {
            "total_gb": round(mem.total / 1024**3, 2),
            "used_gb": round(mem.used / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "percent": round(mem.percent, 1),
            "threshold_gb": round(self.threshold_bytes / 1024**3, 2),
            "over_threshold": mem.used > self.threshold_bytes,
        }
