"""Smart Model Router - 智能模型路由器

根据任务类型自动选择合适的模型：
- 主推理任务 → 30B（Always-On）
- Context Shift Stage 2 → 1.7B（On-Demand）
- Context Shift Stage 1 → 0.6B（On-Demand）

来源：ClawGate 多模型架构设计（2026-03-11）
"""

import logging
from typing import Dict, Optional

from omlx.cloud.lifecycle_manager import ModelLifecycleManager, ModelInstance

logger = logging.getLogger("omlx.cloud.smart_router")


class SmartModelRouter:
    """智能模型路由器"""

    def __init__(
        self,
        lifecycle_manager: ModelLifecycleManager,
        routing_config: Optional[Dict] = None,
    ):
        """
        初始化路由器

        Args:
            lifecycle_manager: 模型生命周期管理器
            routing_config: 路由配置（任务类型 → 模型名称映射）
        """
        self.manager = lifecycle_manager
        self.routing_config = routing_config or {
            "main_inference": "qwen-30b",
            "context_shift_stage1": "qwen-0.6b",
            "context_shift_stage2": "qwen-1.7b",
        }

    async def route(
        self, task_type: str, fallback: Optional[str] = None
    ) -> ModelInstance:
        """
        根据任务类型路由到合适的模型

        Args:
            task_type: 任务类型（如 "main_inference", "context_shift_stage1"）
            fallback: 降级模型（如果主模型不可用）

        Returns:
            ModelInstance

        Raises:
            ValueError: 未知任务类型且无 fallback
        """
        # 1. 查找任务类型对应的模型
        model_name = self.routing_config.get(task_type)

        if not model_name:
            # 未知任务类型 → 使用 fallback 或主模型
            if fallback:
                model_name = fallback
            else:
                model_name = self.routing_config.get("main_inference")
                logger.warning(
                    f"未知任务类型: {task_type}，降级到主模型: {model_name}"
                )

        # 2. 获取模型实例（自动加载）
        instance = await self.manager.get_model(model_name)

        logger.info(
            f"路由: {task_type} → {model_name} "
            f"(port {instance.config.port}, mode {instance.config.mode})"
        )

        return instance

    async def route_main_inference(self) -> ModelInstance:
        """路由到主推理模型（30B Always-On）"""
        return await self.route("main_inference")

    async def route_context_shift_stage1(self) -> ModelInstance:
        """路由到 Context Shift Stage 1（0.6B On-Demand）"""
        return await self.route("context_shift_stage1")

    async def route_context_shift_stage2(self) -> ModelInstance:
        """路由到 Context Shift Stage 2（1.7B On-Demand）"""
        return await self.route("context_shift_stage2")

    def get_routing_table(self) -> Dict:
        """获取路由表

        Returns:
            路由表字典
        """
        return {
            "routing_config": self.routing_config,
            "loaded_models": self.manager.get_stats()["loaded_models"],
        }
