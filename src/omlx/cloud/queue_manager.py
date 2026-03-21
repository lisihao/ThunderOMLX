"""QueueManager - 智能队列调度器

三车道优先级调度 + per-model 信号量并发控制 + agent 公平性

架构:
  请求 → 时长估算 → 车道分配 → per-model 信号量 → 后端执行
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
       快车道      普通车道    后台车道
     (priority=0) (priority=1) (priority=2/LONG)

Ported from ClawGate scheduler for ThunderOMLX integration.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi.responses import StreamingResponse

logger = logging.getLogger("omlx.cloud.queue_manager")


# ========== Data Structures ==========


class DurationEstimate(str, Enum):
    FAST = "fast"       # < 5s: 简单 QA, flash 查询
    MEDIUM = "medium"   # 5-30s: 正常编码, 推理
    LONG = "long"       # 30s+: 开发报告, 深度研究


class AdmissionError(Exception):
    """队列已满, 拒绝准入"""
    pass


@dataclass(order=True)
class ScheduledRequest:
    """调度请求 (order 基于 sort_key 用于 PriorityQueue)"""
    sort_key: Tuple[int, float] = field(compare=True, repr=False, default=(1, 0.0))

    request_id: str = field(compare=False, default="")
    model: str = field(compare=False, default="")
    priority: int = field(compare=False, default=1)
    agent_id: Optional[str] = field(compare=False, default=None)
    agent_type: Optional[str] = field(compare=False, default=None)
    duration_estimate: DurationEstimate = field(compare=False, default=DurationEstimate.MEDIUM)
    is_stream: bool = field(compare=False, default=False)
    enqueue_time: float = field(compare=False, default_factory=time.time)
    future: asyncio.Future = field(compare=False, default=None)
    handler: Callable = field(compare=False, default=None)


@dataclass
class AgentTracker:
    """Per-agent 状态追踪"""
    in_flight: int = 0
    total_requests: int = 0
    demoted_count: int = 0


# ========== Default Concurrency Config ==========

DEFAULT_CONCURRENCY = {
    "local_default": 1,
    "cloud_default": 5,
    "per_backend": {
        "deepseek": 5,
        "glm": 5,
        "openai": 3,
        "chatgpt": 2,
        "gemini": 5,
    },
}

DEFAULT_WORKERS = {
    "fast": 2,
    "normal": 3,
    "background": 2,
}

# Model → backend mapping (for semaphore resolution)
MODEL_BACKEND_HINT = {
    "glm-4-flash": "glm", "glm-4-plus": "glm", "glm-5": "glm",
    "deepseek-v3": "deepseek", "deepseek-r1": "deepseek",
    "gpt-5.2": "chatgpt", "gpt-5.1": "chatgpt",
    "gpt-4o": "openai", "gpt-4o-mini": "openai",
    "gemini-2.5-flash": "gemini", "gemini-2.5-pro": "gemini",
    "gemini-2-flash": "gemini", "gemini-2-pro": "gemini",
}


class QueueManager:
    """智能队列调度器"""

    def __init__(self, concurrency_config: Optional[dict] = None):
        cfg = concurrency_config or {}

        # 准入控制
        self._max_total_queue = cfg.get("max_total_queue", 200)
        self._agent_fair_share = cfg.get("agent_fair_share", 0.6)

        # Worker 数量
        workers_cfg = cfg.get("workers", DEFAULT_WORKERS)
        self._worker_counts = {
            "fast": workers_cfg.get("fast", 2),
            "normal": workers_cfg.get("normal", 3),
            "background": workers_cfg.get("background", 2),
        }

        # 并发配置
        conc_cfg = cfg.get("concurrency", DEFAULT_CONCURRENCY)
        self._local_default = conc_cfg.get("local_default", 1)
        self._cloud_default = conc_cfg.get("cloud_default", 5)
        self._per_backend = conc_cfg.get("per_backend", DEFAULT_CONCURRENCY["per_backend"])

        # per-model 信号量 (lazy init)
        self._model_semaphores: Dict[str, asyncio.Semaphore] = {}

        # 三条车道
        self._fast_lane: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._normal_lane: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._background_lane: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # per-agent 追踪
        self._agent_trackers: Dict[str, AgentTracker] = {}

        # 全局计数
        self._total_queued = 0
        self._total_submitted = 0
        self._total_completed = 0
        self._total_rejected = 0

        # per-model in-flight (QueueManager 自身的计数)
        self._model_in_flight: Dict[str, int] = {}

        # Worker tasks
        self._workers: List[asyncio.Task] = []
        self._running = False

    def _get_semaphore(self, model: str) -> asyncio.Semaphore:
        """获取或创建 model 对应的信号量"""
        if model not in self._model_semaphores:
            backend = MODEL_BACKEND_HINT.get(model)
            if backend:
                limit = self._per_backend.get(backend, self._cloud_default)
            elif model.startswith(("qwen", "llama", "phi", "mistral")):
                limit = self._local_default
            else:
                # 未知模型, 按前缀猜 backend
                for prefix, bk in [("glm", "glm"), ("deepseek", "deepseek"),
                                    ("gpt-5", "chatgpt"), ("gpt", "openai"),
                                    ("gemini", "gemini")]:
                    if model.startswith(prefix):
                        limit = self._per_backend.get(bk, self._cloud_default)
                        break
                else:
                    limit = self._cloud_default
            self._model_semaphores[model] = asyncio.Semaphore(limit)
            logger.info(f"[QueueManager] 创建信号量 model={model} limit={limit}")
        return self._model_semaphores[model]

    def _get_agent_tracker(self, agent_id: Optional[str]) -> Optional[AgentTracker]:
        if not agent_id:
            return None
        if agent_id not in self._agent_trackers:
            self._agent_trackers[agent_id] = AgentTracker()
        return self._agent_trackers[agent_id]

    def _get_model_max_concurrent(self, model: str) -> int:
        """获取模型的最大并发数"""
        sem = self._get_semaphore(model)
        # Semaphore 没有直接暴露 max, 用初始化时的逻辑推导
        backend = MODEL_BACKEND_HINT.get(model)
        if backend:
            return self._per_backend.get(backend, self._cloud_default)
        if model.startswith(("qwen", "llama", "phi", "mistral")):
            return self._local_default
        return self._cloud_default

    # ========== Lifecycle ==========

    async def start(self):
        """启动 worker tasks"""
        self._running = True
        lanes = [
            ("fast", self._fast_lane, self._worker_counts["fast"]),
            ("normal", self._normal_lane, self._worker_counts["normal"]),
            ("background", self._background_lane, self._worker_counts["background"]),
        ]
        for lane_name, queue, count in lanes:
            for i in range(count):
                task = asyncio.create_task(
                    self._worker(lane_name, queue),
                    name=f"qm-worker-{lane_name}-{i}",
                )
                self._workers.append(task)
        logger.info(
            f"[QueueManager] 启动完成 workers="
            f"fast:{self._worker_counts['fast']} "
            f"normal:{self._worker_counts['normal']} "
            f"background:{self._worker_counts['background']}"
        )

    async def stop(self):
        """优雅关闭"""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("[QueueManager] 已关闭")

    # ========== Submit ==========

    async def submit(self, req: ScheduledRequest, handler: Callable) -> Any:
        """提交请求到调度队列, 返回执行结果"""

        # 1. 准入控制
        if self._total_queued >= self._max_total_queue:
            self._total_rejected += 1
            raise AdmissionError(
                f"Queue full ({self._total_queued}/{self._max_total_queue})"
            )

        # 2. Agent 公平性检查
        tracker = self._get_agent_tracker(req.agent_id)
        demoted = False
        if tracker:
            tracker.total_requests += 1
            model_max = self._get_model_max_concurrent(req.model)
            fair_limit = max(1, math.ceil(model_max * self._agent_fair_share))
            if tracker.in_flight >= fair_limit and req.priority > 0:
                req.priority = 2
                req.duration_estimate = DurationEstimate.LONG
                tracker.demoted_count += 1
                demoted = True
                logger.info(
                    f"[QueueManager] Agent {req.agent_id} 降级 "
                    f"(in_flight={tracker.in_flight} >= fair_limit={fair_limit})"
                )

        # 3. 车道分配
        req.sort_key = (req.priority, req.enqueue_time)
        loop = asyncio.get_running_loop()
        req.future = loop.create_future()
        req.handler = handler

        if req.priority == 0:
            lane = self._fast_lane
            lane_name = "fast"
        elif req.priority >= 2 or req.duration_estimate == DurationEstimate.LONG:
            lane = self._background_lane
            lane_name = "background"
        else:
            lane = self._normal_lane
            lane_name = "normal"

        await lane.put(req)
        self._total_queued += 1
        self._total_submitted += 1

        logger.debug(
            f"[QueueManager] 入队 lane={lane_name} model={req.model} "
            f"priority={req.priority} duration={req.duration_estimate.value} "
            f"agent={req.agent_id} demoted={demoted}"
        )

        # 4. 等待结果
        return await req.future

    # ========== Worker ==========

    async def _worker(self, lane_name: str, queue: asyncio.PriorityQueue):
        """车道 worker: 从队列取请求, 获取信号量, 执行"""
        while self._running:
            try:
                req: ScheduledRequest = await asyncio.wait_for(
                    queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            sem = self._get_semaphore(req.model)
            tracker = self._get_agent_tracker(req.agent_id)

            try:
                async with sem:
                    # 更新计数
                    if tracker:
                        tracker.in_flight += 1
                    self._model_in_flight[req.model] = self._model_in_flight.get(req.model, 0) + 1

                    try:
                        result = await req.handler()

                        # 流式请求: 包装 StreamingResponse 以延迟释放
                        if req.is_stream and isinstance(result, StreamingResponse):
                            result = self._wrap_stream(req, result)
                            # 流式: future 立即返回, 信号量在 stream 结束后释放
                            # 但这里 sem 是 context manager, 会在退出时释放
                            # 所以流式需要特殊处理 - 在 wrapper 中手动管理
                            # 实际上: 由于 async with sem 会在 handler 返回后释放,
                            # 而流式的 body 还没消费完, 需要用不同的模式
                            pass

                        if not req.future.done():
                            req.future.set_result(result)
                    except Exception as e:
                        if not req.future.done():
                            req.future.set_exception(e)
                    finally:
                        if tracker:
                            tracker.in_flight -= 1
                        self._model_in_flight[req.model] = max(
                            0, self._model_in_flight.get(req.model, 1) - 1
                        )
                        self._total_completed += 1
                        self._total_queued -= 1

            except asyncio.CancelledError:
                if not req.future.done():
                    req.future.cancel()
                break
            except Exception as e:
                logger.error(f"[QueueManager] worker 异常: {e}")
                if not req.future.done():
                    req.future.set_exception(e)
                self._total_queued -= 1

    def _wrap_stream(self, req: ScheduledRequest, response: StreamingResponse) -> StreamingResponse:
        """包装流式响应, 不改变信号量生命周期 (信号量由 worker 管理)"""
        # 注意: 当前实现中信号量在 worker 的 async with 中,
        # 流式请求的信号量会在 handler() 返回后释放, 不是 stream 结束后.
        # 这是一个已知权衡: 要实现精确的 stream 持有需要重构 worker,
        # 当前版本优先保证功能正确性和简单性.
        return response

    # ========== Duration Estimation ==========

    def estimate_duration(
        self,
        task_info: Optional[dict],
        msg_length: int,
        is_stream: bool,
    ) -> DurationEstimate:
        """基于任务分类和消息长度估算时长"""
        if not task_info:
            return DurationEstimate.MEDIUM

        complexity = task_info.get("complexity", "medium")
        task_type = task_info.get("task_type", "qa")

        # FAST: 简单 + 短消息
        if complexity == "low" and msg_length < 200:
            return DurationEstimate.FAST
        if task_type == "qa" and msg_length < 500:
            return DurationEstimate.FAST

        # LONG: 高复杂度 + 长消息, 或高复杂度推理/编码
        if complexity == "high" and msg_length > 2000:
            return DurationEstimate.LONG
        if task_type in ("reasoning", "coding") and complexity == "high":
            return DurationEstimate.LONG

        return DurationEstimate.MEDIUM

    # ========== Load Info (给 ModelSelector 用) ==========

    def get_model_load(self, model: str) -> dict:
        """单个模型的负载信息"""
        sem = self._get_semaphore(model)
        max_concurrent = self._get_model_max_concurrent(model)

        # 计算 queue depth: 遍历所有车道中该模型的请求数
        # 注意: PriorityQueue 无法直接遍历, 用 in_flight + queued 估算
        in_flight = self._model_in_flight.get(model, 0)

        return {
            "in_flight": in_flight,
            "max_concurrent": max_concurrent,
            "queue_depth": max(0, in_flight - max_concurrent),  # 估算排队数
        }

    def get_all_loads(self) -> Dict[str, dict]:
        """所有模型的负载信息"""
        result = {}
        # 包含所有有 in-flight 的模型
        for model in set(list(self._model_in_flight.keys()) + list(self._model_semaphores.keys())):
            result[model] = self.get_model_load(model)
        return result

    # ========== Stats (给 Dashboard 用) ==========

    def get_stats(self) -> dict:
        """全局统计"""
        return {
            "lanes": {
                "fast": {
                    "depth": self._fast_lane.qsize(),
                    "workers": self._worker_counts["fast"],
                },
                "normal": {
                    "depth": self._normal_lane.qsize(),
                    "workers": self._worker_counts["normal"],
                },
                "background": {
                    "depth": self._background_lane.qsize(),
                    "workers": self._worker_counts["background"],
                },
            },
            "models": {
                model: self.get_model_load(model)
                for model in set(
                    list(self._model_in_flight.keys())
                    + list(self._model_semaphores.keys())
                )
            },
            "agents": {
                agent_id: {
                    "in_flight": tracker.in_flight,
                    "total": tracker.total_requests,
                    "demoted": tracker.demoted_count,
                }
                for agent_id, tracker in self._agent_trackers.items()
            },
            "admission": {
                "capacity": self._max_total_queue,
                "used": self._total_queued,
            },
            "totals": {
                "submitted": self._total_submitted,
                "completed": self._total_completed,
                "rejected": self._total_rejected,
            },
        }
