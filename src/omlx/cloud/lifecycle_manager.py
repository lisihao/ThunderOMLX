"""Model Lifecycle Manager - 模型生命周期管理器

混合模式设计：
- Always-On: 主模型常驻，避免冷启动
- On-Demand + TTL: 辅助模型按需加载，空闲自动卸载

来源：ClawGate 多模型架构设计（2026-03-11）
"""

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger("omlx.cloud.lifecycle_manager")


@dataclass
class ModelConfig:
    """模型配置"""

    name: str  # 模型名称（如 "qwen-30b"）
    model_path: str  # 模型文件路径
    port: int  # llama-server 端口
    mode: str  # "always_on" | "on_demand"
    n_ctx: int = 8192  # 上下文长度
    n_gpu_layers: int = 99  # GPU 层数
    ttl_seconds: int = 0  # On-Demand 模式的 TTL（秒）
    startup_timeout: int = 60  # 启动超时（秒）


class ModelInstance:
    """模型实例"""

    def __init__(self, config: ModelConfig, process: subprocess.Popen):
        self.config = config
        self.process = process
        self.last_access = time.time()
        self._cleanup_task: Optional[asyncio.Task] = None

    def update_access(self):
        """更新访问时间（重置 TTL）"""
        self.last_access = time.time()

    def idle_time(self) -> float:
        """获取空闲时间（秒）"""
        return time.time() - self.last_access

    async def start_ttl_timer(self, manager: "ModelLifecycleManager"):
        """启动 TTL 定时器（On-Demand 模式）"""
        if self.config.mode == "always_on":
            return  # Always-On 模式不启动定时器

        self._cleanup_task = asyncio.create_task(self._ttl_loop(manager))

    async def _ttl_loop(self, manager: "ModelLifecycleManager"):
        """TTL 循环检查"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查

            idle = self.idle_time()
            if idle > self.config.ttl_seconds:
                logger.info(
                    f"[{self.config.name}] 空闲 {idle:.0f}s，触发卸载 "
                    f"(TTL={self.config.ttl_seconds}s)"
                )
                await manager.unload_model(self.config.name)
                break

    def stop_ttl_timer(self):
        """停止 TTL 定时器"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class ModelLifecycleManager:
    """模型生命周期管理器"""

    def __init__(self, configs: Dict[str, ModelConfig]):
        self.configs = configs
        self.instances: Dict[str, ModelInstance] = {}
        self._lock = asyncio.Lock()

    async def start_always_on_models(self):
        """启动 Always-On 模型"""
        for name, config in self.configs.items():
            if config.mode == "always_on":
                logger.info(f"[{name}] 启动 Always-On 模型")
                await self.get_model(name)

    async def get_model(self, name: str) -> ModelInstance:
        """获取模型实例（自动加载 On-Demand 模型）

        Args:
            name: 模型名称

        Returns:
            ModelInstance

        Raises:
            ValueError: 未知模型
            TimeoutError: 启动超时
        """
        async with self._lock:
            # 1. 模型已加载 → 更新访问时间并返回
            if name in self.instances:
                instance = self.instances[name]
                instance.update_access()
                logger.debug(f"[{name}] 复用已加载模型 (port {instance.config.port})")
                return instance

            # 2. 模型未加载 → 加载
            config = self.configs.get(name)
            if not config:
                raise ValueError(f"未知模型: {name}")

            logger.info(f"[{name}] 开始加载模型...")
            instance = await self._load_model(config)
            self.instances[name] = instance

            # 3. 启动 TTL 定时器（On-Demand 模式）
            await instance.start_ttl_timer(self)

            return instance

    async def _load_model(self, config: ModelConfig) -> ModelInstance:
        """加载模型（启动 llama-server 子进程）

        Args:
            config: 模型配置

        Returns:
            ModelInstance

        Raises:
            FileNotFoundError: 模型文件不存在
            TimeoutError: 启动超时
        """
        # 检查模型文件
        model_path = Path(os.path.expanduser(config.model_path))
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 检查端口是否被占用
        if await self._port_in_use(config.port):
            logger.warning(f"[{config.name}] 端口 {config.port} 已被占用，尝试复用")
            # 尝试健康检查，如果服务正常则复用
            if await self._health_check(config.port):
                logger.info(f"[{config.name}] 复用已有服务 (port {config.port})")
                # 创建一个虚拟进程（不 owns_process）
                return ModelInstance(config, None)  # type: ignore
            else:
                raise RuntimeError(
                    f"端口 {config.port} 被占用但服务不可用，请手动清理"
                )

        # 构建启动命令
        cmd = [
            "llama-server",
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(config.port),
            "-ngl",
            str(config.n_gpu_layers),
            "-c",
            str(config.n_ctx),
        ]

        logger.info(f"[{config.name}] 执行命令: {' '.join(cmd)}")

        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # 创建新进程组
        )

        # 等待服务就绪
        deadline = time.time() + config.startup_timeout
        while time.time() < deadline:
            # 检查进程是否退出
            if process.poll() is not None:
                raise RuntimeError(
                    f"[{config.name}] llama-server 启动后立即退出 "
                    f"(exit code {process.returncode})"
                )

            # 健康检查
            if await self._health_check(config.port):
                logger.info(
                    f"[{config.name}] 模型就绪 "
                    f"(PID {process.pid}, port {config.port})"
                )
                return ModelInstance(config, process)

            await asyncio.sleep(1)

        # 超时 → 清理进程
        self._kill_process(process)
        raise TimeoutError(
            f"[{config.name}] 启动超时 ({config.startup_timeout}s)"
        )

    async def _health_check(self, port: int) -> bool:
        """健康检查

        Args:
            port: 服务端口

        Returns:
            是否健康
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://127.0.0.1:{port}/health", timeout=3.0
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def _port_in_use(self, port: int) -> bool:
        """检查端口是否被占用

        Args:
            port: 端口号

        Returns:
            是否被占用
        """
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    async def unload_model(self, name: str):
        """卸载模型

        Args:
            name: 模型名称
        """
        async with self._lock:
            instance = self.instances.pop(name, None)
            if instance:
                # 停止 TTL 定时器
                instance.stop_ttl_timer()

                # 终止进程
                if instance.process:
                    logger.info(
                        f"[{name}] 卸载模型 (PID {instance.process.pid})"
                    )
                    self._kill_process(instance.process)
                else:
                    logger.info(f"[{name}] 卸载模型 (外部服务)")

    def _kill_process(self, process: subprocess.Popen):
        """终止进程组

        Args:
            process: 进程对象
        """
        try:
            # 发送 SIGTERM 到进程组
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            # 强制 SIGKILL
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def shutdown_all(self):
        """关闭所有模型"""
        logger.info("关闭所有模型...")
        for name in list(self.instances.keys()):
            await self.unload_model(name)

    def get_stats(self) -> Dict:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "loaded_models": [
                {
                    "name": name,
                    "mode": instance.config.mode,
                    "port": instance.config.port,
                    "pid": instance.process.pid if instance.process else None,
                    "idle_time": instance.idle_time(),
                    "ttl": instance.config.ttl_seconds,
                }
                for name, instance in self.instances.items()
            ],
            "total_models": len(self.configs),
            "loaded_count": len(self.instances),
        }
