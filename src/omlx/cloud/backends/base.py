"""推理引擎基类"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GenerationRequest:
    """生成请求"""

    messages: List[Dict]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False

    # KB-019: Parameters previously dropped during cloud routing
    tools: Optional[list] = None
    tool_choice: Optional[Any] = None  # "auto", "none", "required", or dict
    response_format: Optional[dict] = None
    stop: Optional[Any] = None  # str or list[str]
    stream_options: Optional[dict] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    min_p: Optional[float] = None
    n: Optional[int] = None
    chat_template_kwargs: Optional[dict] = None

    # OpenClaw 扩展字段
    priority: int = 1
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    request_id: Optional[str] = None

    # 内部字段
    arrival_time: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class GenerationResponse:
    """生成响应"""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_time: float
    ttft: Optional[float] = None  # Time to First Token
    metadata: Optional[Dict] = None


class BaseEngine(ABC):
    """推理引擎基类"""

    def __init__(self, engine_type: str, model_path: str, **kwargs):
        self.engine_type = engine_type
        self.model_path = model_path
        self.config = kwargs

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成响应（非流式）"""
        pass

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """流式生成"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """获取引擎统计信息"""
        pass

    def supports_feature(self, feature: str) -> bool:
        """检查是否支持某功能"""
        features = {
            "mlx": ["stream", "batch"],
            "llamacpp": ["stream", "batch", "quantization"],
            "vllm": ["stream", "batch", "continuous_batching"],
            "sglang": [
                "stream",
                "batch",
                "continuous_batching",
                "radix_attention",
            ],
        }
        return feature in features.get(self.engine_type, [])

    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.engine_type}, model={self.model_path})"
