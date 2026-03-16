#!/usr/bin/env python3
"""
启动 oMLX Admin 控制台
"""
import sys
import uvicorn
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI
from omlx.admin import admin_router, set_admin_getters, set_hf_downloader
from omlx.engine_pool import EnginePool
from omlx.scheduler import SchedulerConfig

# 创建 FastAPI app
app = FastAPI(title="oMLX Admin Panel")

# 初始化 EnginePool
print("📦 初始化 EnginePool...")
engine_pool = EnginePool(
    max_model_memory=40 * 1024**3,  # 40GB
    scheduler_config=SchedulerConfig(
        max_num_seqs=16
    )
)

# 发现模型
print("🔍 发现模型...")
model_dir = Path.home() / "models"
engine_pool.discover_models(str(model_dir))

# 设置 admin getters（用于获取 engine_pool 等）
def get_engine_pool():
    return engine_pool

def get_settings():
    return {}  # 简化版，如需要可以添加

set_admin_getters(
    get_engine_pool=get_engine_pool,
    get_settings=get_settings
)

# 挂载 admin 路由
app.include_router(admin_router, prefix="/admin")

# 添加根路径重定向
@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/admin")

if __name__ == "__main__":
    print()
    print("=" * 80)
    print("🚀 oMLX Admin Panel")
    print("=" * 80)
    print()
    print("📍 访问地址: http://localhost:8080/admin")
    print("🔑 API Key: 在 omlx-api-key.txt 中")
    print()
    print("=" * 80)
    print()

    # 读取 API Key
    api_key_file = Path(__file__).parent / "omlx-api-key.txt"
    if api_key_file.exists():
        api_key = api_key_file.read_text().strip()
        print(f"🔑 你的 API Key: {api_key}")
        print()
    else:
        print("⚠️  未找到 API Key 文件，将使用默认 key")
        print()

    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
