# ThunderOMLX 快速开始

> 5 分钟快速上手开发

---

## 环境要求

- macOS 15.0+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 20GB+ 可用磁盘空间

---

## 开发环境搭建

### 1. Clone 项目

```bash
cd ~
git clone https://github.com/your-org/ThunderOMLX.git
cd ThunderOMLX
```

### 2. 激活虚拟环境

```bash
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装 omlx 依赖（如果需要运行原 UI）
cd src
pip install -e .
cd ..
```

### 4. 配置 ThunderLLAMA

确保 ThunderLLAMA llama-server 已编译：

```bash
# 检查 llama-server 是否存在
ls ~/ThunderLLAMA/build/bin/llama-server

# 如果不存在，先编译 ThunderLLAMA
cd ~/ThunderLLAMA
mkdir -p build && cd build
cmake .. -DGGML_METAL=ON
cmake --build . --config Release
```

### 5. 下载测试模型

```bash
# 创建模型目录
mkdir -p ~/Models

# 下载 TinyLlama（用于测试，约 600MB）
cd ~/Models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## 快速测试

### 1. 启动 ThunderLLAMA llama-server

```bash
~/ThunderLLAMA/build/bin/llama-server \
  --model ~/Models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --port 8081 \
  --ctx-size 4096 \
  --n-gpu-layers 99
```

### 2. 测试 llama-server API

```bash
curl -X POST http://localhost:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, ThunderOMLX!",
    "max_tokens": 50
  }'
```

### 3. 运行原 omlx UI（验证环境）

```bash
cd ~/ThunderOMLX/src
python -m omlx
```

浏览器打开：`http://localhost:8080`

---

## 项目结构说明

```
ThunderOMLX/
├── .solar/              # 项目状态管理
│   └── STATE.md         # Mission + 计划 + 进度
├── src/                 # omlx 源码（Fork 自上游）
│   ├── omlx/            # FastAPI 后端
│   ├── omlx_app/        # macOS 菜单栏应用
│   └── packaging/       # 打包配置
├── docs/                # 文档
│   ├── ARCHITECTURE.md  # 系统架构
│   └── QUICKSTART.md    # 本文档
├── scripts/             # 开发脚本
├── tests/               # 测试套件
├── venv/                # Python 虚拟环境
└── requirements.txt     # 依赖清单
```

---

## 开发工作流

### 1. 创建功能分支

```bash
git checkout -b feature/clawgate-router
```

### 2. 编写代码

```bash
# 创建新模块
mkdir -p src/thunderomlx/router
touch src/thunderomlx/router/clawgate.py
```

### 3. 运行测试

```bash
pytest tests/
```

### 4. 代码格式化

```bash
black src/
ruff check src/
```

### 5. 提交代码

```bash
git add .
git commit -m "feat: implement ClawGate router"
```

---

## 常用命令

### 启动开发服务器

```bash
cd ~/ThunderOMLX/src
uvicorn omlx.server:app --reload --port 8080
```

### 运行测试

```bash
# 所有测试
pytest

# 单个文件
pytest tests/test_router.py

# 带覆盖率
pytest --cov=thunderomlx
```

### 代码质量检查

```bash
# 格式化
black src/ tests/

# Lint
ruff check src/ tests/

# 类型检查
mypy src/
```

---

## 下一步

- [ ] 阅读 [ARCHITECTURE.md](./ARCHITECTURE.md) 理解系统设计
- [ ] 查看 [.solar/STATE.md](../.solar/STATE.md) 了解当前进度
- [ ] 开始 Phase 1：推理引擎替换

---

## 故障排查

### llama-server 启动失败

```bash
# 检查端口占用
lsof -i :8081

# 检查模型文件
ls -lh ~/Models/*.gguf

# 查看日志
~/ThunderLLAMA/build/bin/llama-server --log-file /tmp/llama-server.log
```

### Python 依赖冲突

```bash
# 重建虚拟环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

**Happy Hacking!** 🚀
