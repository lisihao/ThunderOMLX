# ThunderOMLX

**Mac mini 最强本地推理引擎 — 融合 omlx、ThunderLLAMA、ClawGate 的优势**

> 为 openClaw 打造的 Apple Silicon 原生推理引擎，配备 Web 管理面板和 macOS 菜单栏应用

---

## 🎯 项目目标

以 [omlx](https://github.com/jundot/omlx) 为底座，融合以下项目的核心优势：

| 项目 | 采用特性 |
|------|----------|
| **omlx** | Web UI + macOS 菜单栏应用 + 配置管理 |
| **ThunderLLAMA** | llama.cpp + Paged Attention + LMCache |
| **ClawGate** | 端云协同路由 + 智能回退 |
| **ContextPilot** | 上下文压缩 + Token 优化 |

**核心差异化**：
- ✅ Apple Silicon 原生优化（Metal GPU 加速）
- ✅ 端云协同（本地优先，云端回退）
- ✅ 两层缓存（内存 + SSD，跨会话复用）
- ✅ 用户友好（一键安装，菜单栏控制）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     ThunderOMLX Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  macOS Menubar App (PyObjC)                         │   │
│  │  - Start/Stop Server                                │   │
│  │  - Model Management                                 │   │
│  │  - System Monitoring                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │ HTTP                             │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Web UI (FastAPI + HTML/CSS/JS)                     │   │
│  │  - Dashboard                                        │   │
│  │  - Configuration                                    │   │
│  │  - Performance Metrics                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │ REST API                         │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ClawGate Router (端云协同)                         │   │
│  │  ┌─────────────┐         ┌─────────────┐            │   │
│  │  │ Local Path  │         │ Cloud Path  │            │   │
│  │  │ (Priority)  │         │ (Fallback)  │            │   │
│  │  └──────┬──────┘         └──────┬──────┘            │   │
│  │         │                       │                    │   │
│  │         ▼                       ▼                    │   │
│  │  ThunderLLAMA          OpenAI/Anthropic             │   │
│  │  (llama-server)        (Cloud API)                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Performance Optimization Layer                     │   │
│  │  - LMCache (L2: 内存 8GB, L3: SSD 256GB)            │   │
│  │  - ContextPilot (上下文压缩)                        │   │
│  │  - Paged Attention (KV Cache 优化)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 核心特性

### 1. Web 管理面板 + macOS 菜单栏应用

- 📊 **实时监控**：Token 生成速度、缓存命中率、路由状态
- ⚙️ **配置管理**：模型选择、端云切换、性能参数
- 🎮 **一键控制**：菜单栏启动/停止服务

### 2. 高性能 SSD 缓存（P1 完成）✅

- ⚡ **Smart Prefetch**：185x SSD 加速，4 线程并行预取
- 🔒 **Checksum Validation**：XXH64 数据完整性保护（-3.3% 开销）
- 💾 **Adaptive Chunk**：4x-16x 内存优化，动态分块加载
- 🗂️ **两层缓存**：Hot Cache（内存）+ SSD Cache（持久化）

### 3. ThunderLLAMA 推理引擎（待集成）

- 🔥 **Apple Silicon 优化**：Metal GPU 加速 + Unified Memory
- 📦 **Paged Attention**：无碎片化，稳定长时间运行
- 💾 **LMCache**：跨会话复用

### 4. ClawGate 端云协同（待集成）

- 🌐 **智能路由**：简单问题本地，复杂问题云端
- 💰 **成本优化**：本地优先，减少云端 API 费用
- 🔄 **自动回退**：本地失败时自动切换到云端

### 5. ContextPilot 上下文优化（待集成）

- 📉 **Token 压缩**：长对话自动压缩，节省成本
- 🎯 **关键信息保留**：智能保留重要上下文
- 📊 **可视化**："Token 减负账单"显示节省金额

---

## 📦 安装

### 方式 1：DMG 安装包（推荐）

1. 下载 `ThunderOMLX-1.0.0.dmg`
2. 双击打开 DMG
3. 拖动 `ThunderOMLX.app` 到 Applications 文件夹
4. 启动应用（菜单栏出现图标）

### 方式 2：从源码构建

```bash
# 1. Clone 仓库
git clone https://github.com/your-org/ThunderOMLX.git
cd ThunderOMLX

# 2. 安装依赖
cd packaging
pip install venvstacks

# 3. 构建应用
python build.py

# 4. 输出
# - build/ThunderOMLX.app
# - dist/ThunderOMLX-<version>.dmg
```

---

## 🎮 快速开始

### 1. 启动服务

点击菜单栏 ThunderOMLX 图标 → **Start Server**

### 2. 打开 Web 管理面板

浏览器访问：`http://localhost:8080`

### 3. 配置模型

Settings → 选择本地模型路径或云端 API Key

### 4. 发送请求

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, ThunderOMLX!",
    "max_tokens": 50
  }'
```

---

## 📊 性能对比

### P1 缓存优化实测数据（2026-03-13）

| 功能 | 基准 | P1 优化后 | 提升 |
|------|------|----------|------|
| **Smart Prefetch** | 144 ms/块（冷读 SSD）| 0.78 ms/块（预取命中）| **185x** ✅ |
| **Adaptive Chunk** | 32 MB（8K tokens）| 2 MB（分块加载）| **16x** ✅ |
| **Checksum 验证** | - | -3.3% 开销（缓存后）| **比无校验还快** ✅ |
| **缓存块对齐率** | ~70% | 100% | **+30%** ✅ |

### 与 omlx (mlx-lm) 对比

| 指标 | omlx (mlx-lm) | ThunderOMLX (P1) | 提升 |
|------|---------------|------------------|------|
| **Prefill 速度** | 2907 t/s | 2830 t/s | -2.6% |
| **Decode 速度** | 239 t/s | 247 t/s | +3.3% |
| **SSD 缓存加速** | - | **185x** | ✅ |
| **长 prompt 内存** | - | **-94%** (16x) | ✅ |
| **数据完整性** | 无保障 | 100% 校验 | ✅ |
| **显存占用** | 26GB | 13GB（目标）| -50%（待实现）|

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **UI** | PyObjC (菜单栏) + Web UI (FastAPI) |
| **后端** | FastAPI + uvicorn |
| **推理引擎** | llama.cpp (ThunderLLAMA) |
| **路由** | ClawGate (端云协同) |
| **缓存** | LMCache (内存 + SSD) |
| **上下文优化** | ContextPilot |
| **打包** | venvstacks + DMG |

---

## 📁 项目结构

```
ThunderOMLX/
├── .solar/              # 状态管理
│   └── STATE.md
├── src/                 # 源码（Fork 自 omlx）
│   ├── omlx/            # FastAPI 后端
│   ├── omlx_app/        # macOS 菜单栏应用
│   └── packaging/       # 打包配置
├── docs/                # 文档
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── scripts/             # 脚本
│   ├── build.sh
│   └── test.sh
├── tests/               # 测试
└── README.md
```

---

## 🗓️ Roadmap

| 阶段 | 时间 | 状态 |
|------|------|------|
| Phase 0: 项目搭建 | 1 天 | ✅ 完成 |
| **Phase 1: 缓存优化** | 3 天 | ✅ 完成 |
| Phase 2: 推理引擎替换 | 2-3 天 | ⏸️ 待开始 |
| Phase 3: ClawGate 集成 | 3-4 天 | ⏸️ 待开始 |
| Phase 4: 性能优化 | 2-3 天 | ⏸️ 待开始 |
| Phase 5: 打包分发 | 2-3 天 | ⏸️ 待开始 |
| Phase 6: 文档和发布 | 1-2 天 | ⏸️ 待开始 |

### ✅ Phase 1 完成成果（2026-03-13）

**P1-5: Smart Prefetch** - 智能预取
- ✅ 185x SSD 加速（远超目标 2-4x）
- ✅ 4 线程并行预取
- ✅ 访问频率追踪

**P1-6: Checksum Validation** - 数据完整性
- ✅ XXH64 快速校验（~10 GB/s）
- ✅ 缓存优化：-3.3% 开销（比无校验还快）
- ✅ 自动损坏检测和删除

**P1-7: Adaptive Chunk Prefill** - 自适应分块
- ✅ 4x-16x 内存优化
- ✅ 缓存块对齐（100% 对齐率）
- ✅ 动态 chunk size 计算

**详细报告**: [P1_FINAL_SUMMARY.md](./P1_FINAL_SUMMARY.md)

---

## 🤝 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## 📄 License

Apache 2.0 License（继承自 omlx）

---

## 🙏 致谢

本项目基于以下优秀开源项目：
- [omlx](https://github.com/jundot/omlx) - UI 和框架基础
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 推理引擎核心
- [ThunderLLAMA](https://github.com/your-org/ThunderLLAMA) - Paged Attention + LMCache
- [ClawGate](https://github.com/your-org/ClawGate) - 端云协同路由

---

**Made with ⚡ for Apple Silicon by openClaw Team**
