# ThunderOMLX - Mac mini 最强推理引擎

## Mission

以 omlx 为底座，融合 ThunderLLAMA、ClawGate、LMCache、ContextPilot 等项目的优势特性，为 openClaw 打造 Apple Silicon Mac mini 上的最强本地推理引擎。

## Constraints

- 保持 omlx 的 UI/UX（Web 管理面板 + macOS 菜单栏应用）
- 推理引擎必须使用 ThunderLLAMA (llama.cpp)，不使用 mlx-lm
- 集成 ClawGate 端云协同能力（本地优先，云端回退）
- 性能无衰退（相比原 omlx 的 mlx-lm）
- 打包成 DMG，用户友好安装
- 代码质量：测试覆盖率 > 80%

## Current Plan

### Phase 0: 项目搭建 (1 天) — In Progress

1. ⏳ **创建项目结构**
   - 初始化 Git 仓库
   - 创建基础目录（.solar, docs, scripts, tests）
   - 编写 README.md（项目介绍、架构图）

2. ⏳ **Fork omlx 代码**
   - Clone omlx 到 ThunderOMLX 项目
   - 保留 UI 层（Web + macOS 菜单栏）
   - 保留 FastAPI 路由框架
   - 保留配置管理（settings.json）

3. ⏳ **规划架构设计**
   - 绘制系统架构图（omlx UI → FastAPI → ThunderLLAMA/ClawGate）
   - 定义模块边界和接口
   - 编写技术选型文档

### Phase 1: 推理引擎替换 (2-3 天)

1. **移除 mlx-lm 依赖**
   - 删除 mlx、mlx-lm、mlx-vlm 相关代码
   - 删除 mlx 特定的缓存逻辑

2. **集成 ThunderLLAMA**
   - 实现 HTTP 客户端调用 llama-server
   - 参数映射（temperature、max_tokens、stop_sequences）
   - 流式输出适配
   - 错误处理和重试逻辑

3. **内嵌 llama-server 二进制**
   - 从 ThunderLLAMA 拷贝编译好的 llama-server
   - 修改启动脚本，自动启动 llama-server
   - 健康检查和进程管理

### Phase 2: ClawGate 端云协同 (3-4 天)

1. **集成 ClawGate 路由层**
   - 添加端云切换逻辑（本地优先，云端回退）
   - 配置云端 API（OpenAI/Anthropic）
   - 实现智能路由（简单问题本地，复杂问题云端）

2. **配置管理增强**
   - 添加 ClawGate 配置段（云端 API Key、路由策略）
   - Web UI 添加端云切换开关
   - 成本和延迟监控

### Phase 3: 性能优化 (2-3 天)

1. **集成 LMCache**
   - ThunderLLAMA 启用 LMCache（两层缓存）
   - Web UI 显示缓存命中率

2. **集成 ContextPilot**
   - 上下文压缩和优化
   - Token 节省可视化

3. **性能测试**
   - 对比 omlx (mlx-lm) vs ThunderOMLX (llama.cpp)
   - P50/P95/P99 延迟测试
   - 缓存加速验证

### Phase 4: 打包分发 (2-3 天)

1. **修改 venvstacks 配置**
   - 移除 mlx 依赖
   - 添加 llama.cpp Python 绑定（可选）
   - 配置 App 元数据（名称、图标、版本）

2. **创建打包脚本**
   - 一键打包成 .app bundle
   - 生成 DMG 分发包
   - 代码签名（可选）

3. **测试分发流程**
   - 在干净的 macOS 上测试安装
   - 验证所有功能正常

### Phase 5: 文档和发布 (1-2 天)

1. **用户文档**
   - 安装指南
   - 使用教程
   - 配置说明
   - 故障排查

2. **开发者文档**
   - 架构设计
   - API 文档
   - 贡献指南

3. **发布准备**
   - GitHub Release
   - 更新日志
   - 演示视频

## Decisions

- [2026-03-13] **项目启动决策**
  - 选择 omlx 作为底座（成熟的 UI + macOS 菜单栏应用）
  - 推理引擎使用 ThunderLLAMA（Apple Silicon 优化 + Paged Attention）
  - 集成 ClawGate（端云协同，降低成本）
  - 打包方式：venvstacks（omlx 原有方案，成熟可靠）
  - 预计工期：2-3 周

- [2026-03-13] **技术栈决策**
  - UI 层：保留 omlx 的 Web UI + PyObjC 菜单栏应用
  - 后端：FastAPI（保留 omlx 框架）
  - 推理引擎：llama.cpp (ThunderLLAMA) + ClawGate 路由
  - 缓存：LMCache（两层：内存 + SSD）
  - 上下文优化：ContextPilot
  - 打包：venvstacks + DMG

## Progress

### In-Progress

- 项目结构创建
- README.md 编写
- 架构设计

### Done

- ✅ 项目目录创建 (/Users/lisihao/ThunderOMLX)
- ✅ Git 仓库初始化
- ✅ .solar/STATE.md 创建

### Blocked

(无)

## Next Actions

1. **Fork omlx 代码** (30 分钟)
   ```bash
   cd ~/ThunderOMLX
   git clone https://github.com/jundot/omlx.git src
   ```

2. **编写 README.md** (30 分钟)
   - 项目介绍
   - 核心特性
   - 架构图
   - 快速开始

3. **绘制架构图** (1 小时)
   - 系统架构
   - 数据流
   - 模块边界

4. **创建开发环境** (1 小时)
   - Python 虚拟环境
   - 安装依赖
   - 配置 IDE
