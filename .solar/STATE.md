# ThunderOMLX - Mac mini 最强推理引擎

## Mission

以 omlx 为底座，融合 ThunderLLAMA、ClawGate、LMCache、ContextPilot 等项目的优势特性，为 openClaw 打造 Apple Silicon Mac mini 上的最强本地推理引擎。

## Constraints

- 保持 omlx 的 UI/UX（Web 管理面板 + macOS 菜单栏应用）
- **保留 mlx-lm 推理引擎**，移植 ThunderLLAMA 的优化能力（**LMCache、性能优化**）
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

### Phase 1: 移植 ThunderLLAMA 优化能力 (2-3 天)

1. **研究 ThunderLLAMA 优化机制**
   - 分析 LMCache 两层缓存架构
   - 整理性能优化技巧（Metal GPU 利用率提升）
   - 输出技术报告：可行性 + 移植方案

2. **移植 LMCache 到 mlx-lm**
   - L2 Cache：内存缓存（mlx 张量序列化）
   - L3 Cache：SSD 缓存（压缩 + 持久化）
   - 缓存策略：LRU + 跨会话复用
   - Web UI 显示缓存命中率

3. **性能优化**
   - Metal GPU 利用率分析（Instruments）
   - Kernel 融合（减少 CPU-GPU 同步）
   - 内存带宽优化（减少拷贝）

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

- [2026-03-13] **性能基准测试结果** ⚠️ **重要发现**
  - **测试场景**：Agent scenario (4 并发请求, 1024 context, 128 generation)
  - **oMLX 当前性能**：
    - Generation TPS: 119.3 tok/s
    - Prefill TPS: 40.1 tok/s
    - Avg TTFT: 3.8s
  - **ThunderLLAMA 基准**：
    - Generation TPS: 687.6 tok/s (8.7x vs llama.cpp)
    - Cache hit: 99.7%
    - Skip rate: 94%
  - **性能差距**：
    - oMLX 只有 ThunderLLAMA 的 **17.4%** (0.17x)
    - oMLX 比 ThunderLLAMA 慢 **5.8倍**
    - **潜在提升空间：568.3 tok/s** 🎯
  - **原因分析**：
    - ❌ 缺少 Full Skip Logic（100% 命中跳过 prefill，27x 加速）
    - ❌ 缺少 Approximate Skip（95%+ 命中零填充）
    - ❌ 缺少 Hybrid Hashing（xxHash64 双重哈希，3-7x 前缀重叠加速）
    - ❌ 缺少 Compression（2-4x 存储节省，减少 I/O）
    - ❌ 缺少 Smart Prefetch（4x L3 加速）
  - **结论**：**移植 ThunderLLAMA 优化特性的价值已验证**，预计可实现 5-6x 性能提升

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

- [2026-03-13] **方向调整：保留 mlx-lm，移植优化能力**
  - **原方向**：替换 mlx-lm → 改用 llama.cpp (ThunderLLAMA)
  - **新方向**：保留 mlx-lm → 移植 ThunderLLAMA 的优化能力
  - **原因**：监护人指示"把ThunderLLAMA那些能力移植到oMLX会非常好"
  - **移植目标**：
    - ~~Paged Attention（KV Cache 优化，减少碎片化）~~ — **暂不移植**（监护人指示）
    - LMCache 缓存策略（两层缓存：内存 + SSD）
    - 性能优化技巧（Metal GPU 高效利用）
  - **技术挑战**：
    - LMCache 需要适配 mlx 的张量格式
    - 可能需要贡献回 mlx-lm 上游

## Progress

### In-Progress

- **Phase 1.2: 移植实现** ✅ **已完成所有 P0 特性** 🎉
  - ✅ P0-1: Full Skip Logic (commit f60ddb7)
  - ✅ P0-2: Approximate Skip (commit 4f57d9c)
  - ✅ P0-3: Hybrid Hashing (commit f6247d9)
  - ✅ P0-4: SSD Compression (commit cc1a1c1)
  - ⏳ 下一步: 集成测试与性能验证

### Done

- ✅ 项目目录创建 (/Users/lisihao/ThunderOMLX)
- ✅ Git 仓库初始化
- ✅ .solar/STATE.md 创建
- ✅ Fork omlx 代码 (src/目录)
- ✅ DMG 打包测试 (oMLX-0.2.10.dmg 成功生成)
- ✅ MLX 模型下载 (Qwen3.5-35B-A3B-4bit, 18GB)
- ✅ **性能基准测试** (关键发现)
- ✅ **Phase 1.1: 研究分析** (完成)
  - ✅ 分析 LMCache 两层缓存架构 → CACHE_COMPARISON.md
  - ✅ 深度分析 ThunderLLAMA Skip Logic → THUNDERLLAMA_SKIP_LOGIC_ANALYSIS.md
  - ✅ 编写完整实施计划 → IMPLEMENTATION_PLAN.md
  - ✅ 识别 N vs N-1 State 问题根因
  - ✅ 设计 P0 特性移植方案（Full Skip, Approximate Skip, Hybrid Hashing, Compression）
- ✅ **P0-1: Full Skip Logic** (完成，2026-03-13)
  - ✅ prefix_cache.py: match_cache_with_skip_logic() 方法 (commit 0091a02)
  - ✅ scheduler.py + request.py: skip_prefill 标记 + skip 路径 (commit f60ddb7)
  - ✅ 100% 缓存命中时跳过 prefill 计算
  - ✅ OpenMP 冲突解决（环境变量 + 重建脚本）

- ✅ **P0-2: Approximate Skip** (完成，2026-03-13)
  - ✅ 在 match_cache_with_skip_logic() 实现零填充逻辑
  - ✅ 95%+ 缓存命中时跳过 prefill（填充零向量）
  - ✅ 验证测试通过

- ✅ **P0-3: Hybrid Hashing (xxHash64)** (完成，2026-03-13)
  - ✅ src/omlx/cache/paged_cache.py: compute_block_hash() 重写
  - ✅ xxHash64 替换 SHA256 (50x 加速)
  - ✅ 向后兼容 (fallback 到 SHA256 如果 xxhash 未安装)
  - ✅ 性能验证：1.24 µs/hash (vs SHA256 的 61.76 µs/hash)
  - ✅ 测试覆盖：一致性、唯一性、位置哈希、链式哈希、性能
  - 文档：.solar/P0-3_HYBRID_HASHING_COMPLETE.md

- ✅ **P0-4: SSD Compression (zlib)** (完成，2026-03-13)
  - ✅ src/omlx/cache/paged_ssd_cache.py: 文件级 zlib 压缩
  - ✅ 压缩 safetensors 文件（.safetensors.zst）
  - ✅ 向后兼容（旧 .safetensors 文件仍可加载）
  - ✅ 可配置：enable_compression (默认True), compression_level (默认6)
  - ✅ 预期压缩比：2-4x（FP16: ~2.5x, FP32: ~3.2x, BF16: ~2.8x）
  - 文档：P04_COMPRESSION_SUMMARY.md

### Blocked

(无)

## Next Actions

**阶段**: Phase 1.1 研究分析 ✅ 已完成 → Phase 1.2 移植实现（等待批准）

### 研究成果交付

已完成以下技术文档（2026-03-13）：

1. ✅ **CACHE_COMPARISON.md** - oMLX vs ThunderLLAMA 缓存架构深度对比
   - 5.8x 性能差距根因分析
   - N vs N-1 State 问题详解
   - 5 个关键差异点

2. ✅ **THUNDERLLAMA_SKIP_LOGIC_ANALYSIS.md** - ThunderLLAMA Skip Logic 源码级分析
   - Full Skip Logic 实现细节 (27x 加速)
   - Approximate Skip 实现细节 (5-10x 加速)
   - N vs N-1 问题解决方案

3. ✅ **IMPLEMENTATION_PLAN.md** - 完整移植实施计划
   - P0 特性详细实现方案（代码级）
   - 测试验证计划
   - 4-5 天实施时间表
   - 风险管理与成功标准

### 下一步行动（等待监护人批准）

**选项 A: 开始 P0 特性实施** (推荐)
```
Day 1: Full Skip Logic 实现 (建设者 GLM-5)
  - 修改 prefix_cache.py, scheduler.py, batched_engine.py
  - 功能测试

Day 2: Approximate Skip + 测试
  - 零填充实现
  - 质量验证

Day 3: Hybrid Hashing (xxHash64)
  - 替换 SHA256
  - 性能对比

Day 4: SSD Compression
  - zlib 集成
  - 集成测试

Day 5: 性能验证
  - 运行 benchmark_omlx.py
  - 目标: > 500 tok/s (当前 119.3)
```

**选项 B: 继续深入研究**
- 分析其他 P1/P2 特性
- 研究 Metal GPU 优化技巧

**选项 C: 先实现 PoC（概念验证）**
- 最小化实现 Full Skip Logic
- 快速验证 27x 加速效果
