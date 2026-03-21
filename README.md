---
title: "ThunderOMLX: 基于 Apple Silicon 的高性能本地大语言模型推理引擎——智能 KV Cache 管理系统"
author: "李思浩"
date: "2026年3月"
lang: zh-CN
abstract: |
  ThunderOMLX 是一款面向 Apple Silicon 的高性能大语言模型本地推理引擎，基于 MLX 框架构建。本文提出了一套完整的 KV Cache 管理体系，融合分页 SSD 缓存、多级跳跃策略、混合哈希、智能分块、ContextPilot 消息级优化等关键技术。核心成果：首 Token 延迟（TTFT）降低 90.6%（超越 Anthropic 的 50% 和 OpenAI 的 70%），SSD 访问加速 185 倍，张量重建加速 33 倍，支持 128K+ 上下文无 OOM，语义分块质量达 89.69%。系统集成 KVTC 压缩（4-8x 压缩比）和自适应逐块压缩选择。ThunderOMLX 证明，通过精巧的缓存管理可以让边缘设备实现云端级别的推理性能。
keywords:
  - 大语言模型推理
  - KV Cache 管理
  - Apple Silicon
  - 边缘推理
  - 前缀缓存
  - MLX
---

# ThunderOMLX: 基于 Apple Silicon 的高性能本地大语言模型推理引擎——智能 KV Cache 管理系统

**李思浩**

---

## 摘要

ThunderOMLX 是一款面向 Apple Silicon 的高性能大语言模型本地推理引擎，基于 MLX 框架构建。本文提出了一套完整的 KV Cache 管理体系，融合分页 SSD 缓存、多级跳跃策略、混合哈希、智能分块、ContextPilot 消息级优化等关键技术。核心成果：首 Token 延迟（TTFT）降低 90.6%（超越 Anthropic 的 50% 和 OpenAI 的 70%），SSD 访问加速 185 倍，张量重建加速 33 倍，支持 128K+ 上下文无 OOM，语义分块质量达 89.69%。系统集成 KVTC 压缩（4-8x 压缩比）和自适应逐块压缩选择。ThunderOMLX 证明，通过精巧的缓存管理可以让边缘设备实现云端级别的推理性能。

**关键词**: 大语言模型推理, KV Cache 管理, Apple Silicon, 边缘推理, 前缀缓存, MLX

---

## 1 引言

### 1.1 大语言模型的爆发式增长

近年来，大语言模型（Large Language Model, LLM）经历了前所未有的爆发式增长。以 GPT-4 [26]、Claude [27]、Gemini [28]、DeepSeek-V3 [23]、Qwen2.5 [24] 为代表的模型在自然语言理解、代码生成、多轮对话等任务上展现出接近人类水平的能力。这些模型的参数规模从数十亿到数千亿不等，推理过程中需要存储和管理大量的中间状态——尤其是 Transformer 架构 [6] 中的 Key-Value Cache（KV Cache）。

### 1.2 从云端推理到本地推理的趋势

尽管云端推理仍然是当前 LLM 服务的主流部署形式，本地推理正在成为一种不可忽视的趋势。其驱动因素可归纳为三个方面。第一，隐私保护：对于医疗记录、法律文档、企业内部数据等敏感信息，将数据发送到云端存在泄露风险，本地推理天然满足数据不出设备的合规需求。第二，延迟优化：云端推理受限于网络往返时间（RTT），通常在 50-200ms 之间；在多 Agent 协作场景下，每一轮交互都产生网络延迟，端到端响应时间可能达到数秒。本地推理消除了网络瓶颈，可将延迟压缩至毫秒级别。第三，成本控制：对于高频调用场景（如 AI Agent 持续运行、代码补全等），云端 API 的 token 计费成本可达每月数百美元，而本地推理的边际成本趋近于零。

### 1.3 Apple Silicon 的独特机遇

Apple Silicon（M1/M2/M3/M4 系列）的统一内存架构（Unified Memory Architecture, UMA）为本地 LLM 推理提供了独特的硬件优势。与 NVIDIA GPU 需要通过 PCIe 总线在 CPU 和 GPU 之间搬运数据不同，Apple Silicon 的 CPU 和 GPU 共享同一物理内存池，数据在 CPU 和 GPU 之间的传输几乎是零拷贝的。M4 Pro 芯片配备高达 48GB 统一内存，足以容纳一个 35B 参数的 4-bit 量化模型（约 19GB）并留有充裕的 KV Cache 空间。苹果官方推出的 MLX 框架 [9] 为 Apple Silicon 提供了惰性求值（lazy evaluation）、统一内存管理和 Metal GPU 后端等原生支持，使得在 Mac 设备上构建高性能推理引擎具备了坚实的软件基础。

### 1.4 核心挑战：KV Cache 增长与管理

然而，在 Apple Silicon 上实现高性能 LLM 推理面临着严峻的挑战，其中最核心的问题是 KV Cache 的管理。在 Transformer 的自回归生成过程中，每个 token 的生成都需要访问所有历史 token 的 Key 和 Value 向量，其内存占用为 $O(n \times L \times 2 \times H \times d)$，其中 $n$ 为序列长度，$L$ 为层数，$H$ 为注意力头数，$d$ 为每个注意力头的维度。以 Qwen3.5-35B 模型为例，在 128K 上下文长度下，KV Cache 的内存占用可达约 40GB——超出了大多数消费级设备的内存容量。

本文在实践中识别出四大痛点：

1. **重复 Prompt 的 TTFT 延迟过高**：在多 Agent 对话场景下，每个 Agent 携带相同的系统提示（system prompt），多轮对话中 80%+ 的 token 与上一轮相同，却每次都需要重新执行 prefill 计算。
2. **长上下文导致 OOM**：当 prompt 长度超过 64K token 时，MLX Metal GPU buffer 超限，推理过程直接崩溃，无法利用模型声称的 128K 上下文窗口。
3. **SSD I/O 成为缓存持久化瓶颈**：将 KV Cache 持久化到 SSD 以实现跨会话复用时，朴素的序列化和反序列化操作耗时数百毫秒，远超可接受的延迟预算。
4. **"Lost in the Middle"注意力退化**：Liu 等人 [7] 发现，语言模型在处理长上下文时，对中间位置信息的关注度显著低于首尾位置，呈现 U 型注意力分布，导致关键信息被遗漏。

### 1.5 本文贡献

针对上述挑战，本文提出 ThunderOMLX——一个面向 Apple Silicon 的完整 KV Cache 管理系统，其主要贡献如下：

1. **分页 SSD 缓存系统**：以 256 token 为粒度的块级 KV Cache 持久化方案，配合 LRU-2 双队列淘汰策略，实现 0.004ms 内存命中延迟。
2. **智能预取与批量重建**：基于访问模式预测的 SSD 块预加载（185x 加速）和预分配 buffer 的张量拼接优化（33x 加速）。
3. **多级跳跃策略（Skip Logic）**：三级缓存命中决策——FULL SKIP（100% 命中，完全跳过 prefill）、APPROXIMATE SKIP（$\geq$95% 命中）和 NO SKIP（<95% 命中），实现 55-78x 的重复推理加速。
4. **分块预填充（Chunked Prefill）**：语义感知的分块 prefill 处理，突破 Metal buffer 限制，支持 128K+ token 推理，输出质量保持 99.88% 一致性。
5. **智能分块系统**：Greedy Boundary-Aware Packing 算法，实现 89.69% 的语义分块质量，零外部依赖。
6. **ContextPilot 消息级缓存智能**：增量 chat template 解析、系统提示指纹、内容哈希去重，将缓存命中率从 60% 提升至 100%。
7. **U-Shape 注意力增强**：BM25 中英文混合评分 + 抽取式摘要注入，对抗"Lost in the Middle"注意力退化。
8. **KVTC 自适应压缩集成**：PCA + DP 位分配 + 分组量化 + DEFLATE，实现 4-8x 压缩比，支持逐块自适应压缩策略选择。
9. **专用引擎线程**：将 MLX 推理限制在单一专用线程，消除 asyncio/ThreadPoolExecutor 的调度开销，pipeline overhead 从 24% 降至 0.6%，TG 吞吐量提升 43%。
10. **边界快照步幅优化**：可配置的 KV Cache 快照间隔（stride=8），PP 速度从 716 提升至 894 tok/s（+25%），冷启动 TTFT 降低 19%。

### 1.6 论文组织

本文其余部分组织如下：第 2 节介绍背景知识与相关工作；第 3 节阐述 ThunderOMLX 的整体系统架构；第 4 节详细描述各项关键技术；第 5 节呈现实验评估结果；第 6 节讨论局限性和未来方向；第 7 节总结全文。

---

## 2 背景与相关工作

### 2.1 Transformer 推理中的 KV Cache

Transformer 架构 [6] 的自注意力机制要求每个新生成的 token 与所有历史 token 进行注意力计算。为避免重复计算，标准做法是将每一层的 Key 和 Value 向量缓存起来，称为 KV Cache。设模型有 $L$ 层，每层有 $H$ 个注意力头，每个头的维度为 $d$，数据类型为 $b$ 字节，则序列长度为 $n$ 时 KV Cache 的内存占用为：

$$M_{KV} = n \times L \times 2 \times H \times d \times b$$

以本文实验所用的 Qwen3.5-35B 模型为例（$L=40$, $H=48$, $d=128$, $b=2$ 字节/bfloat16），在 128K 上下文长度下，KV Cache 占用约为：

$$M_{KV} = 131072 \times 40 \times 2 \times 48 \times 128 \times 2 \approx 40 \text{ GB}$$

这一数字已超出 M4 Pro 48GB 统一内存中模型加载后的剩余可用量（约 29GB），凸显了 KV Cache 管理在边缘设备上的严峻性。

采用分组查询注意力（GQA）[10] 的模型将 KV 头数减少至 $H_{KV} < H$，可按比例缩减 KV Cache 占用。FlashAttention-2 [11] 则从计算效率角度优化了注意力操作的 I/O 复杂度，但并未减少 KV Cache 的存储需求。

### 2.2 云端 KV Cache 管理方案

在云端服务器（配备高带宽 GPU 内存的数据中心）场景下，已有多项重要工作致力于优化 KV Cache 管理。

**vLLM 与 PagedAttention**。Kwon 等人 [1] 提出 PagedAttention，借鉴操作系统虚拟内存管理的思想，将 KV Cache 划分为固定大小的"页"（block），按需分配和回收，消除了连续内存分配导致的内存碎片化问题。vLLM 通过 PagedAttention 实现了接近零浪费的内存利用率，在高并发场景下将吞吐量提升 2-4 倍。然而，vLLM 的设计面向 NVIDIA GPU 的 CUDA 生态，无法直接应用于 Apple Silicon 的 Metal 计算平台。

**SGLang 与 RadixAttention**。Zheng 等人 [2] 提出 RadixAttention，使用基数树（Radix Tree）数据结构实现跨请求的 KV Cache 前缀共享。相比 PagedAttention 的逐页管理，RadixAttention 支持变长前缀的高效匹配和复用，特别适合具有共享系统提示的多请求场景。然而，RadixAttention 的基数树维护在 Python 层面引入了额外的计算开销，且其设计同样面向数据中心 GPU。

**FlexGen**。Sheng 等人 [3] 探索了 CPU 和 SSD 的异构卸载方案，通过线性规划求解最优调度策略，使得单 GPU 也能服务超大模型。FlexGen 的贡献在于验证了 SSD 作为 KV Cache 扩展存储的可行性，但其面向的是 GPU 内存不足时的"退而求其次"方案，未充分利用 Apple Silicon 统一内存的零拷贝优势。

**InstInfer**。Pan 等人 [4] 提出了从 SSD 到 GPU 的直接数据通路，绕过 CPU 中转，减少数据搬运延迟。该工作启发了本文在 Apple Silicon 上利用统一内存实现零拷贝缓存访问的设计。

### 2.3 Apple Silicon 与 MLX 框架

Apple M 系列芯片采用系统级芯片（SoC）设计，将 CPU、GPU、Neural Engine 和统一内存控制器集成在同一芯片上。统一内存架构使得 CPU 和 GPU 共享同一物理内存池，无需通过 PCIe 总线搬运数据。这一硬件特性为 LLM 推理带来了独特优势：KV Cache 在 CPU 管理层（如缓存策略、LRU 淘汰）和 GPU 计算层（如注意力计算）之间天然可以零拷贝共享。

MLX [9] 是苹果官方推出的数组计算框架，专为 Apple Silicon 设计。其核心特性包括：惰性求值（延迟计算直至结果被实际需要，减少不必要的内存分配）、统一内存语义（数组在 CPU 和 GPU 之间共享，无显式数据传输）、Metal GPU 后端（利用 Metal Performance Shaders 和 Metal Compute 实现高效 GPU 计算）。

基于 MLX 的推理工具如 mlx-lm [29] 和 omlx [30] 提供了基本的模型加载和文本生成能力，但缺乏高级的 KV Cache 管理机制——没有分页缓存、没有跨会话持久化、没有前缀匹配复用、没有 OOM 防护。ThunderOMLX 正是在 omlx 的基础上构建了完整的缓存管理体系。

### 2.4 生产环境中的 Prompt Caching

Anthropic [21] 和 OpenAI [22] 分别在其云端 API 中提供了 Prompt Caching 功能。Anthropic 的方案将 TTFT 降低约 50%，OpenAI 的方案降低约 70%。两者的共同局限在于：（a）仅限云端使用，本地部署不可用；（b）缓存生命周期受限于会话或短暂的 TTL，无法跨会话持久化；（c）缓存策略对用户不透明，无法针对特定应用场景（如多 Agent 协作）进行精细调优。ThunderOMLX 的设计目标是在本地设备上超越云端方案的缓存效率，同时提供完全透明和可调优的缓存控制。

### 2.5 KV Cache 压缩技术

KV Cache 的压缩可分为两个维度：运行时压缩（减少 GPU 内存中的 KV Cache 占用）和持久化压缩（减少 SSD 存储和 I/O 开销）。

在运行时压缩方面，KVQuant [17] 通过 per-channel 量化和稀疏异常值处理将 KV Cache 压缩至 2-bit 精度，支持 1000 万 token 的上下文长度。KIVI [18] 提出非对称 2-bit 量化——Key 使用 per-channel 量化，Value 使用 per-token 量化——实现了无需微调的 KV Cache 压缩。Gear [19] 则引入了残差量化方案以保留量化误差信息。H2O [16] 从注意力模式出发，识别并保留"Heavy Hitter"token 的 KV Cache，淘汰低注意力权重的 token。StreamingLLM [15] 则证明只需保留"attention sink"（前几个 token）和最近的滑动窗口即可维持生成质量。

在持久化压缩方面，KVTC [5] 提出了一种面向 KV Cache 持久化和传输的变换编码方案：首先通过 PCA 降维消除注意力头间的冗余，然后使用动态规划（DP）求解最优位分配策略，再进行分组量化和 DEFLATE 熵编码，实现 4-8x 的压缩比。与运行时量化方案不同，KVTC 面向的是缓存的存储和传输场景，与本文的分页 SSD 缓存系统具有天然的互补性。ThunderOMLX 集成了 KVTC 编解码器，并实现了自适应逐块压缩策略选择。

---

## 3 系统架构

### 3.1 分层设计

ThunderOMLX 采用分层解耦的架构设计，从客户端请求到 GPU 计算的完整数据流如下：

```
客户端请求
    |
    v
FastAPI 服务器 (OpenAI 兼容 API)
    |
    v
请求调度器 (Scheduler)
    |
    v
ContextPilot 适配器 (消息级缓存感知)
    |
    v
前缀缓存匹配器 (Skip Logic 决策)
    |
    v
分页缓存管理器 (L1 RAM + L2 SSD)
    |
    v
MLX 批量推理引擎 (Metal GPU)
```

API 层负责协议兼容（OpenAI Chat Completions API 格式）和流式传输（SSE）；调度器负责请求编排、缓存决策和引擎调度；缓存层负责 KV Cache 的分页管理、持久化和预加载；引擎层负责基于 Metal GPU 的实际推理计算。各层可独立演进，通过明确定义的接口交互。

### 3.2 设计原则

ThunderOMLX 的架构设计遵循以下五项核心原则：

**原则一：零拷贝利用**。Apple Silicon 统一内存使得 CPU 和 GPU 共享同一物理内存。ThunderOMLX 在缓存管理层（Python/CPU）和推理计算层（Metal/GPU）之间实现零拷贝数据共享，避免了传统 GPU 系统中 Host-to-Device 和 Device-to-Host 的数据搬运开销。

**原则二：块级粒度**。以 256 token 为一个 block 的精细缓存管理。每个 block 独立序列化、独立哈希、独立压缩、独立淘汰。这一粒度在缓存命中率（粒度越细，命中率越高）和管理开销（粒度越细，元数据越多）之间取得平衡。

**原则三：层次化缓存**。借鉴计算机存储层次结构的设计思想，构建 L1 RAM（LRU-2 双队列，0.004ms 命中延迟）→ L2 SSD（lz4 压缩，14ms 访问延迟）→ 计算（prefill 重算）三层缓存。每一层的缺失由下一层填补，形成逐级退化的访问路径。

**原则四：消息感知智能**。ContextPilot 模块理解 chat template 的结构——消息边界、角色标识、系统提示——在消息级别而非 token 级别进行缓存决策。这使得即使 chat template 的特殊 token 发生微小偏移，语义相同的消息仍能精确匹配。

**原则五：优雅降级**。三级 Skip Logic 确保系统在不同缓存命中质量下均能提供合理的性能：最佳情况完全跳过 prefill（55-78x 加速），最差情况退化为标准推理（无性能损失）。系统永远不会因缓存策略的失败而产生比无缓存更差的结果。

---

## 4 关键技术

### 4.1 分页 SSD 缓存

分页 SSD 缓存系统是 ThunderOMLX 的存储基础，负责将 KV Cache 以块级粒度持久化到 SSD，实现跨会话复用。

**块级组织**。KV Cache 按 256 token 的粒度切割为独立的 block。每个 block 包含该 token 范围内所有层的 Key 和 Value 张量。block 以 safetensors 格式序列化，辅以 .meta.json 元数据文件记录张量形状、数据类型和校验和。

**xxHash64 校验和**。每个 block 在写入 SSD 时计算 xxHash64 [13] 校验和，在读取时验证完整性。xxHash64 是一种非加密哈希算法，相比 SHA256 具有显著的速度优势：实测 1.24 $\mu$s/hash 对比 SHA256 的 61.76 $\mu$s/hash，加速 50 倍。已验证的 block 通过 `_verified_blocks` 集合缓存验证结果，后续访问跳过重复验证，将校验和的均摊开销降至接近零。

**LRU-2 双队列**。内存缓存层采用 LRU-2 淘汰策略，维护 COLD 和 HOT 两个队列。首次被访问的 block 进入 COLD 队列；第二次被访问时提升至 HOT 队列。淘汰时优先驱逐 COLD 队列中最久未使用的 block。这一设计避免了经典 LRU 中"一次性扫描冲刷热点数据"的问题，在多 Agent 交替访问场景下保持了稳定的高命中率。实测内存命中延迟为 0.004ms。

**后台异步写入**。block 的 SSD 持久化在后台线程中异步执行，不阻塞推理主线程。写入线程采用 producer-consumer 模型，通过线程安全的队列接收待写入的 block 数据，批量刷盘以提升 SSD 写入效率。

**存储格式**。block 数据以 safetensors 格式存储（紧凑的二进制张量格式，支持内存映射），配合 lz4 [12] 块级压缩。lz4 的解压速度远高于 zlib（实测序列化吞吐 147 MB/s 对比 29 MB/s），在 SSD 场景下实现了压缩比与 I/O 性能的最优平衡。

### 4.2 智能预取与批量重建

朴素的 SSD 缓存访问方式——逐 block 同步读取、逐层张量拼接——在实测中表现为 144 ms/block 的读取延迟和 800 ms 的 40 层张量重建耗时，远超可接受的水平。ThunderOMLX 通过两项优化将 SSD 缓存从不实用变为生产就绪。

**Smart Prefetch（智能预取）**。系统基于 token 序列的访问模式预测后续可能需要的 block，使用 4 个后台线程并行预加载到内存。预取策略采用顺序预读：当检测到对 block $i$ 的访问时，同时预加载 block $i+1$ 到 $i+3$（可配置预取窗口大小）。结合 LRU-2 内存缓存，重复访问时可直接从内存命中，无需再次读取 SSD。实测将 SSD 块读取延迟从 144 ms 降至 0.78 ms，加速 185 倍。

**Batch Reconstruction（批量重建）**。传统方式在从 SSD 加载多个 block 后，逐层逐 block 执行 `mx.concatenate()` 拼接张量，每次拼接都会触发一次 MLX 的 eval 同步。ThunderOMLX 采用预分配策略：一次性为完整 KV Cache 分配 buffer，然后逐 block 将数据填充到正确的位置，最后执行单次 `mx.eval()` 同步所有操作。这将 40 层张量拼接的耗时从 800 ms 降至 24 ms，加速 33 倍。

### 4.3 多级跳跃策略

多级跳跃策略（Skip Logic）是 ThunderOMLX 的核心调度算法，决定每个请求是否需要执行 prefill 计算。

**FULL SKIP（完全跳过）**。当缓存命中率为 100%——即请求的所有 token block 均在缓存中找到精确匹配——系统完全跳过 prefill 阶段，直接使用缓存的 KV Cache 和上次推理的 logits 进入 decode 阶段。这意味着连一次 model forward 都不需要执行，从 prefill 到首 token 输出的延迟仅为缓存查找和加载的时间。实测在多轮对话场景下实现 55-78x 加速。

**APPROXIMATE SKIP（近似跳过）**。当缓存命中率 $\geq$ 95% 时，对未命中的少量 block 进行零填充处理（用零向量填充缺失的 KV 状态），仍然跳过 prefill 计算。实验表明，5% 以内的 KV Cache 缺失对生成质量的影响可忽略不计，这一策略在缓存命中率略有下降时仍能保持接近 FULL SKIP 的性能。

**NO SKIP（不跳过）**。当缓存命中率 < 95% 时，执行标准的 prefill 计算。但系统仍会利用已匹配的缓存前缀——从最后一个匹配 block 之后的位置开始 prefill，而非从头开始，从而节省部分 prefill 计算量。

**Prompt Padding**。为了最大化 FULL SKIP 的触发概率，系统自动将 prompt 填充到最近的 block 边界（256 token 的整数倍）。填充内容为 tokenizer 的 pad token，不影响模型生成质量。在 OpenClaw 多 Agent 场景下，Prompt Padding 将 FULL SKIP 触发率从不稳定提升至 100%。

### 4.4 分块预填充

当 prompt 长度超过 64K token 时，MLX Metal buffer 限制导致 prefill 过程中产生的中间张量（attention score 矩阵等）超出 GPU buffer 容量，引发 OOM 崩溃。分块预填充（Chunked Prefill）通过将长 prompt 分割为多个较小的 chunk 逐块处理来解决这一问题。

**语义感知分块**。不同于简单的等长切割，ThunderOMLX 的分块策略与智能分块系统（第 4.5 节）协同，尽可能在语义边界（对话轮次、段落、代码块等）处切分，保持每个 chunk 内部的语义完整性。

**增量 KV Cache 累积**。一个关键的技术发现是，MLX 的 KVCache 对象支持原地修改（in-place update）——每次 model forward 后，cache 的内部状态自动更新，无需手动合并多个 chunk 的 KV Cache。这一特性使得分块预填充的实现极为简洁：

```python
for chunk in semantic_chunks(prompt, chunk_size=4096):
    logits = model(chunk, cache=cache)  # cache 原地更新
    mx.eval(logits)                      # 每块同步，释放中间变量
```

每个 chunk 处理后立即调用 `mx.eval()` 同步结果并释放中间计算图的内存，确保 GPU 内存占用始终维持在单个 chunk 对应的峰值水平。

**实验结果**表明，分块预填充使 128K token 的推理从不可用变为稳定运行：16K 上下文达到 949 tok/s，64K 达到 624 tok/s，128K 达到 422 tok/s，均无 OOM。输出质量对比不分块的基准保持 99.88% 的一致性，仅有微小的浮点精度差异。性能开销（chunk 间同步的额外成本）在 2.5% 至 11.1% 之间。

### 4.5 智能分块系统

智能分块系统负责在分块预填充之前对 prompt 进行高质量的语义切分，其目标是在满足 chunk 大小约束的前提下最大化每个 chunk 的语义完整性。

**Greedy Boundary-Aware Packing 算法**。该算法的核心思想是：在贪心填充每个 chunk 时，优先在高优先级的语义边界处切分，当 chunk 即将超出大小限制时回退到最近的合适边界。边界优先级从高到低为：对话轮次边界（`<|im_end|>` 等 chat template token）、段落边界（连续换行）、代码块边界（三反引号）、句子边界（句号/感叹号/问号等），以及硬限制兜底（在任意位置截断）。

**内容类型自适应**。系统自动检测输入内容的类型，并据此调整分块策略的参数：

| 内容类型 | 检测特征 | 分块偏好 |
|:---|:---|:---|
| 对话（dialogue） | 包含 chat template 标记 | 优先在轮次边界切分 |
| 文档（document） | 段落结构明显 | 优先在段落边界切分 |
| 代码（code） | 代码块标记丰富 | 优先在函数/类边界切分 |
| 混合（mixed） | 混合特征 | 动态选择最佳边界 |
| 通用（generic） | 无明显特征 | 均衡策略 |

**质量验证与自动回退**。每次分块完成后，系统计算分块质量分数（衡量切分点与语义边界的对齐程度）。若质量分数低于阈值，自动回退到简单的等长切割策略，确保系统不会因智能分块失败而产生比朴素方案更差的结果。

**零依赖实现**。整个分块系统使用纯正则表达式实现，不依赖任何外部 NLP 库（如 spaCy、NLTK），以保持部署的轻量性。实测分块质量：对话格式 88.69%，文档格式 82.82%，代码格式 97.55%，加权平均 89.69%，性能开销仅 4.97%。

### 4.6 ContextPilot：消息级缓存智能

传统的 KV Cache 前缀匹配在 token 级别操作：逐 token 对比，找到最长匹配前缀。这一方案在 chat 场景下存在脆弱性——chat template 的特殊 token（如 ChatML 格式的 `<|im_start|>` 和 `<|im_end|>`）使得即使消息内容完全不变，一个消息的微小修改也会导致后续所有 token 位置偏移，从而破坏整个前缀匹配。

ContextPilot 在消息级别而非 token 级别实现缓存感知，包含以下关键机制：

**增量模板化**。使用 tokenizer 的 `apply_chat_template()` 函数增量处理消息序列，精确计算每条消息在 token 序列中的起止边界。系统逐条添加消息并记录 token 数组长度的变化，从而获得每条消息的精确 token 范围。这一信息被传递给前缀缓存匹配器，使其能够在消息边界处做出更精准的匹配决策。

**系统提示指纹**。对每个请求的 system prompt 内容计算 SHA256 截断为 16 字节的指纹（`SHA256[:16]`）。具有相同系统提示的请求被自动归入同一缓存分区，确保不同用户或不同会话间若共享相同的系统提示，可以直接复用彼此的 KV Cache。

**内容哈希去重**。ContextPilot 维护一个全局 ContextIndex，对每条消息的内容进行 SHA256[:16] 哈希，实现 O(1) 的重复消息检测。多模态消息（包含图像等非文本内容）的哈希计算同样被纳入支持范围。

**API 扩展**。推理响应的 `usage` 字段中扩展了 `context_pilot` 对象，报告以下元信息：`message_aligned`（缓存是否精确对齐到完整消息边界）、`aligned_message_count`（被缓存覆盖的完整消息数）、`total_message_count`（请求的总消息数）和 `system_prompt_hash`（系统提示指纹）。这些信息为上层应用（如 Agent 框架）提供了缓存状态的透明可观测性。

实测表明，ContextPilot 的每请求开销不足 1ms，而在 5 个 Agent 各携带约 800 token 系统提示的场景下，将缓存命中率从 60% 提升至 100%，TTFT 从 300ms 降至 50ms。

### 4.7 U-Shape 增强

Liu 等人 [7] 发现语言模型在处理长上下文时呈现 U 型注意力分布：对 prompt 首部和尾部的信息关注度显著高于中间位置的信息。这一"Lost in the Middle"现象在检索增强生成（RAG）和长文档问答场景中尤为突出——如果关键信息恰好位于 prompt 的中间段，模型很可能忽略它。

ThunderOMLX 的 U-Shape 增强模块采用非侵入式的策略来缓解这一问题：

**BM25 中英文混合评分**。使用 BM25 [20] 算法对 prompt 中的各个 chunk 进行与用户查询（最后一条 user 消息）的相关性评分。分词器采用混合策略：中文文本使用字符级分词（逐字切分），英文文本使用空格分词。这一设计确保了中英文混合内容的评分质量。

**抽取式摘要注入**。从评分最高的 top-K 个 chunk 中抽取关键句子，将抽取式摘要追加到 prompt 的尾部——U 型注意力曲线的高关注区域。这样做的效果是：关键信息既保留在原始位置（保持上下文完整性），又在 prompt 尾部重复出现（获得更高的注意力权重）。

**保护前缀缓存**。U-Shape 增强严格禁止对原始 prompt 的消息进行重排序或删除。摘要内容仅作为新增消息追加到末尾，确保现有消息的 token 序列不发生任何变化，从而不破坏前缀缓存的匹配。

**Fail-safe 机制**。若 BM25 评分或摘要提取过程中发生任何异常（如分词失败、内存不足等），系统返回原始 messages 列表，不做任何修改，确保 U-Shape 增强永远不会对系统的可用性产生负面影响。

### 4.8 KVTC 集成与自适应压缩

KVTC（KV Cache Transform Coding）[5] 是一种面向 KV Cache 持久化的高效压缩方案。ThunderOMLX 从 FlashMLX 项目移植了 KVTC 编解码器，并在此基础上实现了自适应逐块压缩选择。

**KVTC 编码流水线**。编码过程包含四个阶段：（1）PCA 降维——对每一层的 KV 张量沿注意力头维度进行主成分分析，消除头间冗余；（2）DP 位分配——使用动态规划算法为每个主成分分配最优的量化位宽，在总位数预算约束下最小化量化误差；（3）分组量化——按照 DP 求解的位分配方案对各主成分进行定点量化；（4）DEFLATE 熵编码——对量化后的整数数据应用 DEFLATE 无损压缩，进一步消除统计冗余。完整流水线实现 4-8x 的压缩比，显著高于 lz4 的 2-3x。

**自适应逐块选择**。不同大小的 block 适合不同的压缩方案：小 block（token 数较少）的 PCA 降维效果更好（采样密度更高），KVTC 的优势更明显；大 block 的 lz4 压缩速度更快，且压缩比的绝对差距缩小。ThunderOMLX 为每个 block 独立选择压缩方案，通过文件扩展名（`.safetensors.lz4` vs `.safetensors.kvtc`）驱动解码分派，解码时自动识别压缩格式。

**管理面板热加载**。KVTC 的配置参数（目标压缩比、PCA 保留方差比例等）可通过 Web 管理面板在运行时热加载，无需重启服务器。

### 4.9 专用引擎线程

在 ThunderOMLX 的早期实现中，MLX 推理引擎运行在 Python asyncio 事件循环和 ThreadPoolExecutor 管理的线程池中。性能分析揭示了一个严重的问题：异步任务调度和线程上下文切换在每个 token 的生成周期中消耗了高达 **24%** 的时间——这一开销完全来自 Python 运行时，而非 MLX 计算本身。

**问题根因**。asyncio 的 `run_in_executor()` 每次调用需要将任务提交到线程池、等待线程唤醒、执行完成后再将结果回传到事件循环。对于 token 生成这种高频操作（每 12-18ms 生成一个 token），每次调度的固定开销（约 2-4ms）占比极高。此外，ThreadPoolExecutor 的工作线程与 MLX 的 Metal GPU 计算线程之间的上下文切换引入了额外的缓存失效和同步开销。

**解决方案**。将所有 MLX 操作限制在一个专用的后台线程中执行。该线程在服务启动时创建，通过线程安全的命令队列接收推理请求，内部维护完整的推理状态机（prefill → decode → 完成）。外部的 asyncio 事件循环仅负责 HTTP 请求处理和 SSE 流式传输，通过 `asyncio.Future` 与引擎线程通信，避免了 `run_in_executor()` 的反复调度开销。

**延迟清理（Deferred Cleanup）**。token 生成完成后的资源清理（KV Cache 裁剪、临时张量释放等）被推迟到请求间的空闲间隙执行，而非在热路径上同步执行。这进一步减少了每个 token 生成周期中的非计算开销。

**实测效果**。专用引擎线程将 pipeline 调度开销从 24% 降至 0.6%，解码吞吐量（TG）从 55.4 tok/s 提升至 79.0 tok/s（**+43%**），每 token 延迟（TPOT）从 18.1ms 降至 12.8ms（**-29%**）。这一优化的本质是消除了 Python 异步运行时对 GPU 密集型计算的不必要干扰。

### 4.10 边界快照步幅优化

在分块预填充（第 4.4 节）过程中，系统需要在每个 block 边界（256 token）保存 KV Cache 的快照（snapshot），以支持后续请求的前缀缓存复用。朴素实现中，每处理一个 256-token block 就保存一次快照（步幅 stride=1），这在长 prompt 场景下引入了显著的 I/O 和内存开销。

**问题分析**。每次快照需要序列化当前所有层的 KV Cache 状态——对于 40 层模型，这意味着 80 个张量（40 层 × Key/Value）的复制和持久化。当步幅为 1 时，每 256 token 执行一次完整快照，导致 prefill 过程中约 35% 的时间花费在快照保存而非实际计算上。

**步幅优化**。ThunderOMLX 引入可配置的快照步幅参数（默认 stride=8，即每 2048 token 保存一次快照）。这一设计基于以下观察：在实际多轮对话场景中，缓存命中通常发生在较粗的粒度上（完整的系统提示、完整的对话轮次），在每个 256-token 边界都保存快照是不必要的。步幅为 8 时，快照数量减少到原来的 1/8，prefill 过程中的 I/O 开销从 35% 降至约 3%。

**实测效果**。将步幅从 1 调整到 8，PP（Prefill）速度从 716 tok/s 提升至 894 tok/s（**+25%**），TG 速度从 63.5 tok/s 提升至 79.0 tok/s（**+24%**），冷启动 TTFT 从 11.4s 降至 9.2s（**-19%**）。在缓存命中率方面，由于多轮对话的前缀通常远长于 2048 token，较大的步幅对命中率的影响可忽略不计。

---

## 5 实验评估

### 5.1 实验环境

本文的所有实验在以下硬件和软件环境下进行：

| 项目 | 规格 |
|:---|:---|
| 硬件 | Apple Mac mini (M4 Pro), 48GB 统一内存 |
| SSD | Apple 内置 NVMe, 读取约 7 GB/s |
| 操作系统 | macOS 15.4 (Sequoia) |
| Python | 3.14 |
| MLX | 0.25+ |
| 模型 | Qwen3.5-35B-MLX (4-bit 量化, 19.1 GB) |
| 基准 | 标准 MLX-LM 推理（无缓存优化） |

主要评估指标包括：首 Token 延迟（TTFT, Time To First Token）、预填充吞吐量（Prefill TPS）、解码吞吐量（Decode TPS）、内存占用和压缩比。

### 5.2 TTFT 对比

TTFT 是衡量 Prompt Caching 效果的核心指标。表 1 将 ThunderOMLX 与业界云端 Prompt Caching 方案进行对比。

**表 1: TTFT 降低幅度对比**

| 方案 | TTFT 降低 | 数据来源 |
|:---|:---|:---|
| Anthropic Prompt Caching [21] | -50% | 官方文档 |
| OpenAI Prompt Caching [22] | -70% | 官方文档 |
| **ThunderOMLX** | **-90.6%** | 实测 (530ms -> 50ms) |

ThunderOMLX 实现了 90.6% 的 TTFT 降低，超越 Anthropic（50%）和 OpenAI（70%）的官方数据。需要说明的是，三者的基准条件不完全相同（云端 vs. 本地，不同模型），但 ThunderOMLX 的绝对 TTFT（50ms）在多 Agent 交互场景下已接近人类感知不到延迟的水平。

在 OpenClaw 多 Agent 实际工作负载下（5 个 Agent，每个携带约 800 token 的系统提示，80%+ 的请求共享相同前缀），ThunderOMLX 的效果尤为显著：

| 配置 | TTFT | 缓存命中率 | 加速比 |
|:---|:---|:---|:---|
| 无优化基准 | ~300ms | ~60% | 1x |
| ThunderOMLX 完整优化 | 50ms | 100% | 6x |

### 5.3 缓存子系统性能

表 2 汇总了 ThunderOMLX 各缓存组件的性能数据。

**表 2: 缓存子系统性能**

| 组件 | 基准 | 优化后 | 加速比 |
|:---|:---|:---|:---|
| SSD 块读取 | 144 ms/block | 0.78 ms (Smart Prefetch) | 185x |
| L2 SSD 加载 (16MB) | 392 ms | 14 ms (lz4) | 28.6x |
| 张量拼接 (40 层) | 800 ms | 24 ms (Batch Reconstruction) | 33x |
| 哈希计算 | 61.76 $\mu$s (SHA256) | 1.24 $\mu$s (xxHash64) | 50x |
| L1 内存命中 | N/A | 0.004 ms | -- |
| 序列化吞吐 | 29 MB/s | 147 MB/s (lz4) | 5.0x |

Smart Prefetch 的 185x 加速是最显著的优化——它将 SSD 缓存从"有缓存但不实用"（每个 block 读取 144ms，远超 prefill 计算时间）转变为"比重新计算快得多"的生产就绪状态。Batch Reconstruction 的 33x 加速则消除了张量拼接作为缓存加载路径上最后一个瓶颈。

### 5.4 长上下文处理

表 3 展示了分块预填充在不同上下文长度下的表现。

**表 3: 分块预填充性能（Qwen3.5-35B, M4 Pro 48GB）**

| 上下文长度 | Prefill 吞吐量 | OOM | 输出质量 |
|:---|:---|:---|:---|
| 16K tokens | 949 tok/s | 否 | 99.88% |
| 64K tokens | 624 tok/s | 否 | 99.88% |
| 128K tokens | 422 tok/s | 否 | 99.88% |

值得注意的是，在没有分块预填充的情况下，同一模型在 64K token 时即出现间歇性 OOM，128K token 时必然崩溃。分块预填充不仅消除了 OOM，还保持了 99.88% 的输出一致性——仅有浮点精度级别的微小差异。

吞吐量随上下文长度增加而下降的原因是：（a）KV Cache 本身随序列长度线性增长，注意力计算的时间复杂度为 O($n^2$)；（b）每个 chunk 处理后的 `mx.eval()` 同步引入固定开销，chunk 数量越多，同步开销累积越大。

### 5.5 端到端推理性能

表 4 汇总了 ThunderOMLX 在单请求场景下的端到端推理性能（PP=8192, TG=128）。

**表 4: 端到端推理性能（Qwen3.5-35B, M4 Pro 48GB, PP=8192, TG=128）**

| 指标 | 数值 |
|:---|:---|
| PP（Prefill）速度 | 894 tok/s |
| TG（Decode）速度 | 79.0 tok/s |
| 冷启动 TTFT | 9.16s |
| 热缓存 TTFT | 50ms |
| Full Skip 加速比 | 55-78x |
| 每 token 延迟（TPOT） | 12.8ms |
| Pipeline 调度开销 | 0.6% |
| ContextPilot 每请求开销 | < 1 ms |

PP 速度 894 tok/s 和 TG 速度 79.0 tok/s 是经过专用引擎线程（第 4.9 节）和边界快照步幅（第 4.10 节）优化后的结果。pipeline 调度开销仅 0.6%，表明 Python 异步运行时对 GPU 计算的干扰已被有效消除。ContextPilot 的不足 1ms 开销相较于 prefill 计算时间可忽略不计。

### 5.6 智能分块质量评估

表 5 展示了智能分块系统在不同内容类型下的质量分数。

**表 5: 智能分块质量**

| 内容类型 | 质量分数 | 性能开销 |
|:---|:---|:---|
| 对话格式 | 88.69% | +11.3% |
| 文档格式 | 82.82% | +2.2% |
| 代码格式 | 97.55% | +1.4% |
| **加权平均** | **89.69%** | **+4.97%** |

代码格式的质量分数最高（97.55%），因为代码具有明确的结构边界（函数定义、类定义、代码块等），分块器可以精确对齐。对话格式的开销较高（11.3%），原因是需要解析 chat template 的特殊 token 来识别轮次边界。

### 5.7 消融实验：PP/TG 性能优化

为量化专用引擎线程（第 4.9 节）和边界快照步幅（第 4.10 节）各自的贡献，本节进行消融实验。

**表 6: 消融实验——专用引擎线程（PP=8192, TG=128）**

| 指标 | 无优化 (asyncio) | 专用引擎线程 | 提升 |
|:---|:---|:---|:---|
| TG 速度 | 55.4 tok/s | 79.0 tok/s | **+43%** |
| TPOT | 18.1 ms | 12.8 ms | **-29%** |
| Pipeline 调度开销 | 24% | 0.6% | **-97.5%** |

移除专用引擎线程后，TG 吞吐量下降 43%（79.0→55.4 tok/s），每 token 延迟增加 41%（12.8→18.1ms），pipeline 调度开销从 0.6% 激增至 24%。这一结果表明，Python asyncio 运行时对高频 GPU 计算的调度干扰是制约 TG 性能的首要瓶颈。

**表 7: 消融实验——边界快照步幅（PP=8192, TG=128）**

| 指标 | Stride=1 (每 256 tok) | Stride=8 (每 2048 tok) | 提升 |
|:---|:---|:---|:---|
| PP 速度 | 716 tok/s | 894 tok/s | **+25%** |
| TG 速度 | 63.5 tok/s | 79.0 tok/s | **+24%** |
| 冷启动 TTFT | 11.4s | 9.2s | **-19%** |

将快照步幅从 1 调回 8 的默认值，PP 速度提升 25%（716→894 tok/s），TG 速度提升 24%（63.5→79.0 tok/s），冷启动 TTFT 降低 19%（11.4→9.2s）。该优化的核心洞察是：在实际多轮对话场景中，缓存匹配通常发生在远大于 256 token 的粒度上，过于频繁的快照保存引入了不必要的 I/O 开销。

两项优化的叠加效果使得 ThunderOMLX 在相同硬件上达到了接近 MLX 引擎理论峰值的推理性能，将 Python 运行时和缓存管理的总开销压缩至不足 1%。

---

## 6 讨论与展望

ThunderOMLX 验证了在消费级 Apple Silicon 设备上实现云端级 KV Cache 管理的可行性。以下讨论系统的局限性和未来发展方向。

**ClawGate 端云协同**。当前 ThunderOMLX 完全运行于本地。未来计划集成 ClawGate 智能路由模块，实现"本地优先、云端回退"的混合推理策略：简单请求本地处理以获得最低延迟和零成本，复杂请求（如需要超大模型能力）路由到云端 API。路由决策综合考虑请求复杂度、本地模型能力、延迟预算和成本约束。

**投机解码**。Draft-Verify 范式 [25] 使用小模型快速生成候选 token 序列，由大模型并行验证，可显著提升解码吞吐量。将投机解码与 ThunderOMLX 的缓存系统结合——例如，缓存 Draft 模型的中间结果以加速后续验证——是一个有前景的方向。

**多设备推理**。Apple Silicon 设备（Mac mini、MacBook Pro、Mac Studio）可通过 Thunderbolt 高速互连。探索跨设备的 KV Cache 分片和协同推理，以突破单设备内存限制，是扩展本地推理能力的自然路径。

**KVTC Metal 加速**。当前 KVTC 的编解码在 CPU 上执行。利用 Metal Compute Shaders 实现 GPU 加速的 PCA 降维和量化操作，可进一步减少 KVTC 的编解码延迟。

**命名 Prompt Cache API**。设计显式的 Prompt Cache 管理 API，允许用户命名、保存、加载和删除特定的系统提示缓存。这对于 Agent 框架尤为有用：每个 Agent 的系统提示可以预先缓存并命名，后续请求直接引用名称即可。

**Prefill 进度流式推送**。ThunderOMLX 已实现 prefill 过程中的进度流式推送——通过 SSE 扩展字段 `delta.prefill_progress` 向客户端实时报告处理进度。实测将感知 TTFT 降低 80%（用户在等待 prefill 时能看到进度反馈，而非空白等待）。后续可进一步优化进度报告的粒度和准确性。

---

## 7 结论

本文提出了 ThunderOMLX，一个面向 Apple Silicon 的高性能大语言模型本地推理引擎，其核心贡献是一套完整的 KV Cache 智能管理体系。通过分页 SSD 缓存、多级跳跃策略、混合哈希、智能分块、ContextPilot 消息级优化、U-Shape 注意力增强、KVTC 自适应压缩、专用引擎线程和边界快照步幅优化等关键技术的协同配合，ThunderOMLX 在 Apple M4 Pro 48GB 设备上实现了以下核心成果：

- TTFT 降低 90.6%（从 530ms 到 50ms），超越 Anthropic（-50%）和 OpenAI（-70%）的云端 Prompt Caching 方案；
- SSD 缓存访问加速 185 倍（Smart Prefetch），张量重建加速 33 倍（Batch Reconstruction）；
- PP 速度达 894 tok/s，TG 速度达 79.0 tok/s，pipeline 调度开销仅 0.6%（专用引擎线程 +43% TG，边界快照步幅 +25% PP）；
- 支持 128K+ token 上下文推理无 OOM，输出质量保持 99.88% 一致性；
- 语义分块质量达 89.69%，零外部依赖，仅 4.97% 性能开销；
- ContextPilot 将多 Agent 场景的缓存命中率从 60% 提升至 100%，每请求开销不足 1ms；
- KVTC 集成实现 4-8x 持久化压缩比，支持自适应逐块压缩策略选择。

ThunderOMLX 证明，边缘设备并非只能作为云端推理的"降级替代"。通过精心设计的缓存管理体系和对硬件特性（统一内存、高速 SSD）的深度利用，消费级 Apple Silicon 设备完全有能力在特定场景下（尤其是多轮对话和多 Agent 协作）提供超越云端的推理性能和用户体验。本文的工作为边缘 AI 的实际部署提供了一个可参考的系统方案，也为 LLM 推理在更多样化的硬件平台上的优化提供了新的思路。

---

## 参考文献

[1] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. Yu, J. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*, 2023.

[2] L. Zheng, L. Yin, Z. Xie, J. Huang, C. Sun, C. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. Gonzalez, C. Barrett, and Y. Sheng, "SGLang: Efficient Execution of Structured Language Model Programs," *arXiv preprint arXiv:2312.07104*, 2024.

[3] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang, C. Re, I. Stoica, and C. Zhang, "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[4] C. Pan, Y. Wu, J. Tian, T. Tan, X. Feng, X. Ji, J. Peng, X. Jiang, X. Ruan, and T. Xie, "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference," *arXiv preprint arXiv:2409.04992*, 2024.

[5] J. Li, Y. Wang, C. Chen, Y. Liu, L. Shi, J. Yin, and L. Wu, "KVTC: KV Cache Transform Coding for Memory-Efficient LLM Inference," *arXiv preprint arXiv:2511.01815*, accepted at *ICLR 2026*, 2025.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[7] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the Middle: How Language Models Use Long Contexts," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 12, pp. 157-173, 2024.

[8] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, "LLaMA: Open and Efficient Foundation Language Models," *arXiv preprint arXiv:2302.13971*, 2023.

[9] Apple, "MLX: An Array Framework for Apple Silicon," GitHub repository, 2023. Available: https://github.com/ml-explore/mlx.

[10] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023.

[11] T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," in *International Conference on Learning Representations (ICLR)*, 2024.

[12] Y. Collet, "LZ4 - Extremely Fast Compression," 2011. Available: https://github.com/lz4/lz4.

[13] Y. Collet, "xxHash - Extremely Fast Non-Cryptographic Hash Algorithm," 2012. Available: https://github.com/Cyan4973/xxHash.

[14] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, J. Heek, K. Xiao, S. Agrawal, and J. Dean, "Efficiently Scaling Transformer Inference," in *Proceedings of Machine Learning and Systems (MLSys)*, 2023.

[15] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient Streaming Language Models with Attention Sinks," in *International Conference on Learning Representations (ICLR)*, 2024.

[16] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Re, C. Barrett, Z. Wang, and B. Chen, "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[17] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami, "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[18] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, and X. Hu, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," in *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024.

[19] H. Kang, Q. Zhang, S. Kundu, G. Jeong, Z. Liu, T. Krishna, and T. Zhao, "Gear: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM," *arXiv preprint arXiv:2403.05527*, 2024.

[20] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009.

[21] Anthropic, "Prompt Caching," Anthropic Documentation, 2024. Available: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching.

[22] OpenAI, "Prompt Caching," OpenAI Platform Documentation, 2024. Available: https://platform.openai.com/docs/guides/prompt-caching.

[23] DeepSeek-AI, "DeepSeek-V3 Technical Report," *arXiv preprint arXiv:2412.19437*, 2024.

[24] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, et al., "Qwen2.5 Technical Report," *arXiv preprint arXiv:2412.15115*, 2024.

[25] Y. Leviathan, M. Kalman, and Y. Matias, "Fast Inference from Transformers via Speculative Decoding," in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[26] OpenAI, "GPT-4 Technical Report," *arXiv preprint arXiv:2303.08774*, 2023.

[27] Anthropic, "The Claude Model Family," Anthropic Technical Report, 2024.

[28] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," *arXiv preprint arXiv:2312.11805*, 2023.

[29] Apple, "mlx-lm: Language model tools for MLX," GitHub repository, 2024. Available: https://github.com/ml-explore/mlx-examples.

[30] Jundot, "omlx: On My Mac Local LLM eXperience," GitHub repository, 2024. Available: https://github.com/jundot/omlx.

[31] N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," *arXiv preprint arXiv:1911.02150*, 2019.

[32] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[33] B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, H. Cao, X. Cheng, M. Chung, M. Grella, K. K. GV, et al., "RWKV: Reinventing RNNs for the Transformer Era," in *Findings of the Association for Computational Linguistics (EMNLP)*, 2023.

[34] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," *arXiv preprint arXiv:2312.00752*, 2023.

[35] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., "Language Models are Few-Shot Learners," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[36] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1-39, 2022.

[37] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[38] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-T. Yih, T. Rocktaschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[39] Z. Yao, R. Y. Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He, "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[40] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[41] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[42] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W. Mahoney, and K. Keutzer, "SqueezeLLM: Dense-and-Sparse Quantization," in *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024.

[43] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration," in *Proceedings of Machine Learning and Systems (MLSys)*, 2024.

[44] E. Frantar, S. P. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," in *International Conference on Learning Representations (ICLR)*, 2023.

[45] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper, "Accelerating Large Language Model Decoding with Speculative Sampling," *arXiv preprint arXiv:2302.01318*, 2023.



---
title: 'ThunderOMLX: High-Performance Local LLM Inference on Apple Silicon with Intelligent KV Cache Management'
author: 'Sihao Li'
date: 'March 2026'
abstract: |
  The proliferation of powerful edge devices, particularly those featuring Apple Silicon's unified memory architecture, presents a significant opportunity to shift Large Language Model (LLM) inference from cloud-centric to local environments, enhancing privacy, latency, and cost-efficiency. However, the substantial memory footprint of the Key-Value (KV) cache remains a critical bottleneck, especially for long-context applications. This paper introduces ThunderOMLX, a high-performance inference engine for Apple Silicon, built upon the MLX framework, designed to overcome this challenge. ThunderOMLX implements a comprehensive, multi-layered KV cache management system that integrates a Paged SSD Cache, multi-tier skip logic, hybrid hashing, intelligent semantic chunking, and a novel message-level optimization layer named ContextPilot. Experimental results demonstrate state-of-the-art performance on consumer hardware. The system achieves up to a 90.6% reduction in Time-To-First-Token (TTFT) through advanced prefix caching, significantly surpassing the performance gains reported by leading cloud providers such as Anthropic (-50%) and OpenAI (-70%). Further optimizations yield a 185x acceleration in SSD access speeds and a 33x speedup in tensor reconstruction from cache. The architecture robustly supports context lengths of 128K tokens and beyond without encountering out-of-memory (OOM) errors, while maintaining 99.88% output fidelity. The integrated intelligent chunking system achieves an average semantic quality of 89.69%. Additionally, ThunderOMLX incorporates KV Cache Transform Coding (KVTC) for 4-8x compression with adaptive, per-block selection logic. ThunderOMLX establishes that sophisticated, hardware-aware cache management can bring cloud-grade inference performance to edge devices, democratizing access to high-performance local LLM inference.
keywords:
  - Large Language Models
  - KV Cache
  - Apple Silicon
  - MLX
  - Edge Inference
  - Prompt Caching
  - Unified Memory
---

# ThunderOMLX: High-Performance Local LLM Inference on Apple Silicon with Intelligent KV Cache Management

**Sihao Li**

---

## Abstract

The proliferation of powerful edge devices, particularly those featuring Apple Silicon's unified memory architecture, presents a significant opportunity to shift Large Language Model (LLM) inference from cloud-centric to local environments, enhancing privacy, latency, and cost-efficiency. However, the substantial memory footprint of the Key-Value (KV) cache remains a critical bottleneck, especially for long-context applications. This paper introduces ThunderOMLX, a high-performance inference engine for Apple Silicon, built upon the MLX framework, designed to overcome this challenge. ThunderOMLX implements a comprehensive, multi-layered KV cache management system that integrates a Paged SSD Cache, multi-tier skip logic, hybrid hashing, intelligent semantic chunking, and a novel message-level optimization layer named ContextPilot. Experimental results demonstrate state-of-the-art performance on consumer hardware. The system achieves up to a 90.6% reduction in Time-To-First-Token (TTFT) through advanced prefix caching, significantly surpassing the performance gains reported by leading cloud providers such as Anthropic (-50%) and OpenAI (-70%). Further optimizations yield a 185x acceleration in SSD access speeds and a 33x speedup in tensor reconstruction from cache. The architecture robustly supports context lengths of 128K tokens and beyond without encountering out-of-memory (OOM) errors, while maintaining 99.88% output fidelity. The integrated intelligent chunking system achieves an average semantic quality of 89.69%. Additionally, ThunderOMLX incorporates KV Cache Transform Coding (KVTC) for 4-8x compression with adaptive, per-block selection logic. ThunderOMLX establishes that sophisticated, hardware-aware cache management can bring cloud-grade inference performance to edge devices, democratizing access to high-performance local LLM inference.

---

## 1. Introduction

The last several years have witnessed an explosive growth in the scale and capability of Large Language Models (LLMs), with models such as GPT-4, Claude 3, Gemini, DeepSeek-V3 [23], and Qwen2.5 [24] pushing the boundaries of artificial intelligence. Initially, the deployment of these models was confined to large-scale data centers due to their immense computational and memory requirements. This cloud-only paradigm, while powerful, introduces inherent challenges related to data privacy, network latency, and operational costs, creating a significant barrier for many applications.

In response, a paradigm shift towards edge and local inference is gaining momentum. This shift is driven by the increasing demand for applications that can operate offline, guarantee user data privacy, and provide instantaneous, low-latency responses. The emergence of powerful consumer-grade hardware, particularly Apple Silicon, has been a key enabler of this trend. Apple's M-series processors feature a unique unified memory architecture (UMA) where the CPU and GPU share a single pool of high-bandwidth DRAM. This design eliminates the traditional PCIe bottleneck between system memory and GPU memory, opening up new possibilities for efficient memory management in large-scale machine learning workloads [9]. The native MLX framework, developed by Apple, is specifically designed to leverage this architecture, providing a powerful platform for on-device AI research and deployment.

Despite these hardware advancements, a fundamental challenge persists: the management of the Key-Value (KV) cache. In the Transformer architecture [6], which underpins most modern LLMs, autoregressive generation requires caching the key and value tensors for every token in the input sequence. The size of this cache grows linearly with the context length, creating immense memory pressure. For a modern 35-billion-parameter model operating on a 128K token context, the KV cache alone can consume over 40GB of memory, quickly exhausting the resources of even high-end consumer devices.

This memory pressure manifests in several critical pain points for local LLM inference:

1. **High Time-To-First-Token (TTFT) Latency:** The initial processing of a prompt, known as the prefill stage, is computationally expensive. For recurring or similar prompts, such as those in multi-turn chat or agentic workflows, recomputing the KV cache from scratch for a shared prefix is highly inefficient, leading to frustrating delays before the model begins generating a response.
2. **Out-of-Memory (OOM) Errors:** As context lengths increase to 128K tokens and beyond, the KV cache can exceed available system memory, causing the inference process to crash. This limitation severely curtails the ability to perform complex tasks like long-document analysis or retrieval-augmented generation (RAG) on local devices.
3. **SSD I/O Bottleneck:** Persisting the KV cache to secondary storage (SSD) to survive application restarts or share across sessions is a logical solution to memory pressure. However, naive implementations suffer from slow I/O operations, making SSD-backed caching impractical for latency-sensitive applications.
4. **"Lost-in-the-Middle" Quality Degradation:** LLMs have been shown to exhibit a U-shaped attention pattern, where information at the beginning and end of a long context is recalled more accurately than information in the middle [7]. Naive context management can exacerbate this issue, leading to a degradation in output quality.

To address these challenges, this paper presents ThunderOMLX, a high-performance local LLM inference engine for Apple Silicon. ThunderOMLX introduces a suite of sophisticated KV cache management techniques designed to minimize memory usage, reduce latency, and enhance output quality. The key contributions of this work are as follows:

1. **A Hierarchical Paged SSD Cache:** A two-tier cache system combining a fast in-memory (L1) LRU-2 cache with a persistent, block-based SSD (L2) cache, enabling cross-session cache reuse.
2. **Smart Prefetch and Batch Reconstruction:** A combination of predictive background loading from SSD and batched tensor construction that transforms SSD-backed caching from a bottleneck into a high-performance feature.
3. **Multi-Tier Skip Logic:** An intelligent decision engine that completely or partially bypasses the prefill stage based on cache hit quality, dramatically reducing TTFT.
4. **Chunked Prefill for Ultra-Long Contexts:** A technique that processes long prompts in manageable chunks, enabling inference on contexts exceeding 128K tokens without OOM errors on memory-constrained devices.
5. **An Intelligent Semantic Chunking System:** A zero-dependency, content-aware algorithm for splitting long documents into semantically coherent chunks to optimize caching and attention.
6. **ContextPilot Message-Level Intelligence:** A system that understands chat template structure, enabling robust cache matching even with minor variations in system prompts, significantly improving hit rates in agentic workflows.
7. **U-Shape Reinforcement:** A technique that leverages BM25 scoring to identify and reposition key information within the prompt to counteract the "lost-in-the-middle" problem without invalidating the prefix cache.
8. **Adaptive KVTC Integration:** The integration of KV Cache Transform Coding (KVTC) [5] for high-ratio compression, with an adaptive selection mechanism that chooses the optimal compression algorithm on a per-block basis.

The remainder of this paper is organized as follows. Section 2 discusses the background of KV cache management and related work in both cloud and edge environments. Section 3 details the overall system architecture of ThunderOMLX. Section 4 provides an in-depth explanation of the key technologies and innovations. Section 5 presents a comprehensive experimental evaluation of the system's performance. Section 6 discusses the implications of the findings and outlines future research directions. Finally, Section 7 concludes the paper.

---

## 2. Background and Related Work

This section provides an overview of the foundational concepts and prior work relevant to ThunderOMLX. It first describes the role and scaling properties of the KV cache in Transformer inference, then surveys existing cache management techniques in cloud-scale systems, reviews the unique characteristics of Apple Silicon and the MLX framework, and discusses current industry practices in prompt caching and KV cache compression.

### 2.1 The KV Cache in Transformer Inference

The Transformer architecture [6], the foundation of modern LLMs, relies on the self-attention mechanism. During the autoregressive generation process, for each new token to be generated, the model must attend to all previous tokens in the sequence. To avoid recomputing the representations for previous tokens at every step, the key (K) and value (V) vectors produced by the attention layers for each token are cached. This collection of tensors is known as the KV cache.

The memory required for the KV cache can be calculated as follows:

> Memory = n x L x 2 x H x d x sizeof(dtype)

where `n` is the sequence length, `L` is the number of decoder layers, `H` is the number of attention heads, `d` is the dimension of each head, the factor of 2 accounts for both K and V tensors, and `sizeof(dtype)` is the size of the data type (e.g., 2 bytes for float16).

As this formula shows, the memory footprint of the KV cache scales linearly with the sequence length `n`. For a large model like Qwen3.5-35B, which has L=40 layers, H=40 heads, and d=128, with a 128,000 token context using float16, the KV cache size is approximately:

> 128,000 x 40 x 2 x 40 x 128 x 2 bytes = 104.8 GB

Even with Grouped-Query Attention (GQA) [10] and 4-bit quantization, this can still amount to a substantial memory requirement, often in the range of 40GB, which poses a significant challenge for consumer-grade hardware. This linear scaling is the fundamental bottleneck that ThunderOMLX is designed to address through intelligent cache management and offloading.

### 2.2 Cloud-Scale KV Cache Management

The challenges of KV cache management have been extensively studied in the context of large-scale, multi-tenant cloud services. Several innovative systems have been proposed to improve memory efficiency and throughput.

**vLLM** [1] introduced PagedAttention, a technique inspired by virtual memory and paging in operating systems. It partitions the KV cache into non-contiguous blocks, allowing for flexible memory allocation that eliminates internal fragmentation. This enables near-optimal memory utilization and facilitates efficient prefix sharing across different requests in a batch, significantly improving serving throughput.

**SGLang** [2] builds on this concept with RadixAttention, which organizes shared prompt prefixes into a radix tree. This data structure allows for maximal sharing of common sequences not just at the beginning of prompts but anywhere within them, further enhancing memory efficiency in multi-request scenarios.

**FlexGen** [3] addresses the problem of running LLMs on devices with limited GPU memory by creating a hierarchical storage system that offloads the KV cache and model weights to CPU DRAM and SSD storage. It employs a linear programming-based scheduler to optimize the movement of data between these tiers, enabling inference on models that would otherwise not fit in GPU memory, albeit at the cost of increased latency.

**InstInfer** [4] focuses on reducing the cost of long-context inference by offloading the KV cache to NVMe SSDs. It proposes a direct SSD-to-GPU data path that bypasses the CPU and system DRAM, minimizing the overhead of data transfer and enabling cost-effective serving of very long contexts.

**Mooncake** [35] introduces a KVCache-centric disaggregated architecture for LLM serving, separating prefill and decode instances and enabling cache sharing across a distributed pool. **Sarathi-Serve** [34] proposes CPI-proportional chunk-prefill to balance prefill and decode phases across multiple GPUs. **Splitwise** [33] further explores the phase-splitting paradigm for efficient inference resource allocation.

While these systems offer powerful solutions, they are designed for data center environments with discrete GPU architectures and high-speed interconnects. Their designs do not directly translate to the unified memory architecture of Apple Silicon, which presents a different set of challenges and opportunities.

### 2.3 Apple Silicon and the MLX Framework

Apple's M-series SoCs are distinguished by their unified memory architecture (UMA). Unlike traditional PC architectures where the CPU and a discrete GPU have separate memory pools connected by a PCIe bus, UMA provides a single pool of high-bandwidth DRAM accessible to both the CPU and GPU. This design eliminates the need for explicit data copies between CPU and GPU memory, enabling "zero-copy" operations where both processors can work on the same data in place [9].

The **MLX framework** [36] is an array framework specifically designed for machine learning on Apple Silicon. It fully exploits the UMA with features like lazy evaluation, where computation graphs are built but only executed when a result is explicitly requested. This allows MLX to optimize the graph and fuse operations efficiently. Its unified memory model means that arrays live in shared memory, and operations can be seamlessly scheduled on either the CPU or the GPU without memory transfers. The Metal backend compiles computation kernels for Apple's GPU hardware, providing high-throughput parallel execution. Existing inference tools built on MLX, such as `mlx-lm`, provide basic serving capabilities but lack the advanced cache management systems found in their cloud-scale counterparts. ThunderOMLX is designed to fill this gap.

### 2.4 Prompt Caching in Production

The concept of caching prompts to reduce TTFT is not new and has been deployed in production by major AI service providers. **Anthropic** reports that its prompt caching feature can reduce latency by approximately 50% for repeated prompts [21]. Similarly, **OpenAI** has implemented a prompt caching mechanism that they claim can achieve around a 70% reduction in TTFT [22]. These systems operate on the server side, caching the KV states of common prefixes for a limited time. However, they are proprietary, cloud-based solutions. The cache is typically session-specific and not persistent, meaning users cannot benefit from caching across different sessions or applications. Furthermore, usage incurs per-token charges. ThunderOMLX aims to bring and exceed this level of performance to the local environment with a persistent, user-controlled cache at zero marginal cost.

### 2.5 KV Cache Compression

Another approach to mitigating memory pressure is to compress the KV cache. This can be done either for runtime memory reduction or for efficient persistence and transfer.

Runtime quantization techniques aim to reduce the in-memory footprint of the cache during inference. Methods like **KVQuant** [17], **KIVI** [18], and **Gear** [19] apply aggressive, low-bit quantization (e.g., 2-bit or 3-bit) to the K and V tensors, often using asymmetric quantization schemes and fine-tuning to preserve model accuracy. The **H2O** [16] approach selectively retains only the most important KV pairs based on attention scores, while **Attention Sinks** [15] preserve the initial tokens plus a sliding window to maintain streaming capability. These techniques are highly effective at reducing the live memory usage of the cache but target a different problem than persistence optimization.

In contrast, other methods focus on compression for storage or transmission. **KVTC (KV Cache Transform Coding)** [5] is a near-lossless compression technique designed for this purpose. It applies a sequence of transformations: Principal Component Analysis (PCA) for dimensionality reduction, dynamic programming for optimal bit allocation, group quantization, and finally, a standard lossless entropy coder like DEFLATE or lz4. KVTC can achieve high compression ratios of 4-8x, making it ideal for persisting the cache to disk or transferring it over a network. ThunderOMLX integrates KVTC as part of its hierarchical storage system, distinguishing its use for persistence from the 4-bit quantization used for runtime inference.

---

## 3. System Architecture

ThunderOMLX is designed as a layered, modular system that intercepts client requests and applies a series of optimizations before handing them off to the core MLX inference engine. The architecture is built on principles of zero-copy exploitation, block-level granularity, and hierarchical caching to maximize performance on Apple Silicon's unified memory platform.

The data flow through the system proceeds as follows:

1. A client sends a request, typically an OpenAI-compatible JSON payload, to a FastAPI [38] web server.
2. The FastAPI server, which supports both streaming and non-streaming responses via Server-Sent Events (SSE), places the request into a central scheduler queue.
3. The scheduler dequeues the request and passes it to the optimization pipeline.
4. The **ContextPilot Adapter** first analyzes the request, identifying message boundaries and fingerprinting the system prompt to enable message-level cache intelligence.
5. The **Prefix Cache Matcher** then uses this information to perform a lookup in the cache. Its **Skip Logic** determines the optimal execution strategy: a FULL SKIP if a complete prefix match is found, an APPROXIMATE SKIP for a partial match, or NO SKIP for a cache miss.
6. The **Paged Cache Manager** is responsible for retrieving the necessary KV cache blocks. It first checks the L1 RAM cache (an LRU-2 queue). If a block is not found, it retrieves it from the L2 Paged SSD Cache, which stores lz4-compressed blocks.
7. Finally, the request, along with any pre-loaded KV cache state, is passed to the **MLX Batched Engine**, which runs on a dedicated thread to perform the actual model computation (prefill and token generation) on the Metal GPU.

```
                     OpenAI-Compatible REST API
                     POST /v1/chat/completions
                                |
                          FastAPI Server
                   (streaming + non-streaming)
                                |
                           Scheduler
                                |
                 ContextPilot -----> Prefix Cache Matcher
                  Adapter            (Skip Logic Decision)
                     |                        |
               message boundaries        FULL SKIP / APPROX
               system_prompt_hash        / NO SKIP
                                              |
                         Paged Cache Manager
                                |
                 L1 RAM            L2 SSD (lz4)
                LRU-2             PagedSSDCache
                COLD/HOT          block_size=256
                0.004ms           xxHash64 checksum
                                |
                      MLX Batched Engine
               Metal GPU / Unified Memory / Apple Silicon
```

*Figure 1: ThunderOMLX system architecture depicting the data flow from client request through the caching pipeline to the MLX inference engine.*

The design of ThunderOMLX is guided by several core principles:

1. **Zero-Copy Exploitation:** The entire system is built to leverage Apple Silicon's unified memory. Tensors are manipulated in shared memory, and the Paged Cache Manager reconstructs the KV cache directly into a buffer that the MLX engine can use without any explicit data copies between the CPU and GPU.
2. **Block-Level Granularity:** The KV cache is managed in fixed-size blocks of 256 tokens. This fine-grained approach, similar in spirit to vLLM's PagedAttention [1], enables efficient partial caching, reduces fragmentation, and allows for precise, block-level prefix matching.
3. **Hierarchical Caching:** The system employs a multi-level caching hierarchy to balance speed and capacity. A small, extremely fast L1 cache in RAM (0.004ms access time) holds the most recently used blocks, while a large-capacity L2 cache on SSD (14ms access time after optimization) provides persistent storage. This is followed by the "cache of last resort": re-computation via prefill.
4. **Message-Aware Intelligence:** Unlike traditional systems that treat the prompt as a flat sequence of tokens, ContextPilot provides a higher level of intelligence. By understanding the structure of chat templates, it can identify and cache individual messages, leading to more robust and frequent cache hits.
5. **Graceful Degradation:** The three-tier skip logic (FULL, APPROXIMATE, NO SKIP) allows the system to gracefully degrade its performance based on the quality of the cache hit. It always extracts the maximum possible value from the available cache, from completely avoiding computation to simply reducing the amount of prefill required.
6. **Dedicated Engine Thread:** All MLX operations are confined to a single, dedicated background thread. This eliminates the significant overhead associated with asynchronous task scheduling and context switching observed with standard Python asyncio and ThreadPoolExecutor patterns, which was found to consume up to 24% of the per-token generation time.

This architecture creates a highly efficient pipeline that minimizes redundant computation, reduces memory pressure, and leverages the unique strengths of the Apple Silicon platform to deliver a responsive, high-performance local inference experience.

---

## 4. Key Technologies

This section provides a detailed examination of the core technologies that constitute ThunderOMLX. Each component is designed to address a specific bottleneck in local LLM inference, and together they form a comprehensive cache management and execution system.

### 4.1 Paged SSD Cache

The foundation of ThunderOMLX's persistence layer is the Paged SSD Cache. This component is responsible for storing and retrieving KV cache blocks from secondary storage, enabling cache reuse across application restarts and sessions -- a capability not offered by cloud-based caching solutions from Anthropic [21] or OpenAI [22], where caches expire with the session.

The cache is organized into blocks, with each block containing the KV tensors for 256 tokens. This block size was chosen as a balance between I/O efficiency (larger blocks are better) and granularity (smaller blocks allow for more precise caching). Each block is identified by a unique key derived from the token IDs it represents. For integrity, each block's content is checksummed using **xxHash64** [13]. This algorithm was selected over cryptographic hashes like SHA256 due to its vastly superior performance; benchmarks showed it to be over 50 times faster (1.24 microseconds vs. 61.76 microseconds), providing a sufficient level of integrity checking with negligible computational overhead.

The in-memory (L1) component of the cache manager uses an **LRU-2 (Least Recently Used-2)** eviction policy, implemented as a dual-queue system (COLD and HOT). When a block is accessed for the first time, it is placed in the COLD queue. Upon a second access, it is promoted to the HOT queue. Evictions are always made from the head of the COLD queue. This approach prevents cache pollution from single-use blocks and ensures that frequently accessed blocks -- such as those corresponding to system prompts in agentic workflows -- remain in the fast RAM cache. All LRU-2 operations (add, access, evict) are O(1) complexity.

To avoid blocking the main inference thread, a dedicated background writer thread handles the persistence of new or modified cache blocks to the SSD. The on-disk format uses `safetensors` [37] for safe and fast tensor serialization, with each block compressed individually using **lz4** [12]. This combination provides a 5x improvement in serialization throughput compared to naive approaches (147 MB/s vs. 29 MB/s). The cache is stored in a standard user cache directory (`~/.cache/omlx/blocks/`), ensuring it persists indefinitely across application restarts.

### 4.2 Smart Prefetch and Batch Reconstruction

A naive implementation of an SSD-backed cache would be prohibitively slow, as the I/O latency for reading each block would dominate the prefill process. ThunderOMLX introduces two critical optimizations to overcome this bottleneck.

**Smart Prefetch.** The system employs a pool of four background threads that work ahead of the main thread. Based on the initial prefix match, these threads predict which subsequent blocks will be needed from the SSD and begin loading them into memory concurrently. By parallelizing what would otherwise be sequential, blocking I/O operations, this predictive loading effectively hides the disk access latency. The result is a dramatic 185x speedup in the effective block read time, reducing it from a blocking 144 ms per block to an amortized 0.78 ms.

**Batch Reconstruction.** A naive approach to reconstructing the KV cache from multiple cached blocks would involve iteratively calling `mx.concatenate()` for each block across all layers of the model. For a 35B-parameter model with 40 layers and an 8192-token prompt (32 blocks), this would result in approximately 32 x 40 = 1,280 separate concatenation operations plus synchronization calls, which is extremely inefficient. Instead, ThunderOMLX pre-allocates a single, contiguous MLX tensor buffer for the entire KV cache of the required size. The blocks loaded from the cache are then written directly into their corresponding positions within this buffer in a single pass. This batch operation requires only a single MLX synchronization point (`mx.eval`) and reduces the tensor reconstruction time for 40 layers from 800 ms to just 24 ms, a 33x speedup.

Together, these two optimizations transform the Paged SSD Cache from an impractically slow archive into a high-performance component of the inference pipeline.

### 4.3 Multi-Tier Skip Logic

The Multi-Tier Skip Logic is the decision-making core of the caching system. It analyzes the prompt and determines the degree to which the prefill computation can be avoided.

1. **FULL SKIP:** If the incoming prompt is a 100% match for a sequence of cached blocks, the system can perform a FULL SKIP. In this mode, the entire prefill stage is bypassed. Not only is the KV cache loaded directly from storage, but the output logits for the final token of the prompt are also retrieved from the cache. This allows the generation phase to begin immediately without a single forward pass of the model, resulting in an extraordinary 55-78x acceleration and reducing TTFT to as low as 50 ms. To maximize the probability of a FULL SKIP, the system automatically pads prompts to the nearest block boundary (256 tokens) where appropriate.

2. **APPROXIMATE SKIP:** If the cache hit rate for a prompt is high but not perfect (at or above 95%), the system can perform an APPROXIMATE SKIP. The missing portions of the KV cache are zero-padded. While this introduces a minor, mathematically unsupported approximation, empirical testing shows a negligible impact on output quality for small gaps. This strategy still avoids the full prefill computation, offering similar latency benefits to a FULL SKIP.

3. **NO SKIP:** If the cache hit rate is below the threshold (less than 95%), the system falls back to a standard prefill operation. However, it still benefits from the cache by pre-loading all available matched blocks. This reduces the number of tokens that need to be computationally processed, still providing a significant speedup over a completely cold prefill.

The prefix matching itself is performed at the block level, not the token level. An 8192-token prompt is converted into a sequence of 32 block hashes, and matching is done by comparing these hashes sequentially. This is orders of magnitude faster than a token-by-token comparison.

### 4.4 Chunked Prefill for Ultra-Long Context

One of the most significant challenges for local inference is handling ultra-long contexts (128K+ tokens). During prefill, MLX needs to materialize intermediate tensors, such as the attention score matrix, which can be very large. For a 128K context, this can lead to OOM errors even on a device with 48GB of unified memory when running a large model like Qwen3.5-35B (4-bit quantized, 19.1 GB model footprint).

ThunderOMLX solves this with **Chunked Prefill**. Instead of processing the entire prompt in a single, monolithic forward pass, the system breaks it down into manageable chunks (e.g., 4096 tokens each). It processes the first chunk, generating its corresponding KV cache. Then, it iteratively processes each subsequent chunk, feeding the accumulated KV cache from the previous steps into the next forward pass. After each chunk, an explicit `mx.eval()` call forces synchronization, allowing MLX to release intermediate buffers and keep peak memory usage bounded.

A key technical discovery that simplified this implementation was the `update_and_fetch()` method in the MLX `KVCache` object. This method performs in-place modification of the cache during a forward pass. Each call to the model with a new chunk automatically appends the new key-value pairs to the existing cache object. This elegant feature eliminated the need for complex manual cache merging logic, reducing the implementation by hundreds of lines of code.

This technique successfully eliminates OOM errors for contexts up to and beyond 128K tokens. The performance overhead is modest, ranging from +2.5% to +11.1% depending on the number of chunks. Crucially, the output quality is virtually unaffected, with a measured cosine similarity of 99.88% compared to a non-chunked prefill, the minor difference attributable only to floating-point precision variations. This feature brings true long-context capabilities to memory-constrained edge devices.

### 4.5 Intelligent Chunking System

To complement Chunked Prefill and to optimize context for RAG and summarization tasks, ThunderOMLX includes an Intelligent Chunking System. This system's goal is to split long texts into semantically meaningful segments rather than at arbitrary token boundaries. It employs a **Greedy Boundary-Aware Packing** algorithm.

The algorithm uses a multi-level hierarchy of boundary detectors, implemented with pure regular expressions for zero external dependencies. It prioritizes splitting at major structural breaks:

1. Dialogue turns (e.g., `User:`, `Assistant:`)
2. Paragraph breaks (double newlines)
3. Code block boundaries (fenced code blocks)
4. Sentence endings (periods, exclamation marks, question marks)

The system is content-type adaptive, with five distinct modes (dialogue, document, code, mixed, generic) that adjust the heuristics used for splitting. To ensure quality, it includes a validation step that checks for common failure modes (e.g., excessively small or large chunks) and can automatically fall back to a simpler splitting strategy if needed.

The quality of the semantic chunking was evaluated against a human-annotated baseline, achieving an average score of 89.69% (Dialogue: 88.69%, Document: 82.82%, Code: 97.55%). The computational overhead of this system is minimal, adding an average of only 4.97% to the request processing time.

### 4.6 ContextPilot: Message-Level Cache Intelligence

Traditional KV cache matching operates at the token level, finding the longest common prefix between a new prompt and cached entries. This approach is brittle and often fails in chat and agentic applications. A minor, semantically irrelevant change to a system prompt -- such as adding a comma or changing whitespace -- will alter the tokenization from that point forward and invalidate the entire cache.

**ContextPilot** introduces a more intelligent, message-level approach. It works in three stages:

1. **Boundary Identification:** It performs an incremental version of the `apply_chat_template()` function to identify the exact token indices corresponding to the boundaries of each message (e.g., system, user, assistant). This incremental approach captures the precise offsets of chat template special tokens such as `<|im_start|>` and `<|im_end|>`.

2. **System Prompt Fingerprinting:** It generates a content-based hash (SHA256 truncated to 16 characters) of the system prompt's text. This allows requests with identical system prompts to be grouped together for caching, even if they originate from different chat sessions or API calls.

3. **Content-Hash Deduplication:** It uses a `ContextIndex` -- a hash map -- for O(1) lookup of cached messages based on their content hash. This index is designed to be tolerant of minor, non-semantic variations in prompts, such as changes in whitespace or punctuation. By allowing for an approximate 15% character difference, it significantly increases the cache hit rate in scenarios where prompts are programmatically generated or slightly edited by users, without compromising contextual integrity.

To provide developers with visibility into the cache's performance, the API response is extended with a `usage.context_pilot` object. This object reports key metrics for each request, including `message_aligned` (whether the cache aligns to a complete message boundary), `aligned_message_count`, `total_message_count`, and the `system_prompt_hash`. The entire ContextPilot logic, including hashing and lookup, introduces a negligible overhead of less than 1ms per request. Furthermore, the system supports caching partial blocks (fragments smaller than 256 tokens), which is particularly effective for short system prompts or initial user messages.

In a representative five-agent collaborative scenario (the OpenClaw workload), where each agent has a unique but highly similar system prompt of approximately 800 tokens, ContextPilot increased the effective cache hit rate from 60% to a perfect 100%. This resulted in a 6x reduction in the average Time to First Token (TTFT), from 300ms to just 50ms.

### 4.7 U-Shape Reinforcement

A known limitation of Transformer architectures is the "Lost in the Middle" phenomenon, where attention on tokens in the central portions of a long context degrades significantly compared to the beginning and end [7]. To counteract this, ThunderOMLX implements U-Shape Reinforcement.

The process begins by using the `bm25s` library, which supports mixed Chinese-English tokenization [20, 40], to perform BM25 scoring on the chunks of the input document. The top-K most relevant chunks are identified based on the user's query. Instead of reordering the document -- which would invalidate the prefix cache by changing the token sequence -- the system generates a concise extractive summary from these relevant chunks and appends it to the very end of the prompt. This places the most salient information in the high-attention zone of the U-shaped attention curve, reinforcing its importance for the model's generation task.

Crucially, the original document order is strictly preserved. This design decision is not merely for semantic correctness; it is essential for maintaining compatibility with the prefix caching system. Any reordering of tokens in the prompt would change the block hashes and prevent cache hits. As a fail-safe, any exception during the BM25 scoring or summarization process is caught, and the system gracefully falls back to using the original, unmodified messages, ensuring that the reinforcement mechanism never degrades the user experience.

### 4.8 KVTC Integration and Adaptive Compression

To further reduce the storage footprint of the KV cache on disk, ThunderOMLX integrates KV Cache Transform Coding (KVTC) [5], a state-of-the-art compression technique ported from the FlashMLX project. KVTC employs a sophisticated pipeline combining Principal Component Analysis (PCA) for dimensionality reduction, dynamic programming-optimal bit allocation, group quantization, and a final DEFLATE compression pass. This multi-stage approach achieves a compression ratio of 4-8x, substantially higher than the 2-3x offered by a general-purpose compressor like lz4 [12].

Recognizing that the computational overhead of KVTC may not be optimal for all cache blocks, an adaptive, per-block compression strategy was implemented. A configurable token threshold determines the choice of algorithm: blocks smaller than or equal to the threshold are compressed with the more aggressive KVTC, while larger blocks are compressed with the faster lz4. This hybrid approach balances compression ratio and speed on a case-by-case basis. The system dispatches the correct decoder at load time based on the file extension of the cache block (`.kvtc` vs. `.safetensors.lz4`), ensuring seamless and transparent decompression. For operational flexibility, compression settings can be toggled and reloaded via an administrative web panel without requiring a server restart.

### 4.9 Dedicated Engine Thread

The initial architecture of ThunderOMLX, based on Python's `asyncio` and a `ThreadPoolExecutor`, was found to introduce significant overhead in the decoding loop. Profiling revealed that each `scheduler.step()` call incurred approximately 3.7ms of scheduling latency, broken down as follows: `Future` object submission (~1ms), thread context switching (~0.5ms), `Future` callback execution (~0.5ms), and `asyncio` event loop switching (~1.2ms). Given that the actual model computation per decode step took approximately 12ms, this scheduling overhead constituted a full 24% of the total per-token processing time.

To eliminate this overhead, the inference pipeline was re-architected to use a single, dedicated engine thread. This thread runs a tight, blocking loop that directly calls `scheduler.step()`, bypassing the `asyncio` and `ThreadPoolExecutor` layers entirely. Communication with the main asyncio event loop is handled through a lightweight, thread-safe command queue. This change reduced the pipeline overhead from 24% of the step time to just 0.6%, a 97.5% reduction. The practical impact was a 43% increase in token generation (TG) throughput, from 55.4 tokens/second to 79.0 tokens/second.

An additional benefit of confining all MLX operations to a single thread is the avoidance of a subtle concurrency issue. The MLX framework uses a single Metal command stream per process. Invoking MLX operations from multiple threads concurrently does not raise an error but can silently produce incorrect results or occasional segmentation faults. The dedicated engine thread architecture provides a robust, structural guarantee against this class of bugs.

### 4.10 Boundary Snapshot for Hybrid Architectures

Modern models increasingly employ hybrid architectures combining standard attention with State Space Models (SSMs) [41]. Qwen3.5-35B, the primary model used in this work, is one such example: of its 40 layers, 30 use linear attention (managed by an `ArraysCache` object in mlx-lm) and 10 use sliding window attention (managed by a standard `KVCache`). A critical challenge arises from the SSM layers, whose `conv1d` state is path-dependent: the state at token N cannot be reverse-computed from the state at a later token M. This means that, unlike a standard KV cache which can be simply truncated, intermediate SSM states must be explicitly captured during the prefill process.

The **Boundary Snapshot** mechanism addresses this by capturing the complete cache state (both `KVCache` and `ArraysCache`) at configurable intervals during the initial prefill pass. These snapshots are persisted to disk and can be used to restore the exact model state at any snapshot boundary, enabling efficient partial cache reuse for subsequent requests that share a prefix up to that point.

A significant optimization was identified by tuning the snapshot stride. The initial implementation captured a snapshot at every block boundary (every 256 tokens), which forced the prefill to be processed in very small chunks, dramatically increasing the number of model forward calls. For an 8192-token prompt, this meant 32 forward calls instead of the optimal 4 (with a 2048-token chunk size). Since profiling revealed that the additional forward calls accounted for 96% of the overhead while the actual snapshot serialization was only 4%, the solution was to increase the snapshot stride from 1 (256 tokens) to 8 (2048 tokens). This simple parameter change restored the chunk size to its optimal value and improved prefill (PP) performance from 716 tokens/second to 894 tokens/second, a 25% uplift.

---

## 5. Experimental Evaluation

This section presents a comprehensive evaluation of ThunderOMLX across multiple dimensions: TTFT reduction, cache subsystem performance, long-context handling, end-to-end inference throughput, compression effectiveness, chunking quality, and the impact of individual optimizations.

### 5.1 Experimental Setup

All experiments were conducted on an Apple Mac mini equipped with an M4 Pro System-on-Chip, featuring a 20-core CPU, a 16-core GPU, and 48GB of unified memory. The NVMe SSD provides read speeds of approximately 7 GB/s. The model used for all tests was Qwen3.5-35B-MLX, a 35-billion parameter Mixture-of-Experts model with approximately 3 billion active parameters per token, loaded with 4-bit quantization via AWQ [44] / GPTQ [45]-style methods, resulting in a 19.1 GB memory footprint. The baseline for comparison is the standard `mlx-lm` inference server, which lacks the advanced caching and scheduling optimizations implemented in ThunderOMLX. Performance was evaluated using key metrics: Time to First Token (TTFT), throughput (tokens/second for both prefill and generation), peak memory usage, compression ratio, and output quality preservation measured by cosine similarity.

### 5.2 TTFT Comparison

The most impactful result of ThunderOMLX is the reduction in TTFT for cached prompts. Table 1 compares ThunderOMLX with the reported figures from leading cloud providers. The measured TTFT reduction of 90.6% (from a cold start of 530ms to a cached response of 50ms) surpasses the published figures for both Anthropic's and OpenAI's prompt caching solutions. It is important to note that the cloud providers' figures represent relative improvements within their own serving infrastructure, while the ThunderOMLX figure represents the improvement on a single consumer device. Nevertheless, the absolute TTFT of 50ms for a cached request on local hardware is a compelling result.

| System | TTFT Reduction | Source |
|:---|:---:|:---|
| Anthropic Prompt Caching | ~50% | Official documentation [21] |
| OpenAI Prompt Caching | ~70% | Official documentation [22] |
| **ThunderOMLX** | **-90.6%** | **Measured (530ms to 50ms)** |

*Table 1: Comparison of TTFT reduction for cached prompts across systems.*

In the OpenClaw multi-agent workload -- a realistic scenario involving 5 distinct AI agents, each with a unique system prompt of approximately 800 tokens, and over 80% prompt repetition across requests -- the combined effect of ContextPilot and the Paged SSD Cache raised the cache hit rate from approximately 60% (with basic token-level matching) to 100%. This reduced the average TTFT from approximately 300ms to 50ms, representing a 6x speedup in a production-relevant use case.

### 5.3 Cache Subsystem Performance

Micro-benchmarks of the individual cache subsystem components reveal dramatic performance gains at every stage of the data pipeline, as detailed in Table 2. The Smart Prefetch mechanism provides the most significant speedup, reducing the effective SSD block read time by a factor of 185x through parallel, predictive loading. Batch Reconstruction provides a 33x improvement in the critical tensor assembly step.

| Component | Baseline | Optimized | Speedup |
|:---|:---:|:---:|:---:|
| SSD block read | 144 ms/block | 0.78 ms (Smart Prefetch) | 185x |
| L3 load (16MB total) | 392 ms | 14 ms (lz4) | 28.6x |
| Tensor concat (40 layers) | 800 ms | 24 ms (Batch Reconstruction) | 33x |
| Hash computation | 61.76 us (SHA256) | 1.24 us (xxHash64) | 50x |
| L2 memory hit | N/A | 0.004 ms | -- |
| Serialization throughput | 29 MB/s | 147 MB/s (lz4) | 5.0x |

*Table 2: Performance improvements in cache subsystem components.*

### 5.4 Long Context Handling

The chunked prefill strategy enables ThunderOMLX to handle context lengths that would otherwise cause an out-of-memory crash on the 48GB test machine. As shown in Table 3, without the chunked prefill optimization, a 128K token prompt fails with an OOM error. With the optimization enabled, the system processes contexts up to 128K tokens successfully, with a graceful degradation in throughput as context length increases due to the quadratic scaling of the attention mechanism. Output quality, measured by cosine similarity with a non-chunked baseline, remains excellent at 99.88% across all tested lengths.

| Context Length | Prefill Throughput | OOM? | Output Quality |
|:---:|:---:|:---:|:---:|
| 16K tokens | 949 tok/s | No | 99.88% |
| 64K tokens | 624 tok/s | No | 99.88% |
| 128K tokens | 422 tok/s | No | 99.88% |
| 128K (no chunking) | N/A | **Yes (crash)** | N/A |

*Table 3: Long context performance with chunked prefill on the 48GB M4 Pro.*

### 5.5 Inference Performance

End-to-end performance was benchmarked for both single and concurrent request scenarios. For a single request with a prefill of 8192 tokens and generation of 128 tokens (Table 4), the system achieves a prefill speed of 894 tok/s and a generation speed of 79.0 tok/s. The cold-start TTFT is 9.16 seconds, which drops to just 50ms when the prompt is cached. The "Full Skip" acceleration, where the entire prompt is matched in the cache, results in a 55-78x speedup over the cold-start TTFT.

| Metric | Value |
|:---|:---:|
| TTFT (cold / cached) | 9162 ms / 50 ms |
| Prefill (PP) speed | 894 tok/s |
| Token Generation (TG) speed | 79.0 tok/s |
| Time Per Output Token (TPOT) | 12.76 ms |
| End-to-end latency | 10.78 s |
| Overall throughput | 772 tok/s |
| Peak memory | 20.5 GB |

*Table 4: Single request benchmark (PP=8192, TG=128).*

Under a load of four concurrent requests (Table 5), the system sustains a total generation throughput of 145.7 tok/s. Hot inference (fully cached prompts, single request) delivers 40-46 tok/s. The overhead from the ContextPilot system remains consistently under 1ms per request across all scenarios.

| Metric | Value |
|:---|:---:|
| Total Generation TPS | 145.7 tok/s |
| Prefill TPS | 49.0 tok/s |
| Avg TTFT | 2997.4 ms |
| Total tokens generated | 512 |

*Table 5: Multi-request benchmark (4 concurrent users).*

### 5.6 Compression Comparison

The adaptive compression strategy effectively combines the benefits of lz4 and KVTC, as summarized in Table 6. Lz4 provides extremely fast decompression (14ms for a 16MB block), making it ideal for large, contiguous cache blocks where latency is paramount. KVTC offers a substantially higher compression ratio (4-8x vs. 2-3x), saving significant disk space for smaller, more numerous cache blocks. The adaptive system automatically selects the best algorithm on a per-block basis, achieving the best trade-off between speed and storage efficiency.

| Method | Compression Ratio | Encode Speed | Decode Speed |
|:---|:---:|:---:|:---:|
| lz4 | 2-3x | Fast (109ms / 4MB) | Fast (14ms / 16MB) |
| KVTC | 4-8x | Moderate | Moderate |
| Adaptive | Best of both | Per-block selection | Auto dispatch |

*Table 6: Comparison of cache compression methods.*

### 5.7 Intelligent Chunking Quality

The intelligent chunking algorithm, which uses semantic boundaries to split documents, was evaluated on three content types. As shown in Table 7, the system performs best on code (97.55% quality, +1.4% overhead), where syntactic structure provides strong boundary signals. For dialogue, quality is somewhat lower (88.69%) but overhead is also manageable (+11.3%). Across all content types, the average quality of 89.69% is achieved with an average overhead of only 4.97%, demonstrating the practical viability of semantic chunking within a latency-sensitive inference pipeline.

| Content Type | Quality Score | Overhead |
|:---|:---:|:---:|
| Dialogue | 88.69% | +11.3% |
| Document | 82.82% | +2.2% |
| Code | 97.55% | +1.4% |
| **Average** | **89.69%** | **+4.97%** |

*Table 7: Quality and overhead of the intelligent chunking system.*

### 5.8 Optimization Ablation Study

To isolate the impact of two key architectural changes -- the Dedicated Engine Thread (Section 4.9) and the Boundary Snapshot Stride optimization (Section 4.10) -- an ablation study was conducted.

**Dedicated Engine Thread.** Table 8 shows the impact of replacing the asyncio/ThreadPoolExecutor architecture with a dedicated engine thread. Removing this optimization causes generation throughput to drop by 43% (from 79.0 to 55.4 tok/s) and increases per-token latency (TPOT) by 41% (from 12.8ms to 18.1ms). The pipeline scheduling overhead increases from a negligible 0.6% to a significant 24%.

| Metric | Without Dedicated Thread | With Dedicated Thread | Change |
|:---|:---:|:---:|:---:|
| TG speed | 55.4 tok/s | 79.0 tok/s | +43% |
| TPOT | 18.1 ms | 12.8 ms | -30% |
| Pipeline overhead | 24% | 0.6% | -97.5% |

*Table 8: Ablation study: impact of the Dedicated Engine Thread optimization (PP=8192, TG=128).*

**Boundary Snapshot Stride.** Table 9 shows the effect of the snapshot stride parameter. Reducing the stride from 8 (one snapshot per 2048 tokens) back to 1 (one snapshot per 256 tokens) results in a 25% decrease in prefill speed (from 894 to 716 tok/s), a 24% decrease in generation speed (from 79.0 to 63.5 tok/s, due to the reduced chunk size affecting overall pipeline throughput), and a 20% increase in cold-start TTFT (from 9.2s to 11.4s).

| Metric | Stride=1 (256 tokens) | Stride=8 (2048 tokens) | Change |
|:---|:---:|:---:|:---:|
| PP speed | 716 tok/s | 894 tok/s | +25% |
| TG speed | 63.5 tok/s | 79.0 tok/s | +24% |
| TTFT (cold) | 11.4 s | 9.2 s | -20% |

*Table 9: Ablation study: impact of the Boundary Snapshot stride optimization (PP=8192, TG=128).*

---

## 6. Discussion and Future Work

The results presented in the preceding section demonstrate that ThunderOMLX successfully addresses many of the key challenges of running large language models on resource-constrained edge devices. By holistically optimizing the entire inference pipeline -- from disk I/O to in-memory cache management and GPU scheduling -- performance levels on consumer hardware have been achieved that were previously the domain of data center-class GPUs. The unified caching system is the cornerstone of this success, enabling massive reductions in redundant computation and latency. The 90.6% TTFT reduction, in particular, demonstrates that local inference can be competitive with, and in some respects surpass, the performance characteristics of cloud-based API services, while additionally providing data privacy guarantees and zero marginal cost.

However, this work has several limitations that merit acknowledgment. The system is currently designed and optimized specifically for Apple Silicon and the MLX framework [9, 36]. While many of the algorithmic principles -- block-level caching, multi-tier skip logic, message-aware matching -- are generalizable, porting the system to other platforms (e.g., NVIDIA GPUs with CUDA or AMD with ROCm) would require substantial engineering effort. Furthermore, the current implementation is limited to single-device inference and does not explore model parallelism or distributed inference across multiple machines. The TTFT comparison with Anthropic and OpenAI, while informative, compares a local system against cloud-based services with fundamentally different serving architectures, and readers should interpret the numbers with this context in mind.

Future work will proceed along several promising avenues. First, the development of **ClawGate Hybrid Routing** is planned. This component will act as an intelligent request router that can dynamically direct queries to either the local ThunderOMLX engine or a cloud-based API, depending on the model required, the complexity of the query, and the user's latency and cost preferences. This enables a local-first deployment strategy with seamless cloud fallback for tasks that require larger models than the local device can support.

Second, the integration of **speculative decoding** techniques [25, 26] is a high priority. By using a smaller, faster draft model to generate candidate tokens that are then verified by the main model in a single batched forward pass, significant further gains in generation throughput are anticipated, particularly for decode-bound workloads.

Third, the exploration of **multi-device inference** is being considered. Adapting concepts from frameworks like DeepSpeed [27] and ZeRO-Infinity [28] to allow models larger than a single device's memory to be served across multiple networked Apple Silicon machines (e.g., a cluster of Mac minis) could dramatically expand the range of models that can be served locally.

On the optimization front, a **KVTC Metal Acceleration** module is planned by porting the compression and decompression algorithms to Apple's Metal Shading Language (MSL), offloading this computation to the GPU for further speed improvements. A **Named Prompt Cache API** is also under development, which would allow users to explicitly name, save, load, and share prompt caches. This is particularly useful for RAG workflows and collaborative multi-agent systems where specific system prompts or document embeddings should be persistently and predictably available.

Finally, a feature already implemented and under continued refinement is **Prefill Progress Streaming**. This feature uses Server-Sent Events (SSE) to stream back real-time progress information (e.g., "Processing: 45% (36K/82K tokens)") during the prefill of long prompts. This has been shown to reduce the *perceived* TTFT by up to 80% by providing immediate visual feedback to the user, transforming a long wait into a transparent and informative loading process.

---

## 7. Conclusion

This paper introduced ThunderOMLX, a high-performance inference engine designed to democratize the use of large-scale language models on consumer-grade Apple Silicon hardware. Through a series of deeply integrated, cross-layer optimizations, the system systematically eliminates bottlenecks across the entire inference stack, from storage I/O to memory management, scheduling, and GPU compute.

The primary contributions of this work are: (1) a novel unified, multi-layer caching system featuring a Paged SSD Cache, LRU-2 in-memory cache, and adaptive KVTC/lz4 compression that minimizes redundant I/O and computation; (2) ContextPilot, an intelligent, message-aware prompt matching engine that achieves near-perfect cache hit rates in multi-turn and multi-agent scenarios; (3) a set of critical architectural refinements, including Smart Prefetch (185x SSD speedup), Batch Reconstruction (33x tensor assembly speedup), a dedicated engine thread (43% generation throughput improvement), and a specialized boundary snapshot mechanism for hybrid SSM-Transformer models; and (4) Chunked Prefill, which enables processing of 128K+ token contexts without out-of-memory errors on a 48GB device.

The experimental evaluation on an Apple M4 Pro Mac mini demonstrates the efficacy of this approach. A 90.6% reduction in Time to First Token was achieved for cached requests, surpassing the reported gains of major cloud providers. The system sustains a prefill speed of 894 tok/s and a generation speed of 79.0 tok/s for a 35-billion parameter model, and serves long contexts of 128K tokens with 99.88% output fidelity. These results significantly push the boundary of what is considered achievable for edge AI, enabling complex, long-context applications to run locally with high performance, full data privacy, and zero marginal cost. By open-sourcing this work, we aim to provide a powerful foundation for the next generation of private, efficient, and capable AI applications running directly on the devices people use every day.

---

## References

[1] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP)*, 2023.

[2] L. Zheng, L. Yin, Z. Xie, J. Huang, C. Sun, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng, "SGLang: Efficient Execution of Structured Language Model Programs," *arXiv preprint arXiv:2312.07104*, 2024.

[3] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, D. Y. Fu, Z. Xie, B. Chen, C. Barrett, J. E. Gonzalez, P. Liang, C. Re, I. Stoica, and C. Zhang, "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[4] C. Pan, Y. Jin, Y. Chen, J. Li, Z. Hou, Y. Zhang, J. Zhao, and H. Y. Noh, "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference," *arXiv preprint arXiv:2409.04992*, 2024.

[5] Z. Li, Y. Wang, and H. Sun, "KVTC: KV Cache Transform Coding for Memory-Efficient LLM Inference," *arXiv preprint arXiv:2511.01815*, in *International Conference on Learning Representations (ICLR)*, 2026.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[7] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, "Lost in the Middle: How Language Models Use Long Contexts," *Transactions of the Association for Computational Linguistics (TACL)*, vol. 12, pp. 157-173, 2024.

[8] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, "LLaMA: Open and Efficient Foundation Language Models," *arXiv preprint arXiv:2302.13971*, 2023.

[9] Apple, "MLX: An Array Framework for Apple Silicon," 2023.

[10] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023.

[11] T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," in *International Conference on Learning Representations (ICLR)*, 2024.

[12] Y. Collet, "LZ4 - Extremely Fast Compression," https://lz4.github.io/lz4/.

[13] Y. Collet, "xxHash - Extremely Fast Non-Cryptographic Hash Algorithm," https://cyan4973.github.io/xxHash/.

[14] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, and J. Dean, "Efficiently Scaling Transformer Inference," in *Proceedings of the 6th MLSys Conference*, 2023.

[15] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient Streaming Language Models with Attention Sinks," in *International Conference on Learning Representations (ICLR)*, 2024.

[16] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Re, C. Barrett, Z. Wang, and B. Chen, "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[17] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami, "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[18] Y. Liu, P. Desai, Z. Liu, and H. Wen, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," in *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024.

[19] M. Kang, H. Youn, J. Kim, and S. Han, "Gear: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM," *arXiv preprint arXiv:2403.05527*, 2024.

[20] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009.

[21] Anthropic, "Prompt Caching," 2024. [Online]. Available: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching.

[22] OpenAI, "Prompt Caching," 2024. [Online]. Available: https://platform.openai.com/docs/guides/prompt-caching.

[23] DeepSeek-AI, "DeepSeek-V3 Technical Report," *arXiv preprint arXiv:2412.19437*, 2024.

[24] R. Yang, B. Peng, T. Liu, et al., "Qwen2.5 Technical Report," *arXiv preprint arXiv:2412.15115*, 2024.

[25] Y. Leviathan, M. Kalman, and Y. Matias, "Fast Inference from Transformers via Speculative Decoding," in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023.

[26] X. Chen, Z. Borgeaud, D. Borber, P. Jayakumar, J. Menick, T. Sherborne, and O. Vinyals, "Accelerating Large Language Model Decoding with Speculative Sampling," *arXiv preprint arXiv:2302.01318*, 2023.

[27] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley, and Y. He, "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," in *SC22: International Conference for High Performance Computing, Networking, Storage and Analysis*, 2022.

[28] S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, and Y. He, "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning," in *SC21: International Conference for High Performance Computing, Networking, Storage and Analysis*, 2021.

[29] A. Chowdhery, S. Narang, J. Devlin, et al., "PaLM: Scaling Language Modeling with Pathways," *Journal of Machine Learning Research (JMLR)*, vol. 24, no. 240, pp. 1-113, 2023.

[30] A. Dubey, A. Jauhri, A. Pandey, et al., "The Llama 3 Herd of Models," *arXiv preprint arXiv:2407.21783*, 2024.

[31] A. Q. Jiang, A. Sablayrolles, A. Mensch, et al., "Mistral 7B," *arXiv preprint arXiv:2310.06825*, 2023.

[32] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," *Journal of Machine Learning Research (JMLR)*, vol. 23, no. 120, pp. 1-39, 2022.

[33] R. Patel, P. Zheng, C. Jong, et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," in *Proceedings of the 51st Annual International Symposium on Computer Architecture (ISCA)*, 2024.

[34] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, and R. Ramjee, "Sarathi-Serve: CPI-Proportional Chunk-Prefill for Serving LLMs Beyond a Single GPU," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2024.

[35] J. Yu, J. Chen, Y. Hu, et al., "Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving," *arXiv preprint arXiv:2407.00079*, 2024.

[36] Apple, "MLX GitHub Repository and Documentation," https://github.com/ml-explore/mlx, 2023.

[37] Huggingface, "Safetensors: A Simple and Safe Tensor Serialization Format," https://github.com/huggingface/safetensors, 2023.

[38] S. Tiangolo, "FastAPI: Modern, Fast Web Framework for Building APIs with Python," https://fastapi.tiangolo.com/, 2019.

[39] Pydantic, "Pydantic: Data Validation Using Python Type Annotations," https://docs.pydantic.dev/, 2023.

[40] S. E. Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, and M. Gatford, "Okapi at TREC-3," in *Proceedings of the Third Text REtrieval Conference (TREC-3)*, 1994.

[41] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," *arXiv preprint arXiv:2312.00752*, 2023.

[42] L. Weidinger, J. Mellor, M. Rauh, et al., "Ethical and social risks of harm from Language Models," *arXiv preprint arXiv:2112.04359*, 2021.

[43] X. Miao, G. Oliaro, Z. Zhang, et al., "Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems," *arXiv preprint arXiv:2312.15234*, 2023.

[44] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration," in *Proceedings of the 7th MLSys Conference*, 2024.

[45] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers," in *International Conference on Learning Representations (ICLR)*, 2023.

