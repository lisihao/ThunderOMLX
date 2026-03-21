---
title: "ThunderOMLX: 基于 Apple Silicon 的高性能本地大语言模型推理引擎——智能 KV Cache 管理系统"
author: "昊哥"
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

### 5.5 推理性能总览

表 4 汇总了 ThunderOMLX 在各场景下的推理性能指标。

**表 4: 推理性能指标（Qwen3.5-35B, M4 Pro 48GB）**

| 指标 | 数值 |
|:---|:---|
| 热推理速度 | 40-46 tok/s |
| Prefill TPS | ~2900 tok/s |
| Decode TPS | ~240 tok/s |
| ContextPilot 每请求开销 | < 1 ms |
| Full Skip 加速比 | 55-78x |
| Approximate Skip 阈值 | $\geq$ 95% 缓存命中 |

Prefill 和 Decode 的吞吐量与底层 MLX 引擎的原生性能接近，表明 ThunderOMLX 的缓存管理层在热路径上引入的开销极小。ContextPilot 的不足 1ms 开销相较于 prefill 计算时间（通常数十至数百毫秒）可忽略不计。

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

本文提出了 ThunderOMLX，一个面向 Apple Silicon 的高性能大语言模型本地推理引擎，其核心贡献是一套完整的 KV Cache 智能管理体系。通过分页 SSD 缓存、多级跳跃策略、混合哈希、智能分块、ContextPilot 消息级优化、U-Shape 注意力增强和 KVTC 自适应压缩等关键技术的协同配合，ThunderOMLX 在 Apple M4 Pro 48GB 设备上实现了以下核心成果：

- TTFT 降低 90.6%（从 530ms 到 50ms），超越 Anthropic（-50%）和 OpenAI（-70%）的云端 Prompt Caching 方案；
- SSD 缓存访问加速 185 倍（Smart Prefetch），张量重建加速 33 倍（Batch Reconstruction）；
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
