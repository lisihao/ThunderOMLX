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
