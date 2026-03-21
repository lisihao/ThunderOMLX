import type {
  ContextEngine,
  ThunderOmlxConfig,
  AssembleResult,
  IngestResult,
  CompactResult,
  BootstrapResult,
  AgentMessage,
  OptimizeResponse,
} from "./types.js";

const DEFAULT_CONFIG: ThunderOmlxConfig = {
  endpoint: "http://127.0.0.1:8082",
  timeout: 800,
  fallbackThreshold: 3,
  features: ["prefix_align", "semantic_dedup"],
};

/**
 * ThunderOMLX ContextEngine for OpenClaw.
 *
 * Optimizes context assembly by calling ThunderOMLX's /v1/context/optimize
 * endpoint to reorder messages for maximum cloud prompt cache alignment.
 *
 * Features:
 * - prefix_align: Reorders messages to maximize static prefix for
 *   Anthropic/Google/OpenAI prompt caching
 * - semantic_dedup: Deduplicates semantically similar context blocks
 *
 * Fallback: On ThunderOMLX unavailability, transparently passes through
 * original messages with no modification.
 */
export class ThunderContextEngine implements ContextEngine {
  readonly info = {
    id: "thunder-omlx",
    name: "ThunderOMLX Context Optimizer",
    version: "1.0.0",
    ownsCompaction: true,
  };

  private config: ThunderOmlxConfig;
  private consecutiveFailures = 0;
  private disabled = false;
  /** Pending compact tasks: sessionId → taskId */
  private pendingCompactTasks = new Map<string, string>();

  constructor(config?: Partial<ThunderOmlxConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Bootstrap: Check ThunderOMLX availability.
   */
  async bootstrap(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
  }): Promise<BootstrapResult> {
    try {
      const res = await this.fetchWithTimeout("/health", { method: "GET" });
      if (res.ok) {
        this.consecutiveFailures = 0;
        this.disabled = false;
        return { bootstrapped: true };
      }
      return { bootstrapped: false, reason: `Health check failed: HTTP ${res.status}` };
    } catch (e) {
      return {
        bootstrapped: false,
        reason: `ThunderOMLX unreachable: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  }

  /**
   * Ingest: Pure local operation, zero HTTP overhead.
   * Messages are stored by OpenClaw's runtime and passed to assemble().
   */
  async ingest(params: {
    sessionId: string;
    sessionKey?: string;
    message: AgentMessage;
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    // No-op: OpenClaw runtime manages message storage.
    // We process everything in assemble() to minimize HTTP roundtrips.
    return { ingested: true };
  }

  /**
   * Assemble: Core method. Calls ThunderOMLX to optimize message ordering
   * for maximum cloud prompt cache hit rate.
   */
  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    tokenBudget?: number;
  }): Promise<AssembleResult> {
    const { messages, tokenBudget } = params;

    // If disabled due to repeated failures, use fallback
    if (this.disabled) {
      return this.fallbackAssemble(messages);
    }

    try {
      const res = await this.fetchWithTimeout("/v1/context/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages,
          token_budget: tokenBudget,
          features: this.config.features,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data: OptimizeResponse = await res.json();
      this.consecutiveFailures = 0;

      return {
        messages: data.optimized_messages,
        estimatedTokens: data.estimated_tokens,
        systemPromptAddition: data.cache_hint
          ? `[thunder-omlx:prefix=${data.prefix_hash},dedup=${data.dedup_count}]`
          : undefined,
      };
    } catch (e) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= this.config.fallbackThreshold) {
        this.disabled = true;
        console.warn(
          `[thunder-omlx] Disabled after ${this.consecutiveFailures} failures. Using fallback.`
        );
      }
      return this.fallbackAssemble(messages);
    }
  }

  /**
   * Compact: Async local-model summarization via ThunderOMLX.
   *
   * Uses fire-and-forget pattern:
   * - First call: submits compaction job, returns compacted=false
   * - Subsequent calls: polls for completion, returns compacted=true when done
   */
  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
  }): Promise<CompactResult> {
    if (this.disabled) {
      return { ok: true, compacted: false, reason: "ThunderOMLX disabled" };
    }

    const { sessionId, tokenBudget } = params;
    const existingTaskId = this.pendingCompactTasks.get(sessionId);

    // If we have a pending task, poll for its status
    if (existingTaskId) {
      return this.pollCompactTask(sessionId, existingTaskId);
    }

    // No pending task — submit a new one
    return this.submitCompactTask(sessionId, tokenBudget);
  }

  private async submitCompactTask(
    sessionId: string,
    tokenBudget?: number
  ): Promise<CompactResult> {
    try {
      const res = await this.fetchWithTimeout("/v1/context/compact/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [], // Will be populated by OpenClaw runtime via assemble()
          token_budget: tokenBudget || 2000,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json() as { task_id: string; status: string };
      this.pendingCompactTasks.set(sessionId, data.task_id);

      return {
        ok: true,
        compacted: false,
        reason: `Compaction submitted (task=${data.task_id})`,
      };
    } catch (e) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= this.config.fallbackThreshold) {
        this.disabled = true;
      }
      return {
        ok: true,
        compacted: false,
        reason: `Submit failed: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  }

  private async pollCompactTask(
    sessionId: string,
    taskId: string
  ): Promise<CompactResult> {
    try {
      const res = await this.fetchWithTimeout(
        `/v1/context/compact/${taskId}`,
        { method: "GET" }
      );

      if (!res.ok) {
        this.pendingCompactTasks.delete(sessionId);
        return { ok: false, compacted: false, reason: `Poll failed: HTTP ${res.status}` };
      }

      const data = await res.json() as {
        task_id: string;
        status: string;
        result?: {
          summary: string;
          compacted_messages: Array<{ role: string; content: string }>;
          tokens_before: number;
          tokens_after: number;
          model_used: string;
        };
        error?: string;
      };

      if (data.status === "done" && data.result) {
        this.pendingCompactTasks.delete(sessionId);
        this.consecutiveFailures = 0;
        return {
          ok: true,
          compacted: true,
          result: {
            summary: data.result.summary,
            tokensBefore: data.result.tokens_before,
            tokensAfter: data.result.tokens_after,
          },
        };
      }

      if (data.status === "failed") {
        this.pendingCompactTasks.delete(sessionId);
        return {
          ok: false,
          compacted: false,
          reason: `Compaction failed: ${data.error || "unknown"}`,
        };
      }

      // Still running
      return {
        ok: true,
        compacted: false,
        reason: `Compaction in progress (task=${taskId}, status=${data.status})`,
      };
    } catch (e) {
      this.pendingCompactTasks.delete(sessionId);
      return {
        ok: true,
        compacted: false,
        reason: `Poll error: ${e instanceof Error ? e.message : String(e)}`,
      };
    }
  }

  /**
   * AfterTurn: Check if background compaction completed.
   * If a compact task finished, the runtime can use the result on next assemble.
   */
  async afterTurn?(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
  }): Promise<void> {
    // No-op for now. The compact polling happens in compact() itself.
    // Phase 3 could proactively trigger compaction when token count is high.
  }

  /**
   * Dispose: Reset state.
   */
  async dispose(): Promise<void> {
    this.consecutiveFailures = 0;
    this.disabled = false;
    this.pendingCompactTasks.clear();
  }

  // --- Private helpers ---

  private fallbackAssemble(messages: AgentMessage[]): AssembleResult {
    // Rough token estimate: ~4 chars per token
    const totalChars = messages.reduce((sum, m) => {
      const content =
        typeof m.content === "string"
          ? m.content
          : (m.content || [])
              .filter((p) => p.type === "text")
              .map((p) => p.text || "")
              .join(" ");
      return sum + content.length;
    }, 0);

    return {
      messages,
      estimatedTokens: Math.max(Math.ceil(totalChars / 3), 1),
    };
  }

  private async fetchWithTimeout(
    path: string,
    init: RequestInit
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const url = `${this.config.endpoint}${path}`;
      const headers: Record<string, string> = {
        ...(init.headers as Record<string, string>),
      };
      if (this.config.apiKey) {
        headers["Authorization"] = `Bearer ${this.config.apiKey}`;
      }

      return await fetch(url, {
        ...init,
        headers,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

/**
 * Factory function for OpenClaw plugin registration.
 *
 * Usage in openclaw.json:
 * ```json
 * {
 *   "plugins": {
 *     "slots": { "contextEngine": "@omlx/thunder-context" },
 *     "entries": {
 *       "@omlx/thunder-context": {
 *         "enabled": true,
 *         "options": {
 *           "endpoint": "http://127.0.0.1:8082",
 *           "timeout": 800,
 *           "features": ["prefix_align", "semantic_dedup"]
 *         }
 *       }
 *     }
 *   }
 * }
 * ```
 */
export function createThunderContextEngine(
  options?: Partial<ThunderOmlxConfig>
): ContextEngine {
  return new ThunderContextEngine(options);
}

export default createThunderContextEngine;
export type { ThunderOmlxConfig, ContextEngine } from "./types.js";
