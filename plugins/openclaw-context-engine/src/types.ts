/**
 * Types mirroring OpenClaw's ContextEngine interface.
 * @see https://github.com/openclaw/openclaw/blob/master/src/context-engine/types.ts
 */

export interface AgentMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentPart[];
  name?: string;
  tool_calls?: unknown[];
  tool_call_id?: string;
}

export interface ContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

export interface ContextEngineInfo {
  id: string;
  name: string;
  version?: string;
  ownsCompaction?: boolean;
}

export interface AssembleResult {
  messages: AgentMessage[];
  estimatedTokens: number;
  systemPromptAddition?: string;
}

export interface IngestResult {
  ingested: boolean;
}

export interface CompactResult {
  ok: boolean;
  compacted: boolean;
  reason?: string;
  result?: {
    summary?: string;
    tokensBefore: number;
    tokensAfter?: number;
  };
}

export interface BootstrapResult {
  bootstrapped: boolean;
  importedMessages?: number;
  reason?: string;
}

export interface ContextEngine {
  readonly info: ContextEngineInfo;
  bootstrap?(params: { sessionId: string; sessionKey?: string; sessionFile: string }): Promise<BootstrapResult>;
  ingest(params: { sessionId: string; sessionKey?: string; message: AgentMessage; isHeartbeat?: boolean }): Promise<IngestResult>;
  assemble(params: { sessionId: string; sessionKey?: string; messages: AgentMessage[]; tokenBudget?: number }): Promise<AssembleResult>;
  compact(params: { sessionId: string; sessionKey?: string; sessionFile: string; tokenBudget?: number; force?: boolean }): Promise<CompactResult>;
  dispose?(): Promise<void>;
}

/** Plugin configuration options */
export interface ThunderOmlxConfig {
  /** ThunderOMLX server endpoint (default: http://127.0.0.1:8082) */
  endpoint: string;
  /** API key for ThunderOMLX authentication */
  apiKey?: string;
  /** Timeout in ms for HTTP calls (default: 800) */
  timeout: number;
  /** Number of consecutive failures before disabling remote calls (default: 3) */
  fallbackThreshold: number;
  /** Features to enable */
  features: ("prefix_align" | "semantic_dedup")[];
}

/** Response from /v1/context/optimize */
export interface OptimizeResponse {
  optimized_messages: AgentMessage[];
  estimated_tokens: number;
  prefix_hash: string | null;
  dedup_count: number;
  cache_hint: string | null;
}
