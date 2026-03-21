# @omlx/thunder-context

ThunderOMLX ContextEngine plugin for OpenClaw v2026.3.7+.

Optimizes context assembly by reordering messages to maximize cloud prompt cache hit rates (Anthropic, Google, OpenAI).

## Install

```bash
npm install @omlx/thunder-context
```

## Configure

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "@omlx/thunder-context"
    },
    "entries": {
      "@omlx/thunder-context": {
        "enabled": true,
        "options": {
          "endpoint": "http://127.0.0.1:8082",
          "timeout": 800,
          "features": ["prefix_align", "semantic_dedup"]
        }
      }
    }
  }
}
```

## Requirements

- OpenClaw >= v2026.3.7-beta.1
- ThunderOMLX running on localhost (or reachable endpoint)

## How it works

1. `ingest()` — No-op (zero overhead)
2. `assemble()` — Calls ThunderOMLX `/v1/context/optimize` to reorder messages
3. `compact()` — Delegates to OpenClaw runtime (Phase 2: async local summarization)

Falls back to passthrough if ThunderOMLX is unreachable.
