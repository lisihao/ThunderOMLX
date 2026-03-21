"""ContextPilot integration for OMLX Cloud.

Wraps ContextPilot's reorder/deduplicate APIs to work with OpenAI-format
messages. Provides KV-cache-aware context optimization that complements
OMLX's existing context management (compression/summarization).

Pipeline position:
    context optimization -> ContextPilotOptimizer.optimize() -> backend dispatch

Level 1: Reorder context blocks for KV cache prefix sharing (local + cloud).
Level 2: Deduplicate repeated context in multi-turn conversations (cloud token savings).
"""

import logging
import os
import re
import sys
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("omlx.cloud.context_pilot")

# ---------------------------------------------------------------------------
# Lazy import of vendor ContextPilot
# ---------------------------------------------------------------------------

_cp_available = None
_ContextPilot = None


def _ensure_contextpilot():
    """Lazily import ContextPilot from vendor directory."""
    global _cp_available, _ContextPilot
    if _cp_available is not None:
        return _cp_available

    try:
        # Add vendor path if not already present
        vendor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "vendor", "contextpilot",
        )
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

        from contextpilot.server.live_index import ContextPilot
        _ContextPilot = ContextPilot
        _cp_available = True
        logger.info("[ContextPilot] 库加载成功 (v0.3.5)")
    except ImportError as e:
        _cp_available = False
        logger.warning(f"[ContextPilot] 库不可用: {e}，跳过上下文重排优化")

    return _cp_available


class ContextPilotOptimizer:
    """Optimizes OpenAI-format messages for KV cache reuse and token savings.

    Level 1 (reorder): Reorders context blocks for maximum KV cache prefix
        sharing. Benefits local inference (llama.cpp, vLLM) directly.

    Level 2 (deduplicate): On Turn 2+ of multi-turn conversations, strips
        already-seen documents and replaces them with lightweight reference
        hints. Saves cloud API tokens (~36% reduction for repeated context).

    Flow:
        Turn 1: reorder() — register docs, optimize ordering
        Turn 2+: deduplicate() — strip overlapping, add hints, save tokens
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and _ensure_contextpilot()
        self._pilot = None
        self._stats = {
            "total_calls": 0,
            "total_reordered": 0,
            "total_deduplicated": 0,
            "total_skipped": 0,
            "total_tokens_saved": 0,
        }

    @property
    def pilot(self):
        """Lazy-init ContextPilot instance."""
        if self._pilot is None and self.enabled:
            self._pilot = _ContextPilot(use_gpu=False)
        return self._pilot

    def _has_conversation_history(self, conversation_id: str) -> bool:
        """Check if a conversation has prior reorder history."""
        if not self.pilot:
            return False
        return conversation_id in self.pilot._conversations

    def optimize(
        self,
        messages: List[Dict],
        *,
        conversation_id: Optional[str] = None,
        min_context_blocks: int = 2,
        metadata_mode: bool = False,
    ) -> Tuple[List[Dict], Dict]:
        """Optimize messages for KV cache reuse and token savings.

        On Turn 1 (or without conversation_id), reorders context blocks for
        maximum KV cache prefix sharing.

        On Turn 2+ (conversation_id has prior history), deduplicates repeated
        context blocks — only sends new docs plus lightweight reference hints
        for already-seen docs. This saves cloud API tokens.

        Args:
            messages: OpenAI-format messages list.
            conversation_id: Session/conversation ID for cross-turn dedup.
            min_context_blocks: Minimum context blocks to trigger optimization.
                With fewer blocks, reordering has no benefit.
            metadata_mode: If True, return original messages with metadata
                instead of modifying messages. Used for ThunderLLAMA prefix matching.

        Returns:
            (optimized_messages, metadata) where metadata contains stats
            about what was optimized.
        """
        self._stats["total_calls"] += 1

        if not self.enabled or not messages:
            self._stats["total_skipped"] += 1
            return messages, {"optimized": False, "reason": "disabled_or_empty"}

        # Extract context structure from messages
        system_msg, context_blocks, query, other_messages = self._extract_context(messages)

        # Need at least min_context_blocks to make reordering worthwhile
        if len(context_blocks) < min_context_blocks:
            self._stats["total_skipped"] += 1
            return messages, {
                "optimized": False,
                "reason": f"too_few_blocks ({len(context_blocks)})",
            }

        # Level 2: Multi-turn deduplication (Turn 2+)
        # IMPORTANT: dedup only operates on system-message doc blocks.
        # Conversation turns (user/assistant) must stay in their original
        # positions to preserve message structure for the cloud API.
        if conversation_id and self._has_conversation_history(conversation_id):
            try:
                return self._optimize_with_dedup(
                    messages, conversation_id,
                )
            except Exception as e:
                logger.warning(f"[ContextPilot] 去重失败，回退到重排: {e}")
                # Fall through to reorder

        # Level 1: Reorder for KV cache prefix sharing (Turn 1)
        return self._optimize_with_reorder(
            system_msg, context_blocks, query, other_messages,
            conversation_id, metadata_mode,
        )

    def _optimize_with_reorder(
        self,
        system_msg: Optional[Dict],
        context_blocks: List[str],
        query: str,
        other_messages: List[Dict],
        conversation_id: Optional[str],
        metadata_mode: bool = False,
    ) -> Tuple[List[Dict], Dict]:
        """Level 1: Reorder context blocks for KV cache prefix sharing."""
        try:
            reordered, indices = self.pilot.reorder(
                context_blocks,
                conversation_id=conversation_id,
            )
            reordered_blocks = reordered[0]  # single context -> first element

            # Build importance ranking annotation
            pos = {block: i + 1 for i, block in enumerate(reordered_blocks)}
            importance = " > ".join(
                str(pos[b]) for b in context_blocks if b in pos
            )

            # Metadata mode: return original messages + reorder metadata
            if metadata_mode:
                original_messages = self._rebuild_original(
                    system_msg, context_blocks, query, other_messages,
                )
                metadata = {
                    "optimized": True,
                    "method": "metadata_reorder",
                    "blocks": len(context_blocks),
                    "conversation_id": conversation_id,
                    "optimal_order": list(indices),  # Reorder indices
                    "importance": importance,
                }
                self._stats["total_reordered"] += 1
                logger.info(
                    f"[ContextPilot] 元数据模式：重排 {len(context_blocks)} 块 | "
                    f"conv={conversation_id or 'none'}"
                )
                return original_messages, metadata

            # Standard mode: modify messages with reordering
            optimized = self._reconstruct_messages(
                system_msg, reordered_blocks, importance, query, other_messages,
            )

            self._stats["total_reordered"] += 1
            logger.info(
                f"[ContextPilot] 重排 {len(context_blocks)} 个上下文块 | "
                f"conv={conversation_id or 'none'}"
            )

            return optimized, {
                "optimized": True,
                "method": "reorder",
                "blocks": len(context_blocks),
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.warning(f"[ContextPilot] 重排失败，使用原始消息: {e}")
            self._stats["total_skipped"] += 1
            return (
                self._rebuild_original(system_msg, context_blocks, query, other_messages),
                {"optimized": False, "reason": f"error: {e}"},
            )

    def _optimize_with_dedup(
        self,
        messages: List[Dict],
        conversation_id: str,
    ) -> Tuple[List[Dict], Dict]:
        """Level 2: Deduplicate repeated docs in system message only.

        IMPORTANT: Only deduplicates document blocks within the system
        message. User/assistant conversation turns are preserved in their
        original positions to maintain proper message structure for
        stateless cloud APIs.
        """
        # Step 1: Find system message and extract ONLY its doc blocks
        system_idx = None
        system_content = ""
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                system_content = msg.get("content", "") or ""
                break

        if system_idx is None or not system_content:
            self._stats["total_skipped"] += 1
            return messages, {"optimized": False, "reason": "no_system_msg"}

        doc_blocks = self._split_system_into_blocks(system_content)
        if len(doc_blocks) < 2:
            self._stats["total_skipped"] += 1
            return messages, {"optimized": False, "reason": "too_few_doc_blocks"}

        # Step 2: Deduplicate doc blocks only (not conversation turns)
        dedup_results = self.pilot.deduplicate(
            [doc_blocks],
            conversation_id=conversation_id,
        )
        result = dedup_results[0]
        new_docs = result["new_docs"]
        overlapping_docs = result["overlapping_docs"]

        if not overlapping_docs:
            self._stats["total_skipped"] += 1
            return messages, {
                "optimized": False,
                "reason": "no_overlap",
                "conversation_id": conversation_id,
            }

        # Step 3: Generate compact reference hints
        compact_hints = []
        for i, doc in enumerate(overlapping_docs):
            preview = doc[:50].replace("\n", " ")
            if len(doc) > 50:
                preview += "..."
            compact_hints.append(
                f"[Ref {i + 1}] (previously sent) {preview}"
            )

        # Step 4: Rebuild ONLY the system message, keep everything else
        parts = []
        if compact_hints:
            hints_text = "\n".join(f"- {h}" for h in compact_hints)
            parts.append(f"Previously provided context (already in conversation):\n{hints_text}")
        if new_docs:
            docs_section = "\n".join(
                f"[{i + 1}] {doc}" for i, doc in enumerate(new_docs)
            )
            parts.append(f"New context:\n{docs_section}")

        new_system_content = "\n\n".join(parts) if parts else system_content

        # Step 5: Reconstruct messages — only replace system, keep all
        # user/assistant turns in their original positions
        optimized = []
        for i, msg in enumerate(messages):
            if i == system_idx:
                optimized.append({"role": "system", "content": new_system_content})
            else:
                optimized.append(msg)

        # Calculate token savings
        overlap_chars = sum(len(d) for d in overlapping_docs)
        hint_chars = sum(len(h) for h in compact_hints)
        chars_saved = overlap_chars - hint_chars
        tokens_saved = max(0, chars_saved // 4)

        self._stats["total_deduplicated"] += 1
        self._stats["total_tokens_saved"] += tokens_saved

        logger.info(
            f"[ContextPilot] 去重 {len(overlapping_docs)} 文档块 | "
            f"新增 {len(new_docs)} 文档块 | "
            f"对话轮次保持不变 | "
            f"节省 ~{tokens_saved} tokens | conv={conversation_id}"
        )

        return optimized, {
            "optimized": True,
            "method": "deduplicate",
            "new_blocks": len(new_docs),
            "deduped_blocks": len(overlapping_docs),
            "tokens_saved": tokens_saved,
            "conversation_id": conversation_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_context(
        self, messages: List[Dict]
    ) -> Tuple[Optional[Dict], List[str], str, List[Dict]]:
        """Extract context structure from OpenAI messages.

        Splits messages into:
        - system_msg: The system message (if any)
        - context_blocks: Extractable context chunks (from system + prior turns)
        - query: The last user message
        - other_messages: Messages that shouldn't be reordered (assistant replies etc.)

        Returns:
            (system_msg, context_blocks, query, other_messages)
        """
        system_msg = None
        context_blocks = []
        query = ""
        other_messages = []

        # Find last user message (the query)
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                query = messages[i].get("content", "")
                break

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "system":
                system_msg = msg
                # Split system message into blocks if it contains sections
                blocks = self._split_system_into_blocks(content)
                context_blocks.extend(blocks)

            elif role == "user" and i == last_user_idx:
                # This is the query, skip for now
                continue

            elif role in ("user", "assistant") and i < last_user_idx:
                # Prior conversation turns -> treat as context blocks
                prefix = "User: " if role == "user" else "Assistant: "
                context_blocks.append(prefix + content)

            else:
                other_messages.append(msg)

        return system_msg, context_blocks, query, other_messages

    def _split_system_into_blocks(self, content: str) -> List[str]:
        """Split a system message into context blocks.

        Detects common separators (double newline, XML-like tags, numbered
        sections) to identify individual context chunks within a system prompt.
        """
        if not content or len(content) < 100:
            return [content] if content else []

        # Try XML-like document tags: <doc>, <document>, <context>, etc.
        xml_pattern = re.compile(
            r'<(?:doc|document|context|passage|chunk|source)\b[^>]*>(.*?)</(?:doc|document|context|passage|chunk|source)>',
            re.DOTALL | re.IGNORECASE,
        )
        xml_matches = xml_pattern.findall(content)
        if len(xml_matches) >= 2:
            return [m.strip() for m in xml_matches if m.strip()]

        # Try numbered sections: [1], [2], ... or 1., 2., ...
        numbered_pattern = re.compile(r'\n\s*(?:\[\d+\]|\d+\.)\s+')
        sections = numbered_pattern.split(content)
        if len(sections) >= 3:
            return [s.strip() for s in sections if s.strip()]

        # Try double-newline separation
        paragraphs = content.split("\n\n")
        if len(paragraphs) >= 3:
            return [p.strip() for p in paragraphs if p.strip()]

        # No good split found - return as single block
        return [content]

    def _reconstruct_messages(
        self,
        system_msg: Optional[Dict],
        reordered_blocks: List[str],
        importance: str,
        query: str,
        other_messages: List[Dict],
    ) -> List[Dict]:
        """Reconstruct OpenAI messages with reordered context.

        Puts reordered context blocks into the system message, preserves
        importance ranking annotation, and keeps the query as the last
        user message.
        """
        result = []

        # Build system message with reordered context
        if system_msg or reordered_blocks:
            docs_section = "\n".join(
                f"[{i + 1}] {block}" for i, block in enumerate(reordered_blocks)
            )
            result.append({
                "role": "system",
                "content": f"{docs_section}\n\nRead in importance order: {importance}",
            })

        # Add any other messages (shouldn't normally exist)
        result.extend(other_messages)

        # Query is always last
        if query:
            result.append({"role": "user", "content": query})

        return result

    def _reconstruct_dedup_messages(
        self,
        system_msg: Optional[Dict],
        new_docs: List[str],
        reference_hints: List[str],
        query: str,
        other_messages: List[Dict],
    ) -> List[Dict]:
        """Reconstruct messages with deduplicated context.

        Only includes new (unseen) documents. Overlapping documents are
        replaced with lightweight reference hints that tell the model to
        reuse context from prior turns.
        """
        result = []
        parts = []

        # Reference hints for already-seen context (lightweight)
        if reference_hints:
            hints_text = "\n".join(f"- {hint}" for hint in reference_hints)
            parts.append(f"Previously provided context (already in conversation):\n{hints_text}")

        # New documents not seen in prior turns
        if new_docs:
            docs_section = "\n".join(
                f"[{i + 1}] {doc}" for i, doc in enumerate(new_docs)
            )
            parts.append(f"New context:\n{docs_section}")

        if parts:
            result.append({"role": "system", "content": "\n\n".join(parts)})

        result.extend(other_messages)

        if query:
            result.append({"role": "user", "content": query})

        return result

    def _rebuild_original(
        self,
        system_msg: Optional[Dict],
        context_blocks: List[str],
        query: str,
        other_messages: List[Dict],
    ) -> List[Dict]:
        """Rebuild original message structure (fallback on error)."""
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(other_messages)
        if query:
            result.append({"role": "user", "content": query})
        return result

    def get_stats(self) -> Dict:
        """Return optimization statistics."""
        return {
            "enabled": self.enabled,
            "contextpilot_available": bool(_cp_available),
            **self._stats,
        }
