"""ConversationStore - Persistent conversation memory (24h TTL)

Store-then-compress: persist segments into SQLite before compression,
reconstruct context intelligently based on conversation mode (work/casual).

v2: Structured summaries + anti-pollution
  - User messages -> extract intent, actions, files (trusted backbone)
  - Assistant messages -> only extract code blocks, tool output, status words (verifiable artifacts)
  - Based on MIT "Do LLMs Benefit From Their Own Words?" paper
    to avoid assistant hallucinations propagating through summaries

Flow:
  Request in -> derive_conversation_id -> segment -> store_segments -> reconstruct_context

Ported from ClawGate to ThunderOMLX with inline sqlite3 storage
(replaces SQLiteStore dependency).
"""

import hashlib
import json
import logging
import os
import re
import sqlite3 as _sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Callable

logger = logging.getLogger("omlx.cloud.conversation_store")

# Segment TTL
SEGMENT_TTL_HOURS = 24

# Default database path
_DEFAULT_DB_PATH = "~/.omlx/data/conversations.db"


class TopicSegment:
    """A topic segment (inlined from ClawGate topic_segmenter)"""

    def __init__(self, start: int, end: int, topic_type: str, confidence: float):
        self.start = start            # Start message index (inclusive)
        self.end = end                # End message index (exclusive)
        self.topic_type = topic_type  # "work" | "casual"
        self.confidence = confidence
        self.messages: List[Dict] = []

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self):
        return (
            f"<Segment [{self.start}:{self.end}] "
            f"{self.topic_type} conf={self.confidence:.2f} msgs={self.length}>"
        )


class ConversationDB:
    """Lightweight sqlite3 storage backend for ConversationStore.

    Replaces the shared SQLiteStore with direct sqlite3 calls,
    scoped to conversation_segments and long_term_memories tables.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        expanded = os.path.expanduser(db_path)
        self.db_path = Path(expanded)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        conn = _sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                topic_type TEXT NOT NULL,
                summary TEXT,
                messages TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                UNIQUE(conversation_id, segment_index)
            );

            CREATE INDEX IF NOT EXISTS idx_conv_seg_conv_id
                ON conversation_segments(conversation_id);

            CREATE TABLE IF NOT EXISTS long_term_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT UNIQUE NOT NULL,
                key_files TEXT NOT NULL,
                summary TEXT NOT NULL,
                conversation_id TEXT,
                access_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_ltm_key
                ON long_term_memories(memory_key);
            """
        )
        conn.close()


class ConversationStore:
    """Conversation segment persistent storage with mode-aware context reconstruction"""

    # Cross-session memory TTL (days)
    LTM_TTL_DAYS = 7

    # Background thread pool for async LLM summaries (F3)
    _summary_executor = ThreadPoolExecutor(max_workers=2)

    def __init__(
        self,
        db: ConversationDB,
        topic_segmenter=None,
        llm_summarizer: Optional[Callable[[str], str]] = None,
    ):
        self.db = db
        self.segmenter = topic_segmenter
        self.llm_summarizer = llm_summarizer

    # ========== Conversation ID Derivation ==========

    def derive_conversation_id(self, messages: List[Dict]) -> str:
        """Derive conversation ID from message list automatically

        Uses hash of system_prompt first 500 chars + first user message first 200 chars.
        Same session always produces the same conversation_id.
        """
        system_part = ""
        first_user_part = ""

        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            if role == "system" and not system_part:
                system_part = content[:500]
            elif role == "user" and not first_user_part:
                first_user_part = content[:200]
            if system_part and first_user_part:
                break

        raw = f"{system_part}||{first_user_part}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ========== Segment Storage ==========

    def store_segments(
        self, conversation_id: str, segments: List[TopicSegment]
    ) -> int:
        """Store segments into SQLite (INSERT OR REPLACE)

        Each request from the client sends full messages, so we re-segment
        and overwrite storage each time. Expired segments are cleaned first.

        Returns:
            Number of segments stored
        """
        # Clean expired first
        self.cleanup_expired()

        expires_at = (
            datetime.utcnow() + timedelta(hours=SEGMENT_TTL_HOURS)
        ).strftime("%Y-%m-%d %H:%M:%S")

        conn = _sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        stored = 0
        for idx, seg in enumerate(segments):
            summary = self._generate_segment_summary(seg)
            messages_json = json.dumps(seg.messages, ensure_ascii=False)

            cursor.execute(
                """
                INSERT OR REPLACE INTO conversation_segments
                    (conversation_id, segment_index, topic_type, summary,
                     messages, message_count, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    conversation_id,
                    idx,
                    seg.topic_type,
                    summary,
                    messages_json,
                    seg.length,
                    expires_at,
                ),
            )
            stored += 1

        # Delete old segments beyond current count
        cursor.execute(
            "DELETE FROM conversation_segments WHERE conversation_id = ? AND segment_index >= ?",
            (conversation_id, len(segments)),
        )

        conn.commit()
        conn.close()

        # P4: Promote work segments to long_term_memories
        for seg in segments:
            if seg.topic_type == "work":
                summary = self._generate_segment_summary(seg)
                self._promote_to_long_term(
                    segment=seg,
                    summary=summary,
                    conversation_id=conversation_id,
                )

        # F3: Background async LLM summary (fire-and-forget)
        if self.llm_summarizer:
            for idx, seg in enumerate(segments):
                if seg.topic_type == "work" and seg.length > 5:
                    self._summary_executor.submit(
                        self._background_llm_summary,
                        conversation_id,
                        idx,
                        seg,
                    )

        logger.info(
            f"[ConvStore] Stored conv={conversation_id[:8]}... | "
            f"{stored} segments | TTL={SEGMENT_TTL_HOURS}h"
        )
        return stored

    # ========== Segment Query ==========

    def get_segments(
        self, conversation_id: str, topic_type: Optional[str] = None
    ) -> List[Dict]:
        """Query stored segments for a conversation, filtering expired ones

        Returns:
            [{conversation_id, segment_index, topic_type, summary,
              messages (parsed), message_count, created_at, expires_at}]
        """
        conn = _sqlite3.connect(self.db.db_path)
        conn.row_factory = _sqlite3.Row
        cursor = conn.cursor()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        if topic_type:
            cursor.execute(
                """
                SELECT * FROM conversation_segments
                WHERE conversation_id = ? AND topic_type = ? AND expires_at > ?
                ORDER BY segment_index ASC
                """,
                (conversation_id, topic_type, now),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM conversation_segments
                WHERE conversation_id = ? AND expires_at > ?
                ORDER BY segment_index ASC
                """,
                (conversation_id, now),
            )

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Parse messages JSON
        for row in rows:
            row["messages"] = json.loads(row["messages"])

        return rows

    # ========== Intelligent Reconstruction ==========

    def reconstruct_context(
        self,
        conversation_id: str,
        messages: List[Dict],
        mode: str,
        target_tokens: int,
        tokenizer=None,
        model_tier: str = "medium",
    ) -> Tuple[List[Dict], Dict]:
        """Reconstruct context from Store based on mode

        Args:
            conversation_id: Conversation ID
            messages: Current full message list (for tail extraction)
            mode: "work" or "casual"
            target_tokens: Target token count
            tokenizer: tiktoken tokenizer
            model_tier: Model capability tier ("weak"/"medium"/"strong")

        Returns:
            (reconstructed message list, metadata)
        """
        stored_segments = self.get_segments(conversation_id)

        if not stored_segments:
            # No stored segments (possibly new session) -> try cross-session memory recall
            tail_result = self._tail_fit(messages, target_tokens, tokenizer, model_tier)
            ltm_msgs = self._recall_long_term(messages)
            meta = {
                "strategy": "conv_store_tail",
                "stored_segments": 0,
                "ltm_recall": len(ltm_msgs),
            }
            if ltm_msgs:
                # Inject LTM messages after system messages
                system_msgs = [m for m in tail_result if m.get("role") == "system"]
                non_system = [m for m in tail_result if m.get("role") != "system"]
                tail_result = system_msgs + ltm_msgs + non_system
                logger.info(
                    f"[ConvStore] Cross-session memory recall: {len(ltm_msgs)} memories injected"
                )
            return tail_result, meta

        if mode == "work":
            result = self._reconstruct_work(
                stored_segments, messages, target_tokens, tokenizer, model_tier
            )
        else:
            result = self._reconstruct_casual(
                stored_segments, messages, target_tokens, tokenizer, model_tier
            )

        final_tokens = self._count_tokens(result, tokenizer)
        logger.info(
            f"[ConvStore] Reconstructed conv={conversation_id[:8]}... | mode={mode} | "
            f"segments={len(stored_segments)} | result={len(result)} msgs {final_tokens} tokens"
        )

        return result, {
            "strategy": "conv_store_reconstruct",
            "mode": mode,
            "model_tier": model_tier,
            "stored_segments": len(stored_segments),
            "result_tokens": final_tokens,
        }

    def _reconstruct_work(
        self,
        stored_segments: List[Dict],
        messages: List[Dict],
        target_tokens: int,
        tokenizer,
        model_tier: str = "medium",
    ) -> List[Dict]:
        """Work mode reconstruction: old work summaries + recent work full + latest tail

        Budget allocation:
          - 40% tail (latest messages, fully preserved)
          - 60% historical work segments (summaries preferred, recent segments full)
        """
        result = []

        # 1. Preserve system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        result.extend(system_msgs)
        sys_tokens = self._count_tokens(system_msgs, tokenizer)

        available = target_tokens - sys_tokens
        tail_budget = int(available * 0.4)
        history_budget = available - tail_budget

        # 2. Historical work segments (from store)
        work_segments = [s for s in stored_segments if s["topic_type"] == "work"]
        history_msgs = []

        if work_segments:
            # Last work segment: preserve fully if possible
            last_work = work_segments[-1]
            earlier_work = work_segments[:-1]

            # Earlier work segments: use summaries
            for seg in earlier_work:
                if seg.get("summary"):
                    history_msgs.append({
                        "role": "system",
                        "content": f"[Historical work summary] {seg['summary']}",
                    })

            # Recent work segment: full messages
            last_work_msgs = last_work.get("messages", [])
            last_work_tokens = self._count_tokens(last_work_msgs, tokenizer)
            summary_tokens = self._count_tokens(history_msgs, tokenizer)

            if summary_tokens + last_work_tokens <= history_budget:
                history_msgs.extend(last_work_msgs)
            else:
                # Budget insufficient, use summary only
                if last_work.get("summary"):
                    history_msgs.append({
                        "role": "system",
                        "content": f"[Recent work summary] {last_work['summary']}",
                    })

        # Trim history to budget
        history_msgs = self._fit_to_budget(history_msgs, history_budget, tokenizer)
        result.extend(history_msgs)

        # 3. Tail (latest messages)
        non_system = [m for m in messages if m.get("role") != "system"]
        tail = self._tail_messages(non_system, tail_budget, tokenizer, model_tier)
        result.extend(tail)

        return result

    def _reconstruct_casual(
        self,
        stored_segments: List[Dict],
        messages: List[Dict],
        target_tokens: int,
        tokenizer,
        model_tier: str = "medium",
    ) -> List[Dict]:
        """Casual mode reconstruction: work one-liner summary + recent casual"""
        result = []

        # 1. system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        result.extend(system_msgs)
        sys_tokens = self._count_tokens(system_msgs, tokenizer)

        available = target_tokens - sys_tokens

        # 2. Work segments one-liner summary
        work_segments = [s for s in stored_segments if s["topic_type"] == "work"]
        if work_segments:
            topics = []
            for seg in work_segments:
                if seg.get("summary"):
                    topics.append(seg["summary"])
            if topics:
                combined = "; ".join(topics[:3])
                result.append({
                    "role": "system",
                    "content": f"[Work context summary] {combined}",
                })

        summary_tokens = self._count_tokens(result, tokenizer) - sys_tokens
        remaining = available - summary_tokens

        # 3. Recent casual (from tail)
        non_system = [m for m in messages if m.get("role") != "system"]
        tail = self._tail_messages(non_system, remaining, tokenizer, model_tier)
        result.extend(tail)

        return result

    # ========== Expiration Cleanup ==========

    def cleanup_expired(self) -> int:
        """Clean up expired segments

        Returns:
            Number of deleted rows
        """
        conn = _sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "DELETE FROM conversation_segments WHERE expires_at < ?", (now,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted > 0:
            logger.info(f"[ConvStore] Cleaned expired segments: {deleted}")

        return deleted

    # ========== Structured Summary Generation ==========

    # Action verb normalization mapping
    _ACTION_SYNONYMS = {
        "implement": ["implement", "develop", "write", "add", "create", "new",
                       "实现", "开发", "编写", "新增", "添加", "新建"],
        "fix": ["fix", "debug", "resolve", "修复", "解决", "修正"],
        "optimize": ["optimize", "refactor", "improve", "speed up",
                      "优化", "重构", "提升", "加速", "改进"],
        "test": ["test", "verify", "validate", "测试", "验证", "确保"],
        "analyze": ["analyze", "design", "research",
                     "分析", "设计", "研究", "调研", "讨论"],
        "configure": ["config", "deploy", "setup", "配置", "部署", "设置"],
        "delete": ["delete", "remove", "clean", "删除", "移除", "清理"],
    }

    # Intent lead words
    _INTENT_PATTERN = re.compile(
        r"(我[想需要得]|请帮我|请你|目标是|如何|怎样|怎么|为什么|"
        r"帮我|麻烦|能不能|可以.{0,2}吗|implement|fix|add|create|how to|please)",
        re.IGNORECASE,
    )

    # Status keywords
    _STATUS_DONE = re.compile(
        r"(完成|搞定|已修复|已实现|成功|通过|ok|done|solved|fixed|passed)",
        re.IGNORECASE,
    )
    _STATUS_BLOCKED = re.compile(
        r"(失败|错误|问题|无法|阻塞|不行|报错|failed|error|blocked|broken)",
        re.IGNORECASE,
    )

    # File path extraction
    _FILE_PATTERN = re.compile(
        r"([\w\-\./]+\.(py|ts|js|go|rs|java|yaml|json|md|sh|sql|toml|cfg|proto))\b",
        re.IGNORECASE,
    )

    # Code block extraction (assistant verifiable artifacts)
    _CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n[\s\S]*?```")

    def _generate_segment_summary(self, segment: TopicSegment) -> str:
        """Generate structured segment summary (anti-pollution version)

        Core principles (based on MIT context pollution paper):
        - User messages -> extract intent, actions, files (trusted backbone)
        - Assistant messages -> only extract verifiable artifacts (code block count, status)
        - Do not propagate assistant explanations/reasoning/analysis (potential hallucinations)

        v4 (F3): Always synchronously return rule-based summary, LLM summary runs
                  asynchronously after store_segments. Avoids ~200ms sync blocking.
        """
        if not segment.messages:
            return f"Empty segment ({segment.topic_type})"

        if segment.topic_type == "work":
            return self._summarize_work_segment(segment)
        else:
            return self._summarize_casual_segment(segment)

    def _summarize_work_segment(self, segment: TopicSegment) -> str:
        """Generate structured summary for work segment"""
        files: Set[str] = set()
        actions: Set[str] = set()
        user_intent = ""
        code_block_count = 0
        status = ""

        for msg in segment.messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            # --- All messages: extract file names ---
            for match in self._FILE_PATTERN.finditer(content):
                filepath = match.group(1)
                filename = filepath.split("/")[-1]
                files.add(filename)

            if role == "user":
                # --- User messages: extract intent and actions ---
                if not user_intent and self._INTENT_PATTERN.search(content):
                    first_line = content.split("\n")[0].strip()
                    user_intent = first_line[:80]

                # Normalize action verbs
                for main_action, synonyms in self._ACTION_SYNONYMS.items():
                    for syn in synonyms:
                        if syn in content.lower() if syn.isascii() else syn in content:
                            actions.add(main_action)
                            break

            elif role == "assistant":
                # --- Assistant messages: only extract verifiable artifacts ---
                # 1. Code block count (verifiable)
                code_block_count += len(self._CODE_BLOCK_PATTERN.findall(content))

                # 2. Status words (done/blocked, verifiable)
                if not status:
                    if self._STATUS_DONE.search(content):
                        status = "done"
                    elif self._STATUS_BLOCKED.search(content):
                        status = "blocked"

                # Do not extract assistant explanations/reasoning (anti-pollution)

        # --- Assemble structured summary ---
        parts = ["[WORK]"]

        # Line 1: files + actions
        line1_parts = []
        if files:
            line1_parts.append(f"files: {', '.join(sorted(files)[:5])}")
        if actions:
            line1_parts.append(f"actions: {', '.join(sorted(actions))}")
        if line1_parts:
            parts.append(" | ".join(line1_parts))

        # Line 2: intent (from user only)
        if user_intent:
            parts.append(f"intent: {user_intent}")

        # Line 3: artifacts + status
        line3_parts = []
        if code_block_count > 0:
            line3_parts.append(f"{code_block_count} code blocks")
        if status:
            line3_parts.append(f"status: {status}")
        if line3_parts:
            parts.append(" | ".join(line3_parts))

        # Fallback: if nothing was extracted
        if len(parts) == 1:
            parts.append(f"work segment ({segment.length} messages)")

        return "\n".join(parts)

    def _llm_summarize_work_segment(self, segment: TopicSegment) -> str:
        """Use LLM to generate summary for long work segments (>5 msgs)

        Falls back to rule-based summary on failure.
        """
        # Build conversation text (truncate long content)
        formatted_lines = []
        for msg in segment.messages:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:300]
            formatted_lines.append(f"[{role}]: {content}")
        formatted_messages = "\n".join(formatted_lines)

        prompt = (
            "You are a context compression expert. Extract key facts from the following "
            "conversation, strictly following these rules:\n\n"
            "Rules:\n"
            "1. Only extract facts explicitly stated by the user (intent, files, actions, decisions)\n"
            "2. From assistant replies, only extract: file names, function names, completion status\n"
            "3. Do not speculate, supplement, or explain reasons\n"
            "4. Keep it concise, no more than 150 words total\n\n"
            "Output format:\n"
            "Files: [file list]\n"
            "Actions: [implement/fix/optimize/test/analyze/configure/delete]\n"
            "User intent: [one sentence]\n"
            "Key decisions: [if any, otherwise omit]\n"
            "Status: [done/in-progress/blocked]\n\n"
            f"Conversation:\n{formatted_messages}"
        )

        try:
            llm_result = self.llm_summarizer(prompt)
            # Add [WORK] prefix for downstream consistency
            return f"[WORK]\n{llm_result.strip()}"
        except Exception as e:
            logger.warning(
                f"[ConvStore] LLM summary failed, falling back to rule-based: {e}"
            )
            return self._summarize_work_segment(segment)

    def _background_llm_summary(
        self, conversation_id: str, segment_index: int, segment: TopicSegment
    ):
        """Background task: generate LLM summary and UPDATE the DB row.

        Called via ThreadPoolExecutor.submit() after store_segments commits.
        If LLM fails, the rule-based summary remains intact.
        """
        try:
            llm_summary = self._llm_summarize_work_segment(segment)
            # Update the summary in SQLite
            conn = _sqlite3.connect(self.db.db_path)
            conn.execute(
                "UPDATE conversation_segments SET summary = ? "
                "WHERE conversation_id = ? AND segment_index = ?",
                (llm_summary, conversation_id, segment_index),
            )
            conn.commit()
            conn.close()
            logger.debug(
                f"[ConvStore] Background LLM summary updated: "
                f"conv={conversation_id[:8]}... seg={segment_index}"
            )
        except Exception as e:
            # Rule-based summary already saved, LLM failure is non-critical
            logger.warning(
                f"[ConvStore] Background LLM summary failed: {e}"
            )

    def _summarize_casual_segment(self, segment: TopicSegment) -> str:
        """Generate structured summary for casual segment"""
        user_points: List[str] = []

        for msg in segment.messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()

            if role == "user" and content:
                # Only collect user talking points (first 60 chars)
                point = content.split("\n")[0][:60]
                if point and len(point) > 2:
                    user_points.append(point)

        parts = [f"[CASUAL] ({segment.length} messages)"]

        if user_points:
            # At most 2 points
            for point in user_points[:2]:
                parts.append(f"- {point}")

        return "\n".join(parts)

    def _count_tokens(self, messages: List[Dict], tokenizer) -> int:
        """Count tokens"""
        total = 0
        for msg in messages:
            content = str(msg.get("content", ""))
            if tokenizer:
                total += len(tokenizer.encode(content))
            else:
                total += len(content) // 4
        return total

    def _tail_fit(
        self, messages: List[Dict], target_tokens: int, tokenizer,
        model_tier: str = "medium",
    ) -> List[Dict]:
        """Simple tail truncation"""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        sys_tokens = self._count_tokens(system_msgs, tokenizer)
        remaining = target_tokens - sys_tokens

        tail = self._tail_messages(non_system, remaining, tokenizer, model_tier)
        return system_msgs + tail

    def _strip_assistant_to_artifacts(self, content: str) -> str:
        """Weak models: strip assistant messages to verifiable artifacts only (code blocks)

        Based on MIT paper: weak models are more susceptible to context pollution,
        assistant narrative text (explanations/reasoning) is the main pollution source.
        Only keep code blocks as verifiable structured output.

        Returns:
            Content with only code blocks, or empty string (no code blocks -> discard entire message)
        """
        code_blocks = self._CODE_BLOCK_PATTERN.findall(content)
        if code_blocks:
            return "\n\n".join(code_blocks)
        return ""

    def _tail_messages(
        self, messages: List[Dict], budget: int, tokenizer,
        model_tier: str = "medium",
    ) -> List[Dict]:
        """Take messages from tail, User-First filling (anti-pollution)

        Based on MIT paper findings: user messages are context backbone, assistant messages are annotations.
        1. First fill in tail user messages (backbone, must preserve)
        2. Then insert assistant messages back at corresponding positions (annotations, within budget)
        3. Maintain original message order

        model_tier filtering:
        - weak:   assistant messages keep only code blocks (aggressive filtering)
        - medium: full assistant messages (current behavior)
        - strong: full assistant messages (current behavior)
        """
        if not messages:
            return []

        # Scan from tail to determine candidate range
        candidate_indices = []
        scan_budget = 0
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = self._count_tokens([messages[i]], tokenizer)
            # Widen scan range to 1.5x budget (since we'll filter assistant later)
            if scan_budget + msg_tokens > budget * 1.5:
                break
            candidate_indices.insert(0, i)
            scan_budget += msg_tokens

        if not candidate_indices:
            # Can't fit even one, at least keep the last message
            return [messages[-1]]

        # Pass 1: fill in user messages
        user_filled = []
        used = 0
        for i in candidate_indices:
            msg = messages[i]
            if msg.get("role") == "user":
                msg_tokens = self._count_tokens([msg], tokenizer)
                if used + msg_tokens <= budget:
                    user_filled.append((i, msg))
                    used += msg_tokens

        # Pass 2: supplement assistant messages (maintain order)
        all_filled = list(user_filled)  # copy
        for i in candidate_indices:
            msg = messages[i]
            if msg.get("role") != "user":
                actual_msg = msg
                if model_tier == "weak" and msg.get("role") == "assistant":
                    # Weak model: assistant messages keep only code blocks
                    stripped = self._strip_assistant_to_artifacts(
                        str(msg.get("content", ""))
                    )
                    if not stripped:
                        continue  # No code blocks -> discard entire message
                    actual_msg = {**msg, "content": stripped}
                msg_tokens = self._count_tokens([actual_msg], tokenizer)
                if used + msg_tokens <= budget:
                    all_filled.append((i, actual_msg))
                    used += msg_tokens

        # Sort by original order
        all_filled.sort(key=lambda x: x[0])
        return [msg for _, msg in all_filled]

    def _fit_to_budget(
        self, messages: List[Dict], budget: int, tokenizer
    ) -> List[Dict]:
        """Keep messages from the beginning, not exceeding budget tokens"""
        kept = []
        used = 0
        for msg in messages:
            msg_tokens = self._count_tokens([msg], tokenizer)
            if used + msg_tokens > budget:
                break
            kept.append(msg)
            used += msg_tokens
        return kept

    # ========== P4: Cross-session Long-term Memory ==========

    def _extract_files_from_messages(self, messages: List[Dict]) -> Set[str]:
        """Extract file names from message list"""
        files: Set[str] = set()
        for msg in messages:
            content = str(msg.get("content", ""))
            for match in self._FILE_PATTERN.finditer(content):
                filepath = match.group(1)
                filename = filepath.split("/")[-1]
                files.add(filename)
        return files

    def _promote_to_long_term(
        self,
        segment: TopicSegment,
        summary: str,
        conversation_id: str,
    ) -> bool:
        """Promote work segment to long_term_memories

        Extract key_files -> compute memory_key -> INSERT OR REPLACE (7 day TTL)

        Returns:
            Whether write succeeded
        """
        key_files = self._extract_files_from_messages(segment.messages)
        if not key_files:
            return False

        sorted_files = sorted(key_files)
        memory_key = hashlib.sha256(
            "|".join(sorted_files).encode()
        ).hexdigest()[:16]

        key_files_json = json.dumps(sorted_files, ensure_ascii=False)
        expires_at = (
            datetime.utcnow() + timedelta(days=self.LTM_TTL_DAYS)
        ).strftime("%Y-%m-%d %H:%M:%S")

        conn = _sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO long_term_memories
                (memory_key, key_files, summary, conversation_id,
                 access_count, updated_at, expires_at)
            VALUES (?, ?, ?, ?, COALESCE(
                (SELECT access_count FROM long_term_memories WHERE memory_key = ?), 0
            ), CURRENT_TIMESTAMP, ?)
            """,
            (memory_key, key_files_json, summary, conversation_id,
             memory_key, expires_at),
        )

        conn.commit()
        conn.close()

        logger.debug(
            f"[ConvStore] LTM promoted: key={memory_key[:8]}... | "
            f"files={sorted_files} | TTL={self.LTM_TTL_DAYS}d"
        )
        return True

    def _recall_long_term(
        self, messages: List[Dict], limit: int = 3
    ) -> List[Dict]:
        """Recall cross-session memory by file name matching

        Extract key_files from current messages, query long_term_memories for
        records with intersection, sort by updated_at DESC, take top-N,
        return system message list.

        Returns:
            System messages for injection (may be empty)
        """
        current_files = self._extract_files_from_messages(messages)
        if not current_files:
            return []

        conn = _sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Query each filename with LIKE, collect matching memory_keys
        matched_keys: Set[str] = set()
        for filename in current_files:
            cursor.execute(
                "SELECT memory_key FROM long_term_memories "
                "WHERE key_files LIKE ? AND expires_at > ?",
                (f"%{filename}%", now),
            )
            for row in cursor.fetchall():
                matched_keys.add(row[0])

        if not matched_keys:
            conn.close()
            return []

        # Sort by updated_at DESC, take top-N
        placeholders = ",".join("?" for _ in matched_keys)
        cursor.execute(
            f"SELECT memory_key, key_files, summary FROM long_term_memories "
            f"WHERE memory_key IN ({placeholders}) AND expires_at > ? "
            f"ORDER BY updated_at DESC LIMIT ?",
            (*matched_keys, now, limit),
        )
        rows = cursor.fetchall()

        # Update access_count
        for row in rows:
            cursor.execute(
                "UPDATE long_term_memories SET access_count = access_count + 1 "
                "WHERE memory_key = ?",
                (row[0],),
            )
        conn.commit()
        conn.close()

        # Build system messages
        result = []
        for memory_key, key_files_json, summary in rows:
            files_list = json.loads(key_files_json)
            files_str = ", ".join(files_list)
            result.append({
                "role": "system",
                "content": f"[Prior session context] Previous work on {files_str}: {summary}",
            })

        return result

    def cleanup_expired_ltm(self) -> int:
        """Clean up expired long-term memories

        Returns:
            Number of deleted rows
        """
        conn = _sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "DELETE FROM long_term_memories WHERE expires_at < ?", (now,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted > 0:
            logger.info(f"[ConvStore] Cleaned expired long-term memories: {deleted}")

        return deleted
