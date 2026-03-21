# SPDX-License-Identifier: Apache-2.0
"""RoutingStore - SQLite-backed persistence for routing decisions.

Stores every routing decision made by IntelligentRouter and provides
aggregated analytics and cost-savings estimates.

Usage::

    store = RoutingStore()
    await store.initialize()
    await store.record_decision({...})
    analytics = await store.get_analytics(hours=24)
    savings = await store.get_cost_savings(hours=24)
    await store.close()
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger("omlx.cloud.routing_store")


class RoutingStore:
    """SQLite-backed persistence for routing decisions."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or os.path.expanduser("~/.omlx/routing.db")
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Create connection and ensure table exists."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS routing_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                conversation_id TEXT,
                task_type TEXT,
                coding_subtask TEXT,
                complexity TEXT,
                target TEXT NOT NULL,
                model TEXT NOT NULL,
                reason TEXT,
                confidence REAL,
                was_escalated INTEGER DEFAULT 0,
                latency_ms REAL,
                token_count INTEGER,
                cost_usd REAL,
                session_pinned INTEGER DEFAULT 0,
                tier INTEGER
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_timestamp
            ON routing_decisions(timestamp)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_target
            ON routing_decisions(target)
        """)
        await self._db.commit()

        # --- Incremental training feedback columns (idempotent migration) ---
        cursor = await self._db.execute(
            "PRAGMA table_info(routing_decisions)"
        )
        existing_cols = {row[1] for row in await cursor.fetchall()}

        new_columns = [
            ("decision_id", "TEXT UNIQUE"),
            ("prompt_text", "TEXT"),
            ("embedding", "BLOB"),
            ("mf_win_rate", "REAL"),
            ("outcome_status", "TEXT DEFAULT 'pending'"),
            ("outcome_latency_ms", "REAL"),
            ("outcome_input_tokens", "INTEGER"),
            ("outcome_output_tokens", "INTEGER"),
            ("outcome_cost_usd", "REAL"),
            ("outcome_error", "TEXT"),
            ("outcome_timestamp", "REAL"),
            ("escalated_from_id", "TEXT"),
            ("pair_label", "TEXT"),
        ]
        for col_name, col_def in new_columns:
            if col_name not in existing_cols:
                try:
                    await self._db.execute(
                        f"ALTER TABLE routing_decisions ADD COLUMN {col_name} {col_def}"
                    )
                except Exception:
                    pass  # column already exists in a concurrent migration

        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_decision_id
            ON routing_decisions(decision_id)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_routing_pair_label
            ON routing_decisions(pair_label)
        """)
        await self._db.commit()
        logger.info("RoutingStore initialized: %s", self._db_path)

    async def record_decision(
        self,
        decision_dict: Dict[str, Any],
        *,
        prompt_text: str = "",
        embedding_bytes: bytes = b"",
        mf_win_rate: Optional[float] = None,
    ) -> Optional[str]:
        """Persist a routing decision.

        Args:
            decision_dict: Serialised RoutingDecision (from dataclasses.asdict).
            prompt_text: Last user message for training pair extraction.
            embedding_bytes: Pre-computed embedding as raw bytes (numpy.tobytes()).
            mf_win_rate: MF Router predicted win rate.

        Returns:
            The generated decision_id (16-char hex), or None if DB is unavailable.
        """
        if not self._db:
            return None

        decision_id = uuid.uuid4().hex[:16]

        await self._db.execute(
            """
            INSERT INTO routing_decisions
            (timestamp, conversation_id, task_type, coding_subtask, complexity,
             target, model, reason, confidence, was_escalated, session_pinned, tier,
             decision_id, prompt_text, embedding, mf_win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_dict.get("timestamp", time.time()),
                decision_dict.get("conversation_id"),
                decision_dict.get("task_type", ""),
                decision_dict.get("coding_subtask", ""),
                decision_dict.get("complexity", ""),
                decision_dict.get("target", ""),
                decision_dict.get("model", ""),
                decision_dict.get("reason", ""),
                decision_dict.get("confidence", 1.0),
                int(decision_dict.get("was_escalated", False)),
                int(decision_dict.get("session_pinned", False)),
                decision_dict.get("tier", 1),
                decision_id,
                prompt_text,
                embedding_bytes if embedding_bytes else None,
                mf_win_rate,
            ),
        )
        await self._db.commit()
        return decision_id

    async def record_outcome(
        self,
        decision_id: str,
        status: str = "success",
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        error: str = "",
    ) -> None:
        """Attach outcome feedback to an existing routing decision.

        Args:
            decision_id: The decision_id returned by record_decision().
            status: One of 'success', 'error', 'timeout'.
            latency_ms: End-to-end latency in milliseconds.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.
            cost_usd: Actual cost in USD.
            error: Error message (empty string if success).
        """
        if not self._db:
            return
        await self._db.execute(
            """
            UPDATE routing_decisions
            SET outcome_status = ?,
                outcome_latency_ms = ?,
                outcome_input_tokens = ?,
                outcome_output_tokens = ?,
                outcome_cost_usd = ?,
                outcome_error = ?,
                outcome_timestamp = ?
            WHERE decision_id = ?
            """,
            (
                status,
                latency_ms,
                input_tokens,
                output_tokens,
                cost_usd,
                error,
                time.time(),
                decision_id,
            ),
        )
        await self._db.commit()

    async def record_escalation(
        self, original_decision_id: str, escalated_decision_id: str
    ) -> None:
        """Link an escalation to its original routing decision.

        Sets escalated_from_id on the escalated row and marks the original
        row with pair_label='strong_wins' (the weak model was not enough).

        Args:
            original_decision_id: decision_id of the first (weak) attempt.
            escalated_decision_id: decision_id of the escalated (strong) attempt.
        """
        if not self._db:
            return
        await self._db.execute(
            """
            UPDATE routing_decisions
            SET escalated_from_id = ?
            WHERE decision_id = ?
            """,
            (original_decision_id, escalated_decision_id),
        )
        await self._db.execute(
            """
            UPDATE routing_decisions
            SET pair_label = 'strong_wins'
            WHERE decision_id = ?
            """,
            (original_decision_id,),
        )
        await self._db.commit()

    async def get_training_pairs(
        self, since_timestamp: float = 0, limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """Retrieve labeled training pairs for incremental MF Router training.

        Args:
            since_timestamp: Only return rows with timestamp greater than this.
            limit: Maximum number of rows to return.

        Returns:
            List of dicts with decision_id, prompt_text, embedding (raw bytes),
            mf_win_rate, target, model, pair_label, outcome_status.
        """
        if not self._db:
            return []
        cursor = await self._db.execute(
            """
            SELECT decision_id, prompt_text, embedding, mf_win_rate,
                   target, model, pair_label, outcome_status
            FROM routing_decisions
            WHERE pair_label IS NOT NULL
              AND embedding IS NOT NULL
              AND length(embedding) > 0
              AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (since_timestamp, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "decision_id": row[0],
                "prompt_text": row[1],
                "embedding": row[2],
                "mf_win_rate": row[3],
                "target": row[4],
                "model": row[5],
                "pair_label": row[6],
                "outcome_status": row[7],
            }
            for row in rows
        ]

    async def get_training_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the training feedback store.

        Returns:
            Dict with total_decisions, with_outcomes, with_embeddings,
            labeled_pairs (per-label counts), oldest_labeled, newest_labeled.
        """
        if not self._db:
            return {}

        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM routing_decisions"
        )
        total_decisions = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM routing_decisions WHERE outcome_status != 'pending'"
        )
        with_outcomes = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM routing_decisions "
            "WHERE embedding IS NOT NULL AND length(embedding) > 0"
        )
        with_embeddings = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT pair_label, COUNT(*) FROM routing_decisions "
            "WHERE pair_label IS NOT NULL GROUP BY pair_label"
        )
        labeled_pairs = {row[0]: row[1] for row in await cursor.fetchall()}

        cursor = await self._db.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM routing_decisions "
            "WHERE pair_label IS NOT NULL"
        )
        ts_row = await cursor.fetchone()
        oldest_labeled = ts_row[0] if ts_row else None
        newest_labeled = ts_row[1] if ts_row else None

        return {
            "total_decisions": total_decisions,
            "with_outcomes": with_outcomes,
            "with_embeddings": with_embeddings,
            "labeled_pairs": labeled_pairs,
            "oldest_labeled": oldest_labeled,
            "newest_labeled": newest_labeled,
        }

    async def get_analytics(self, hours: float = 24.0) -> Dict[str, Any]:
        """Return aggregated routing analytics for the given time window.

        Args:
            hours: Look-back window in hours (default 24).

        Returns:
            Dict with total_decisions, per_target, per_task_type,
            per_model, escalation_rate, and recent_decisions.
        """
        if not self._db:
            return {}

        since = time.time() - hours * 3600

        # Total decisions
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM routing_decisions WHERE timestamp > ?",
            (since,),
        )
        total = (await cursor.fetchone())[0]

        # Per target
        cursor = await self._db.execute(
            """SELECT target, COUNT(*) FROM routing_decisions
               WHERE timestamp > ? GROUP BY target""",
            (since,),
        )
        per_target = {row[0]: row[1] for row in await cursor.fetchall()}

        # Per task type
        cursor = await self._db.execute(
            """SELECT task_type, COUNT(*) FROM routing_decisions
               WHERE timestamp > ? GROUP BY task_type""",
            (since,),
        )
        per_task_type = {row[0]: row[1] for row in await cursor.fetchall()}

        # Per model (top 10)
        cursor = await self._db.execute(
            """SELECT model, COUNT(*) FROM routing_decisions
               WHERE timestamp > ? GROUP BY model
               ORDER BY COUNT(*) DESC LIMIT 10""",
            (since,),
        )
        per_model = {row[0]: row[1] for row in await cursor.fetchall()}

        # Escalation count
        cursor = await self._db.execute(
            """SELECT COUNT(*) FROM routing_decisions
               WHERE timestamp > ? AND was_escalated = 1""",
            (since,),
        )
        escalated = (await cursor.fetchone())[0]

        # Recent decisions (most recent 50)
        cursor = await self._db.execute(
            """SELECT timestamp, conversation_id, task_type, coding_subtask,
                      target, model, reason, confidence, was_escalated,
                      session_pinned
               FROM routing_decisions WHERE timestamp > ?
               ORDER BY timestamp DESC LIMIT 50""",
            (since,),
        )
        recent: List[Dict[str, Any]] = []
        for row in await cursor.fetchall():
            recent.append({
                "timestamp": row[0],
                "conversation_id": row[1],
                "task_type": row[2],
                "coding_subtask": row[3],
                "target": row[4],
                "model": row[5],
                "reason": row[6],
                "confidence": row[7],
                "was_escalated": bool(row[8]),
                "session_pinned": bool(row[9]),
            })

        return {
            "hours": hours,
            "total_decisions": total,
            "per_target": per_target,
            "per_task_type": per_task_type,
            "per_model": per_model,
            "escalation_rate": escalated / max(total, 1),
            "escalated_count": escalated,
            "recent_decisions": recent,
        }

    async def get_cost_savings(self, hours: float = 24.0) -> Dict[str, Any]:
        """Estimate cost savings vs all-cloud routing.

        Compares actual routing costs (local=free, economy=cheap, premium=full)
        against a hypothetical all-premium-cloud baseline.

        Args:
            hours: Look-back window in hours (default 24).

        Returns:
            Dict with request counts, cost estimates, and savings percentage.
        """
        if not self._db:
            return {}

        since = time.time() - hours * 3600

        cursor = await self._db.execute(
            """SELECT target, COUNT(*) FROM routing_decisions
               WHERE timestamp > ? GROUP BY target""",
            (since,),
        )
        counts = {row[0]: row[1] for row in await cursor.fetchall()}

        local = counts.get("local", 0)
        economy = counts.get("cloud_economy", 0)
        premium = counts.get("cloud_premium", 0)
        total = local + economy + premium

        # Rough cost estimates per request (USD)
        COST_LOCAL = 0.0
        COST_ECONOMY = 0.002
        COST_PREMIUM = 0.01

        actual_cost = (
            local * COST_LOCAL
            + economy * COST_ECONOMY
            + premium * COST_PREMIUM
        )
        all_cloud_cost = total * COST_PREMIUM  # hypothetical: everything premium
        savings = all_cloud_cost - actual_cost

        return {
            "hours": hours,
            "total_requests": total,
            "local_requests": local,
            "cloud_economy_requests": economy,
            "cloud_premium_requests": premium,
            "estimated_actual_cost_usd": round(actual_cost, 4),
            "estimated_all_cloud_cost_usd": round(all_cloud_cost, 4),
            "estimated_savings_usd": round(savings, 4),
            "savings_percentage": round(
                savings / max(all_cloud_cost, 0.001) * 100, 1
            ),
        }

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
