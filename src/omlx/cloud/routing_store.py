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
        logger.info("RoutingStore initialized: %s", self._db_path)

    async def record_decision(self, decision_dict: Dict[str, Any]) -> None:
        """Persist a routing decision.

        Args:
            decision_dict: Serialised RoutingDecision (from dataclasses.asdict).
        """
        if not self._db:
            return
        await self._db.execute(
            """
            INSERT INTO routing_decisions
            (timestamp, conversation_id, task_type, coding_subtask, complexity,
             target, model, reason, confidence, was_escalated, session_pinned, tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        await self._db.commit()

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
