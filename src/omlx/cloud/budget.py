"""Budget control for OMLX cloud routing.

Enforces daily/monthly spending limits by querying the requests table.
Uses a short in-memory cache (60s TTL) to avoid hitting the DB on every request.

Configuration via config dict or environment variables:
    budget:
      daily_limit: 5.0
      monthly_limit: 100.0
      alert_threshold: 0.8
      action: "reject"        # reject | warn_only

Or override via environment variables:
    OMLX_BUDGET_DAILY=5.0
    OMLX_BUDGET_MONTHLY=100.0
"""

import os
import time
import sqlite3
import logging
from datetime import datetime, date
from typing import Optional, Dict

logger = logging.getLogger("omlx.cloud.budget")


class BudgetDBHelper:
    """Lightweight sqlite3 helper for budget spend queries.

    Expects a table ``requests`` with at least a ``cost`` REAL column and a
    ``created_at`` TEXT column (ISO-8601 datetime).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_table(self):
        """Create the requests table if it does not exist."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT,
                    cost REAL DEFAULT 0,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0,
                    status TEXT DEFAULT 'ok',
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def record_request(
        self,
        model: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        status: str = "ok",
    ) -> None:
        """Insert a request record into the budget database."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO requests (model, cost, input_tokens, output_tokens, latency_ms, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (model, cost, input_tokens, output_tokens, latency_ms, status),
            )
            conn.commit()
        finally:
            conn.close()

    def get_daily_spend(self) -> float:
        """Sum of cost for today (UTC)."""
        today_str = date.today().isoformat()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost), 0) AS total FROM requests WHERE date(created_at) = ?",
                (today_str,),
            ).fetchone()
            return float(row["total"]) if row else 0.0
        finally:
            conn.close()

    def get_monthly_spend(self) -> float:
        """Sum of cost for the current month (UTC)."""
        now = datetime.utcnow()
        month_start = f"{now.year}-{now.month:02d}-01"
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost), 0) AS total FROM requests WHERE created_at >= ?",
                (month_start,),
            ).fetchone()
            return float(row["total"]) if row else 0.0
        finally:
            conn.close()


class BudgetChecker:
    """Checks daily/monthly spend against configured limits."""

    def __init__(self, db_path: str, config: Optional[Dict] = None):
        """
        Args:
            db_path: Path to the sqlite3 database file.
            config: Optional budget configuration dict, e.g.
                    {"daily_limit": 5.0, "monthly_limit": 100.0,
                     "alert_threshold": 0.8, "action": "reject"}
        """
        self.db_helper = BudgetDBHelper(db_path)
        self.db_helper.ensure_table()

        budget_cfg: Dict = config or {}

        # Environment overrides take precedence
        self.daily_limit: float = float(
            os.getenv("OMLX_BUDGET_DAILY", budget_cfg.get("daily_limit", 0))
        )
        self.monthly_limit: float = float(
            os.getenv("OMLX_BUDGET_MONTHLY", budget_cfg.get("monthly_limit", 0))
        )
        self.alert_threshold: float = float(budget_cfg.get("alert_threshold", 0.8))
        self.action: str = budget_cfg.get("action", "reject")  # reject | warn_only

        # In-memory cache (avoids querying DB on every request)
        self._cache_ttl = 60  # seconds
        self._daily_cache: Optional[float] = None
        self._daily_cache_ts: float = 0
        self._monthly_cache: Optional[float] = None
        self._monthly_cache_ts: float = 0

        logger.info(
            f"[Budget] daily_limit=${self.daily_limit} monthly_limit=${self.monthly_limit} "
            f"alert={self.alert_threshold} action={self.action}"
        )

    @property
    def enabled(self) -> bool:
        """Budget enforcement is enabled if at least one limit > 0."""
        return self.daily_limit > 0 or self.monthly_limit > 0

    def get_daily_spend(self) -> float:
        """Cached daily spend lookup."""
        now = time.time()
        if self._daily_cache is not None and (now - self._daily_cache_ts) < self._cache_ttl:
            return self._daily_cache
        self._daily_cache = self.db_helper.get_daily_spend()
        self._daily_cache_ts = now
        return self._daily_cache

    def get_monthly_spend(self) -> float:
        """Cached monthly spend lookup."""
        now = time.time()
        if self._monthly_cache is not None and (now - self._monthly_cache_ts) < self._cache_ttl:
            return self._monthly_cache
        self._monthly_cache = self.db_helper.get_monthly_spend()
        self._monthly_cache_ts = now
        return self._monthly_cache

    def invalidate_cache(self):
        """Force refresh on next check (call after recording a cost)."""
        self._daily_cache = None
        self._monthly_cache = None

    def check(self) -> Dict:
        """Check current spend against limits.

        Returns:
            {
                "allowed": bool,
                "daily_spend": float,
                "daily_limit": float,
                "daily_pct": float,         # 0-1
                "monthly_spend": float,
                "monthly_limit": float,
                "monthly_pct": float,        # 0-1
                "reason": str | None,        # set when not allowed
                "warning": str | None,       # set when above alert threshold
            }
        """
        if not self.enabled:
            return {"allowed": True, "reason": None, "warning": None}

        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()

        daily_pct = (daily_spend / self.daily_limit) if self.daily_limit > 0 else 0
        monthly_pct = (monthly_spend / self.monthly_limit) if self.monthly_limit > 0 else 0

        result = {
            "allowed": True,
            "daily_spend": round(daily_spend, 6),
            "daily_limit": self.daily_limit,
            "daily_pct": round(daily_pct, 4),
            "monthly_spend": round(monthly_spend, 6),
            "monthly_limit": self.monthly_limit,
            "monthly_pct": round(monthly_pct, 4),
            "reason": None,
            "warning": None,
        }

        # Check daily limit
        if self.daily_limit > 0 and daily_spend >= self.daily_limit:
            msg = f"Daily budget exceeded: ${daily_spend:.4f} / ${self.daily_limit:.2f}"
            if self.action == "reject":
                result["allowed"] = False
                result["reason"] = msg
            else:
                result["warning"] = msg
            logger.warning(f"[Budget] {msg}")

        # Check monthly limit
        if self.monthly_limit > 0 and monthly_spend >= self.monthly_limit:
            msg = f"Monthly budget exceeded: ${monthly_spend:.4f} / ${self.monthly_limit:.2f}"
            if self.action == "reject":
                result["allowed"] = False
                result["reason"] = msg
            else:
                result["warning"] = msg
            logger.warning(f"[Budget] {msg}")

        # Alert threshold warnings
        if result["allowed"] and result["reason"] is None:
            if daily_pct >= self.alert_threshold and self.daily_limit > 0:
                result["warning"] = (
                    f"Daily spend at {daily_pct:.0%}: ${daily_spend:.4f} / ${self.daily_limit:.2f}"
                )
                logger.info(f"[Budget] warning: {result['warning']}")
            elif monthly_pct >= self.alert_threshold and self.monthly_limit > 0:
                result["warning"] = (
                    f"Monthly spend at {monthly_pct:.0%}: ${monthly_spend:.4f} / ${self.monthly_limit:.2f}"
                )
                logger.info(f"[Budget] warning: {result['warning']}")

        return result

    def get_budget_info(self) -> Dict:
        """Return budget status for dashboard display (no enforcement)."""
        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()
        return {
            "daily_spend": round(daily_spend, 6),
            "daily_limit": self.daily_limit,
            "daily_pct": round((daily_spend / self.daily_limit), 4) if self.daily_limit > 0 else 0,
            "monthly_spend": round(monthly_spend, 6),
            "monthly_limit": self.monthly_limit,
            "monthly_pct": round((monthly_spend / self.monthly_limit), 4) if self.monthly_limit > 0 else 0,
            "alert_threshold": self.alert_threshold,
            "action": self.action,
            "enabled": self.enabled,
        }
