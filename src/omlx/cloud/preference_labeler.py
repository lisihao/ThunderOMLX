# SPDX-License-Identifier: Apache-2.0
"""Batch preference labeler for MF Router incremental training.

Reads routing decisions from RoutingStore and assigns pair_label
based on four implicit signals (no human annotation needed).

Signal Types
------------
- **strong_wins**: Cascade escalation -- local was tried but escalated to cloud.
- **weak_enough**: Direct local success -- local handled it without escalation.
- **model_loses**: Model failure -- execution resulted in error or timeout.
- **mf_disagree_local_ok**: MF Router predicted cloud but local succeeded.

Usage::

    labeler = PreferenceLabeler(db_path="~/.omlx/routing.db")
    result = await labeler.batch_label()
    stats = await labeler.get_label_stats()
"""

import logging
import time
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger("omlx.cloud.preference_labeler")


class PreferenceLabeler:
    """Assign preference labels to routing decisions for MF training."""

    def __init__(
        self,
        db_path: str,
        mf_threshold: float = 0.77,
    ):
        self._db_path = db_path
        self._mf_threshold = mf_threshold

    async def batch_label(
        self,
        since_timestamp: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Label unlabeled routing decisions.

        Args:
            since_timestamp: Only label decisions after this time.
                If None, label all unlabeled.
            dry_run: If True, don't write labels, just return stats.

        Returns:
            Dict with label counts and timing.
        """
        start = time.time()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            # 1. Label cascade escalations -> strong_wins
            strong_wins = await self._label_escalations(db, since_timestamp, dry_run)

            # 2. Label direct local success -> weak_enough
            weak_enough = await self._label_direct_success(db, since_timestamp, dry_run)

            # 3. Label failures -> model_loses
            model_loses = await self._label_failures(db, since_timestamp, dry_run)

            # 4. Label MF disagreement -> mf_disagree_local_ok
            mf_disagree = await self._label_mf_disagree(db, since_timestamp, dry_run)

            if not dry_run:
                await db.commit()

        elapsed = time.time() - start
        result = {
            "strong_wins": strong_wins,
            "weak_enough": weak_enough,
            "model_loses": model_loses,
            "mf_disagree_local_ok": mf_disagree,
            "total_labeled": strong_wins + weak_enough + model_loses + mf_disagree,
            "elapsed_seconds": round(elapsed, 3),
            "dry_run": dry_run,
        }
        logger.info("[PreferenceLabeler] batch_label result: %s", result)
        return result

    # ------------------------------------------------------------------
    # Private labeling methods -- each targets one signal type
    # ------------------------------------------------------------------

    async def _label_escalations(
        self, db: aiosqlite.Connection, since: Optional[float], dry_run: bool
    ) -> int:
        """Label escalated decisions as strong_wins."""
        base_where = (
            "pair_label IS NULL "
            "AND escalated_from_id IS NOT NULL "
            "AND outcome_status = 'success'"
        )
        params: List[Any] = []
        if since is not None:
            base_where += " AND timestamp > ?"
            params.append(since)

        if dry_run:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM routing_decisions WHERE {base_where}",
                params,
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

        cursor = await db.execute(
            f"UPDATE routing_decisions SET pair_label = 'strong_wins' WHERE {base_where}",
            params,
        )
        return cursor.rowcount

    async def _label_direct_success(
        self, db: aiosqlite.Connection, since: Optional[float], dry_run: bool
    ) -> int:
        """Label successful local routing as weak_enough."""
        base_where = (
            "pair_label IS NULL "
            "AND target = 'local' "
            "AND outcome_status = 'success' "
            "AND escalated_from_id IS NULL"
        )
        params: List[Any] = []
        if since is not None:
            base_where += " AND timestamp > ?"
            params.append(since)

        if dry_run:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM routing_decisions WHERE {base_where}",
                params,
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

        cursor = await db.execute(
            f"UPDATE routing_decisions SET pair_label = 'weak_enough' WHERE {base_where}",
            params,
        )
        return cursor.rowcount

    async def _label_failures(
        self, db: aiosqlite.Connection, since: Optional[float], dry_run: bool
    ) -> int:
        """Label failed executions as model_loses."""
        base_where = (
            "pair_label IS NULL "
            "AND outcome_status IN ('error', 'timeout')"
        )
        params: List[Any] = []
        if since is not None:
            base_where += " AND timestamp > ?"
            params.append(since)

        if dry_run:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM routing_decisions WHERE {base_where}",
                params,
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

        cursor = await db.execute(
            f"UPDATE routing_decisions SET pair_label = 'model_loses' WHERE {base_where}",
            params,
        )
        return cursor.rowcount

    async def _label_mf_disagree(
        self, db: aiosqlite.Connection, since: Optional[float], dry_run: bool
    ) -> int:
        """Label MF Router disagreements where local succeeded anyway."""
        base_where = (
            "pair_label IS NULL "
            "AND mf_win_rate IS NOT NULL "
            "AND mf_win_rate > ? "
            "AND target = 'local' "
            "AND outcome_status = 'success'"
        )
        params: List[Any] = [self._mf_threshold]
        if since is not None:
            base_where += " AND timestamp > ?"
            params.append(since)

        if dry_run:
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM routing_decisions WHERE {base_where}",
                params,
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

        cursor = await db.execute(
            f"UPDATE routing_decisions SET pair_label = 'mf_disagree_local_ok' WHERE {base_where}",
            params,
        )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_label_stats(self) -> Dict[str, int]:
        """Get distribution of pair_label values."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT pair_label, COUNT(*) FROM routing_decisions "
                "WHERE pair_label IS NOT NULL GROUP BY pair_label"
            )
            rows = await cursor.fetchall()
            stats: Dict[str, int] = {row[0]: row[1] for row in rows}

            # Also count unlabeled with outcome
            cursor = await db.execute(
                "SELECT COUNT(*) FROM routing_decisions "
                "WHERE pair_label IS NULL AND outcome_status != 'pending'"
            )
            row = await cursor.fetchone()
            stats["unlabeled_with_outcome"] = row[0] if row else 0

            return stats


# ----------------------------------------------------------------------
# Training format conversion
# ----------------------------------------------------------------------


def convert_to_training_format(
    pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert labeled routing decisions to MF Router training format.

    Each preference pair maps to a (prompt, model_a, model_b, winner) tuple
    with the pre-computed embedding vector.

    Args:
        pairs: Rows from routing_store.get_training_pairs().
            Each has: embedding (bytes), pair_label, target, model, mf_win_rate.

    Returns:
        List of training examples with format::

            {
                "embedding": np.ndarray (3072,),
                "model_a_id": int,   # strong model
                "model_b_id": int,   # weak model
                "winner": str,       # "model_a" or "model_b"
            }
    """
    import numpy as np

    # MF Router model IDs (from train_mf_gemini.py)
    STRONG_ID = 21  # Default strong model index
    WEAK_ID = 36  # Default weak model index

    training_data: List[Dict[str, Any]] = []
    skipped = 0

    for pair in pairs:
        embedding_bytes = pair.get("embedding")
        if not embedding_bytes or len(embedding_bytes) < 100:
            skipped += 1
            continue

        try:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        except Exception:
            skipped += 1
            continue

        label = pair.get("pair_label", "")

        if label == "strong_wins":
            winner = "model_a"  # strong model wins
        elif label == "weak_enough":
            winner = "model_b"  # weak model is sufficient
        elif label == "model_loses":
            # If target was local, weak model failed -> strong wins
            # If target was cloud, strong model failed -> skip (ambiguous)
            if pair.get("target") == "local":
                winner = "model_a"
            else:
                skipped += 1
                continue
        elif label == "mf_disagree_local_ok":
            winner = "model_b"  # MF said cloud but local was fine
        else:
            skipped += 1
            continue

        training_data.append({
            "embedding": embedding,
            "model_a_id": STRONG_ID,
            "model_b_id": WEAK_ID,
            "winner": winner,
        })

    logger.info(
        "[convert_to_training_format] %d training pairs, %d skipped",
        len(training_data),
        skipped,
    )
    return training_data
