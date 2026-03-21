# SPDX-License-Identifier: Apache-2.0
"""
Background scheduler for automatic MF Router incremental training.

Periodically checks for new labeled routing data, runs training,
validates the new checkpoint, and hot-deploys if improved.

Lifecycle: start() -> background loop -> stop()
Same pattern as ProcessMemoryEnforcer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("omlx.cloud.auto_trainer")


class AutoTrainer:
    """Background scheduler for automatic MF Router incremental training.

    Periodically checks for new labeled routing data, runs training,
    validates the new checkpoint, and hot-deploys if improved.

    Lifecycle: start() -> background loop -> stop()
    Same pattern as ProcessMemoryEnforcer.
    """

    def __init__(
        self,
        interval_hours: float,
        min_pairs: int,
        mix_ratio: float,
        mf_router: Any,  # MFRouter instance
        routing_store: Any,  # RoutingStore instance
        preference_labeler: Any,  # PreferenceLabeler instance
        arena_embeddings_path: str = "models/mf-router/cache/gemini_embeddings.npy",
        arena_metadata_path: str = "models/mf-router/cache/gemini_metadata.npz",
    ):
        self._interval_hours = interval_hours
        self._min_pairs = min_pairs
        self._mix_ratio = mix_ratio
        self._mf_router = mf_router
        self._routing_store = routing_store
        self._preference_labeler = preference_labeler
        self._arena_embeddings_path = arena_embeddings_path
        self._arena_metadata_path = arena_metadata_path
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_run_time: float = 0.0
        self._cycle_count: int = 0

    def start(self) -> None:
        """Start the background training loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._training_loop())
        logger.info(
            "[AutoTrainer] started | interval=%.1fh | min_pairs=%d | mix_ratio=%.2f",
            self._interval_hours,
            self._min_pairs,
            self._mix_ratio,
        )

    async def stop(self) -> None:
        """Stop the background training loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[AutoTrainer] stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result

    @property
    def status(self) -> Dict[str, Any]:
        """Return current status for API endpoints."""
        return {
            "running": self._running,
            "interval_hours": self._interval_hours,
            "min_pairs": self._min_pairs,
            "mix_ratio": self._mix_ratio,
            "cycle_count": self._cycle_count,
            "last_run_time": self._last_run_time,
            "last_result": self._last_result,
        }

    async def _training_loop(self) -> None:
        """Main background loop. Sleeps for interval, then runs a training cycle."""
        # Wait for initial interval before first run
        # (let the server collect some data first)
        while self._running:
            try:
                await asyncio.sleep(self._interval_hours * 3600)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            try:
                result = await self._run_cycle()
                self._last_result = result
                self._last_run_time = time.time()
                self._cycle_count += 1
            except Exception as exc:
                logger.error("[AutoTrainer] cycle failed: %s", exc, exc_info=True)
                self._last_result = {"status": "error", "error": str(exc)}

    async def _run_cycle(self) -> Dict[str, Any]:
        """Execute one training cycle: label -> check -> train -> deploy."""
        logger.info("[AutoTrainer] starting training cycle #%d", self._cycle_count + 1)

        # Step 1: Label new data
        label_result = await self._preference_labeler.batch_label()
        logger.info("[AutoTrainer] labeling: %s", label_result)

        # Step 2: Check data volume
        stats = await self._routing_store.get_training_stats()
        labeled_count = stats.get("labeled_with_embedding", 0)

        if labeled_count < self._min_pairs:
            logger.info(
                "[AutoTrainer] only %d labeled pairs (need %d), skipping training",
                labeled_count,
                self._min_pairs,
            )
            return {
                "status": "skipped",
                "reason": f"insufficient data ({labeled_count}/{self._min_pairs})",
                "label_result": label_result,
            }

        # Step 3: Train in thread pool (CPU-intensive)
        from .incremental_trainer import TrainingConfig, run_training_pipeline

        checkpoint_path = getattr(self._mf_router, "_checkpoint_path", "")
        if not checkpoint_path:
            checkpoint_path = "models/mf-router/model_gemini.safetensors"

        output_path = checkpoint_path.replace(".safetensors", "_auto.safetensors")

        config = TrainingConfig(
            routing_db=self._routing_store._db_path,
            arena_embeddings=self._arena_embeddings_path,
            arena_metadata=self._arena_metadata_path,
            old_checkpoint=checkpoint_path,
            output=output_path,
            mix_ratio=self._mix_ratio,
            min_pairs=self._min_pairs,
        )

        logger.info("[AutoTrainer] starting training in thread pool...")
        result = await asyncio.to_thread(run_training_pipeline, config)

        result_dict: Dict[str, Any] = {
            "status": result.status,
            "old_accuracy": result.old_accuracy,
            "new_accuracy": result.new_accuracy,
            "production_pairs": result.production_pairs,
            "arena_pairs": result.arena_pairs,
            "total_pairs": result.total_pairs,
            "elapsed_seconds": result.elapsed_seconds,
            "output_path": result.output_path,
            "error": result.error,
            "label_result": label_result,
        }

        # Step 4: Hot-deploy if accepted
        if result.status == "accepted" and result.output_path:
            try:
                success = self._mf_router.hot_reload_checkpoint(result.output_path)
                if success:
                    logger.info(
                        "[AutoTrainer] deployed new checkpoint: acc %.3f -> %.3f (+%.3f)",
                        result.old_accuracy,
                        result.new_accuracy,
                        result.new_accuracy - result.old_accuracy,
                    )
                    result_dict["deployed"] = True
                else:
                    logger.warning("[AutoTrainer] hot_reload_checkpoint returned False")
                    result_dict["deployed"] = False
            except Exception as exc:
                logger.error("[AutoTrainer] deploy failed: %s", exc)
                result_dict["deployed"] = False
                result_dict["deploy_error"] = str(exc)
        else:
            logger.info("[AutoTrainer] training result: %s", result.status)

        return result_dict

    async def trigger_now(self) -> Dict[str, Any]:
        """Manually trigger one training cycle (called from API endpoint).

        Returns:
            Training result dict.
        """
        result = await self._run_cycle()
        self._last_result = result
        self._last_run_time = time.time()
        self._cycle_count += 1
        return result
