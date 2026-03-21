"""Logprobs-based confidence checker for cascade escalation.

Determines whether a locally-generated response should be escalated to a
cloud model by analysing the per-token log probabilities produced during
generation.

The checker operates on a simple statistical model:

* **High confidence** (mean logprob > threshold_high) -- the local model is
  producing tokens it is very sure about.  No escalation needed.
* **Low confidence** (mean logprob < threshold_low) -- the local model is
  struggling.  Escalate to cloud.
* **In between** -- use task-type-specific thresholds to make the call.
  Safety-critical subtasks (debugging, security) use aggressive thresholds
  while others stay conservative.

The thresholds were chosen empirically:

* ``-0.5``  corresponds to ~61% average token probability -- quite confident.
* ``-2.0``  corresponds to ~13.5% average token probability -- very unsure.
* ``-1.0``  (task override) corresponds to ~37% -- moderate uncertainty.
* ``-1.5``  (task override) corresponds to ~22% -- noticeable uncertainty.

Usage::

    checker = ConfidenceChecker()
    if checker.check(logprobs, task_type="debugging"):
        # escalate to cloud
        ...
    analysis = checker.analyze(logprobs)
    # {"mean_logprob": -0.82, "confidence_level": "medium", ...}
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger("omlx.cloud.confidence_checker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD_HIGH: float = -0.5
_DEFAULT_THRESHOLD_LOW: float = -2.0

# Task-type specific thresholds used in the "medium zone" between high/low.
# More safety-critical tasks use a more aggressive (higher) threshold so they
# escalate sooner.
_TASK_THRESHOLDS: Dict[str, float] = {
    # Aggressive -- escalate even for moderate uncertainty
    "debugging": -1.0,
    "security": -1.0,
    # Moderate
    "refactor": -1.5,
    "architecture": -1.5,
}

# Tokens with logprob below this are considered padding / special tokens and
# are excluded from the analysis.
_SPECIAL_TOKEN_FLOOR: float = -100.0

# Tokens with logprob below this are counted as "low confidence" in the
# detailed analysis.
_LOW_CONFIDENCE_TOKEN_THRESHOLD: float = -3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float:
    """Convert a logprob value to a plain Python float.

    Handles ``float``, ``int``, ``mx.array`` (scalar), and ``numpy``
    scalar types transparently.

    Args:
        value: A numeric logprob value.

    Returns:
        A plain Python float.

    Raises:
        TypeError: If the value cannot be converted.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    # mx.array, numpy scalar, or anything with __float__
    return float(value)


def _filter_logprobs(raw: Sequence[Any]) -> List[float]:
    """Convert and filter a raw logprob sequence.

    Converts each element to float, then drops values below
    ``_SPECIAL_TOKEN_FLOOR`` (padding / special tokens) and non-finite
    values.

    Args:
        raw: Sequence of logprob values (float, mx.array, numpy, etc.).

    Returns:
        A filtered list of plain Python floats.
    """
    result: List[float] = []
    for v in raw:
        try:
            f = _to_float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            continue
        if f < _SPECIAL_TOKEN_FLOOR:
            continue
        result.append(f)
    return result


# ---------------------------------------------------------------------------
# ConfidenceChecker
# ---------------------------------------------------------------------------

class ConfidenceChecker:
    """Logprobs-based confidence checker for cascade escalation.

    Analyses per-token log probabilities to decide whether a local
    response is confident enough or should be escalated to a cloud model.

    This class is **thread-safe**: all mutable state is set once at
    construction and never modified afterwards.

    Args:
        threshold_high: Mean logprob above which the response is considered
            high confidence (no escalation).  Default ``-0.5``.
        threshold_low: Mean logprob below which the response is considered
            low confidence (always escalate).  Default ``-2.0``.
        task_overrides: Optional dict mapping task_type strings to custom
            thresholds used in the medium zone.
    """

    def __init__(
        self,
        threshold_high: float = _DEFAULT_THRESHOLD_HIGH,
        threshold_low: float = _DEFAULT_THRESHOLD_LOW,
        task_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low

        # Merge built-in task thresholds with caller overrides.
        self._task_thresholds: Dict[str, float] = dict(_TASK_THRESHOLDS)
        if task_overrides:
            self._task_thresholds.update(task_overrides)

        logger.info(
            "[__init__] ConfidenceChecker ready | high=%.2f low=%.2f "
            "task_overrides=%d",
            self._threshold_high,
            self._threshold_low,
            len(self._task_thresholds),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        logprobs: Sequence[Any],
        task_type: str = "",
    ) -> bool:
        """Determine whether a response needs cloud escalation.

        Args:
            logprobs: Per-token log probabilities from the local model.
            task_type: Optional task type (e.g. ``"debugging"``,
                ``"security"``) used for task-specific thresholds.

        Returns:
            ``True`` if the response should be escalated to a cloud model,
            ``False`` if the local response is confident enough.
        """
        values = _filter_logprobs(logprobs)

        if not values:
            logger.debug("[check] empty logprobs, skipping escalation")
            return False

        mean_lp = sum(values) / len(values)

        if mean_lp > self._threshold_high:
            logger.debug(
                "[check] high confidence (mean=%.3f > %.3f) -> no escalation",
                mean_lp, self._threshold_high,
            )
            return False

        if mean_lp < self._threshold_low:
            logger.debug(
                "[check] low confidence (mean=%.3f < %.3f) -> escalate",
                mean_lp, self._threshold_low,
            )
            return True

        # Medium zone -- use task-specific threshold
        task_threshold = self._task_thresholds.get(
            task_type, self._threshold_low
        )

        should_escalate = mean_lp < task_threshold

        logger.debug(
            "[check] medium zone (mean=%.3f) task=%s threshold=%.3f "
            "-> %s",
            mean_lp,
            task_type or "(none)",
            task_threshold,
            "escalate" if should_escalate else "keep local",
        )

        return should_escalate

    def analyze(self, logprobs: Sequence[Any]) -> Dict[str, Any]:
        """Return a detailed confidence analysis of the logprob sequence.

        Args:
            logprobs: Per-token log probabilities from the local model.

        Returns:
            A dict containing:
                - ``mean_logprob`` (float)
                - ``min_logprob`` (float)
                - ``max_logprob`` (float)
                - ``std_logprob`` (float)
                - ``num_tokens`` (int) -- number of valid tokens analysed
                - ``num_low_confidence_tokens`` (int) -- tokens below -3.0
                - ``confidence_level`` (str) -- ``"high"`` / ``"medium"``
                    / ``"low"``
                - ``should_escalate`` (bool) -- using default (no task type)
        """
        values = _filter_logprobs(logprobs)

        if not values:
            return {
                "mean_logprob": 0.0,
                "min_logprob": 0.0,
                "max_logprob": 0.0,
                "std_logprob": 0.0,
                "num_tokens": 0,
                "num_low_confidence_tokens": 0,
                "confidence_level": "high",
                "should_escalate": False,
            }

        n = len(values)
        mean_lp = sum(values) / n
        min_lp = min(values)
        max_lp = max(values)

        # Standard deviation (population)
        variance = sum((v - mean_lp) ** 2 for v in values) / n
        std_lp = math.sqrt(variance)

        num_low = sum(1 for v in values if v < _LOW_CONFIDENCE_TOKEN_THRESHOLD)

        # Determine confidence level
        if mean_lp > self._threshold_high:
            confidence_level = "high"
        elif mean_lp < self._threshold_low:
            confidence_level = "low"
        else:
            confidence_level = "medium"

        should_escalate = self.check(logprobs)

        return {
            "mean_logprob": round(mean_lp, 4),
            "min_logprob": round(min_lp, 4),
            "max_logprob": round(max_lp, 4),
            "std_logprob": round(std_lp, 4),
            "num_tokens": n,
            "num_low_confidence_tokens": num_low,
            "confidence_level": confidence_level,
            "should_escalate": should_escalate,
        }
