"""Intelligent Router - 3-Tier cascade rule router for ThunderOMLX.

Routes inference requests through a tiered decision cascade:
  Tier 0: Force rules  - [[tag]], sensitivity, budget  (<1ms)
  Tier 1: Task routing  - coding subtask / task-type rules  (<1ms)
  Tier 2: Load-aware    - local queue overflow detection      (<1ms)

Plus session-pinning to avoid mid-conversation model switches.

Usage::

    router = IntelligentRouter(classifier, selector, settings)
    decision = await router.route(request, conversation_id="conv-123")
    # decision.target  -> RouteTarget.LOCAL / CLOUD_ECONOMY / CLOUD_PREMIUM
    # decision.model   -> "qwen3.5-35b-mlx" / "gemini-2.5-flash" / ...
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from omlx.cloud.confidence_checker import ConfidenceChecker
from omlx.cloud.mf_router import MFRouter
from omlx.cloud.ml_classifier import MLClassifier
from omlx.cloud.routing_store import RoutingStore

logger = logging.getLogger("omlx.cloud.intelligent_router")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RouteTarget(str, Enum):
    """Possible routing targets for an inference request."""

    LOCAL = "local"
    CLOUD_ECONOMY = "cloud_economy"
    CLOUD_PREMIUM = "cloud_premium"


@dataclass
class RoutingDecision:
    """Immutable record of a single routing decision."""

    target: RouteTarget
    model: str                      # Concrete model name
    reason: str                     # Human-readable explanation
    confidence: float = 1.0         # Decision confidence 0-1
    session_pinned: bool = False    # Was this pinned by session history
    tier: int = 1                   # Which tier made the decision (0/1/2)
    task_type: str = ""
    coding_subtask: str = ""
    complexity: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Coding subtasks that are lightweight enough for local inference
# ---------------------------------------------------------------------------

_LOCAL_CODING_SUBTASKS = frozenset({
    "completion",
    "simple_fix",
    "documentation",
    "explanation",
})

_PREMIUM_CODING_SUBTASKS = frozenset({
    "architecture",
    "security",
})

_COMPLEXITY_SCALING_SUBTASKS = frozenset({
    "debugging",
    "refactor",
})


# ---------------------------------------------------------------------------
# Tag -> (RouteTarget, model) mapping for force-route resolution
# ---------------------------------------------------------------------------

_FORCE_ROUTE_ALIASES: Dict[str, Tuple[RouteTarget, str]] = {
    # Gemini family
    "gemini":           (RouteTarget.CLOUD_ECONOMY, "gemini-2.5-flash"),
    "gemini-2.5-flash": (RouteTarget.CLOUD_ECONOMY, "gemini-2.5-flash"),
    "gemini-2.5-pro":   (RouteTarget.CLOUD_PREMIUM, "gemini-2.5-pro"),
    # DeepSeek family
    "deepseek":         (RouteTarget.CLOUD_PREMIUM, "deepseek-r1"),
    "deepseek-r1":      (RouteTarget.CLOUD_PREMIUM, "deepseek-r1"),
    "deepseek-v3":      (RouteTarget.CLOUD_ECONOMY, "deepseek-v3"),
    # GLM family
    "glm":              (RouteTarget.CLOUD_ECONOMY, "glm-5"),
    "glm-5":            (RouteTarget.CLOUD_ECONOMY, "glm-5"),
    "glm-4-flash":      (RouteTarget.CLOUD_ECONOMY, "glm-4-flash"),
    # OpenAI family
    "gpt":              (RouteTarget.CLOUD_PREMIUM, "gpt-4o"),
    "gpt-4o":           (RouteTarget.CLOUD_PREMIUM, "gpt-4o"),
}


# ---------------------------------------------------------------------------
# IntelligentRouter
# ---------------------------------------------------------------------------

class IntelligentRouter:
    """3-Tier cascade rule router with session pinning.

    Tier 0 - Force Rules (<1ms):
        ``[[model]]`` tag, high-sensitivity content, budget exceeded.

    Tier 1 - Task-Based Routing (<1ms):
        Maps (task_type, coding_subtask, complexity) to a route target.

    Tier 2 - Load-Aware (<1ms):
        Overflows local requests to cloud economy when queue is deep.

    Session Pin:
        After *pin_threshold* consecutive turns with the same model in a
        conversation, that model is pinned for the rest of the conversation.
    """

    def __init__(
        self,
        classifier: Any,
        selector: Any,
        settings: Any,
        local_models: Optional[List[str]] = None,
        budget_checker: Optional[Any] = None,
        routing_store: Optional[RoutingStore] = None,
        ml_classifier: Optional[MLClassifier] = None,
        mf_router: Optional[MFRouter] = None,
    ) -> None:
        """Initialize the IntelligentRouter.

        Args:
            classifier: TaskClassifier instance with ``classify(messages)``
            selector:   ModelSelector instance with ``select(...)``
            settings:   CloudSettingsV2 instance
            local_models: Available local model IDs
            budget_checker: Optional BudgetChecker with ``check()``
            routing_store: Optional RoutingStore for SQLite persistence
            ml_classifier: Optional 0.8B MLClassifier for hybrid mode
            mf_router: Optional MFRouter for preference-based scoring
        """
        self._classifier = classifier
        self._selector = selector
        self._settings = settings
        self._routing_store: Optional[RoutingStore] = routing_store
        self._ml_classifier: Optional[MLClassifier] = ml_classifier
        self._mf_router: Optional[MFRouter] = mf_router
        self._local_models = local_models or ["qwen3.5-35b-mlx"]
        self._primary_local = (
            self._local_models[0] if self._local_models else "qwen3.5-35b-mlx"
        )
        self._budget_checker = budget_checker

        # Session pin tracking: conv_id -> pinned model
        self._session_pins: Dict[str, str] = {}
        # conv_id -> consecutive-same-model count
        self._session_counts: Dict[str, int] = {}
        # conv_id -> last model used
        self._session_last_model: Dict[str, str] = {}

        # Configuration with safe defaults
        self._shadow_mode: bool = getattr(
            settings, "intelligent_routing_shadow", True
        )
        self._overflow_threshold: int = getattr(
            settings, "local_overflow_threshold", 4
        )
        self._pin_threshold: int = getattr(
            settings, "session_pin_threshold", 3
        )

        # Bounded routing log for statistics
        self._routing_log: List[RoutingDecision] = []
        self._max_log_size: int = 1000

        # Confidence checker for cascade escalation (Phase 2)
        self._confidence_checker = ConfidenceChecker()

        # Cloud model tiers
        self._economy_models: List[str] = [
            "gemini-2.5-flash",
            "glm-4-flash",
            "deepseek-v3",
        ]
        self._premium_models: List[str] = [
            "deepseek-r1",
            "gemini-2.5-pro",
            "gpt-4o",
        ]

        logger.info(
            "[__init__] IntelligentRouter ready | local=%s | "
            "overflow_threshold=%d | pin_threshold=%d | shadow=%s",
            self._primary_local,
            self._overflow_threshold,
            self._pin_threshold,
            self._shadow_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        request: Any,
        conversation_id: Optional[str] = None,
        local_queue_depth: int = 0,
    ) -> RoutingDecision:
        """Route a request through the 3-tier cascade.

        Args:
            request: Incoming request with ``.messages`` (Pydantic models).
            conversation_id: Optional conversation identifier for session pin.
            local_queue_depth: Current local engine queue depth for Tier 2.

        Returns:
            A RoutingDecision describing where to send the request.
        """
        # 1. Extract messages (Pydantic → plain dicts)
        messages = self._extract_messages(request)

        # 2. Classify (rule-based)
        classification = self._classifier.classify(messages)

        # 2.5. Hybrid classification (0.8B supplement when confidence is low)
        classification = await self._hybrid_classify(classification, messages)

        # 3. Tier 0 - Force Rules
        decision = self._tier0_force_rules(classification)

        if decision is None:
            # 4. Tier 1 - Task-Based Routing
            decision = self._tier1_task_routing(classification)

            # 4.5. MF Router scoring (override Tier 1 for ambiguous cases)
            decision = await self._mf_router_refine(decision, messages)

            # 5. Tier 2 - Load-Aware
            decision = self._tier2_load_aware(decision, local_queue_depth)

        # 6. Session Pin
        decision = self._check_session_pin(decision, conversation_id)

        # 7. Record and log
        self._record_decision(decision)

        logger.info(
            "[route] tier=%d target=%s model=%s reason='%s' "
            "task=%s subtask=%s complexity=%s pinned=%s conv=%s",
            decision.tier,
            decision.target.value,
            decision.model,
            decision.reason,
            decision.task_type,
            decision.coding_subtask,
            decision.complexity,
            decision.session_pinned,
            conversation_id or "none",
        )

        return decision

    def get_routing_stats(self) -> Dict[str, Any]:
        """Return aggregate routing statistics.

        Returns:
            Dict with total_decisions, per-target/task/tier counts, and
            active session pin count.
        """
        total = len(self._routing_log)
        target_counts: Dict[str, int] = {}
        task_counts: Dict[str, int] = {}
        tier_counts: Dict[int, int] = {}

        for d in self._routing_log:
            target_counts[d.target.value] = (
                target_counts.get(d.target.value, 0) + 1
            )
            task_counts[d.task_type] = task_counts.get(d.task_type, 0) + 1
            tier_counts[d.tier] = tier_counts.get(d.tier, 0) + 1

        result = {
            "total_decisions": total,
            "per_target": target_counts,
            "per_task_type": task_counts,
            "per_tier": tier_counts,
            "session_pins": len(self._session_pins),
        }
        if self._mf_router is not None:
            result["mf_router"] = self._mf_router.get_stats()
        return result

    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent routing decisions as serialisable dicts.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            List of decision dicts, newest first.
        """
        recent = self._routing_log[-limit:]
        return [asdict(d) for d in reversed(recent)]

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    def set_shadow_mode(self, enabled: bool) -> None:
        """Toggle shadow mode on or off.

        In shadow mode the router logs decisions but cascade escalation
        is suppressed (the local result is always returned even when
        confidence is low).
        """
        self._shadow_mode = enabled
        logger.info("[set_shadow_mode] shadow_mode=%s", enabled)

    async def route_with_cascade(
        self,
        request: Any,
        conversation_id: Optional[str] = None,
        local_queue_depth: int = 0,
    ) -> Tuple[RoutingDecision, bool]:
        """Route with optional cascade escalation for non-streaming requests.

        For tasks in the "complexity scaling" group (debugging, refactor),
        this method first makes a standard routing decision.  When the
        decision targets the local model **and** shadow mode is off, the
        caller can later check confidence and call :meth:`escalate` to
        upgrade to a cloud model.

        Args:
            request: The inference request (Pydantic model or dict).
            conversation_id: Optional conversation identifier.
            local_queue_depth: Current local engine queue size.

        Returns:
            A (decision, cascade_eligible) tuple.  *cascade_eligible* is
            ``True`` when the caller should check logprobs after local
            generation and potentially call :meth:`escalate`.
        """
        decision = await self.route(request, conversation_id, local_queue_depth)

        # Cascade only applies to local, non-shadow, complexity-scaling tasks
        cascade_eligible = (
            decision.target == RouteTarget.LOCAL
            and not self._shadow_mode
            and decision.coding_subtask in _COMPLEXITY_SCALING_SUBTASKS
        )

        if cascade_eligible:
            logger.info(
                "[route_with_cascade] cascade eligible: model=%s subtask=%s",
                decision.model, decision.coding_subtask,
            )

        return decision, cascade_eligible

    def check_confidence(
        self, logprobs: list, task_type: str = ""
    ) -> bool:
        """Check whether local generation logprobs indicate low confidence.

        Args:
            logprobs: Per-token log probabilities from the local model.
            task_type: The task type for threshold selection.

        Returns:
            ``True`` if confidence is low and escalation is recommended.
        """
        return self._confidence_checker.check(logprobs, task_type)

    def escalate(self, original: RoutingDecision) -> RoutingDecision:
        """Create an escalated routing decision targeting a cloud model.

        Replaces the local target with a premium cloud model while
        preserving the original task metadata.

        Args:
            original: The original local routing decision.

        Returns:
            A new RoutingDecision targeting a premium cloud model.
        """
        escalated = RoutingDecision(
            target=RouteTarget.CLOUD_PREMIUM,
            model=self._pick_premium_model(),
            reason=f"cascade escalation from {original.model} (low confidence)",
            confidence=original.confidence,
            session_pinned=False,
            tier=original.tier,
            task_type=original.task_type,
            coding_subtask=original.coding_subtask,
            complexity="high",
        )
        self._record_decision(escalated)
        logger.info(
            "[escalate] %s -> %s (reason: %s)",
            original.model, escalated.model, escalated.reason,
        )
        return escalated

    def analyze_confidence(self, logprobs: list) -> dict:
        """Return detailed confidence analysis for debugging/monitoring.

        Args:
            logprobs: Per-token log probabilities.

        Returns:
            Dict with mean/min/max/std logprob, confidence level, etc.
        """
        return self._confidence_checker.analyze(logprobs)

    # ------------------------------------------------------------------
    # Hybrid Classification (rule + 0.8B ML)
    # ------------------------------------------------------------------

    _HYBRID_CONFIDENCE_THRESHOLD = 0.9

    async def _hybrid_classify(
        self,
        classification: Dict[str, Any],
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Optionally refine classification using the 0.8B ML classifier.

        Strategy:
          1. If rule classifier confidence >= 0.9, trust it directly.
          2. If confidence < 0.9 and MLClassifier is available, call 0.8B.
          3. If both agree, adopt the result.
          4. If they disagree, prefer the more conservative option
             (the one recommending cloud over local).

        Args:
            classification: Output from the rule-based TaskClassifier.
            messages: Plain-dict messages for extracting the last user text.

        Returns:
            Potentially refined classification dict (mutated in-place).
        """
        if self._ml_classifier is None:
            return classification

        # Only relevant for coding tasks with subtask info
        subtask_info = classification.get("coding_subtask")
        if not subtask_info or not isinstance(subtask_info, dict):
            return classification

        rule_confidence = subtask_info.get("confidence", 1.0)
        if rule_confidence >= self._HYBRID_CONFIDENCE_THRESHOLD:
            return classification

        # Extract last user message for ML classification
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        if not last_user:
            return classification

        ml_result = await self._ml_classifier.classify(last_user)
        if ml_result is None:
            logger.debug("[_hybrid_classify] ML classifier returned None, keeping rule result")
            return classification

        rule_subtask = subtask_info.get("subtask", "")
        ml_subtask = ml_result.get("task_type", "")

        if rule_subtask == ml_subtask:
            # Agreement — boost confidence
            subtask_info["confidence"] = min(rule_confidence + 0.2, 1.0)
            logger.info(
                "[_hybrid_classify] agreement: %s (confidence %.2f -> %.2f)",
                rule_subtask, rule_confidence, subtask_info["confidence"],
            )
        else:
            # Disagreement — prefer the more conservative (cloud-leaning) option
            ml_needs_cloud = ml_result.get("needs_cloud", False)
            rule_is_local = rule_subtask in _LOCAL_CODING_SUBTASKS

            if ml_needs_cloud and rule_is_local:
                # ML says cloud, rule says local — adopt ML's classification
                subtask_info["subtask"] = ml_subtask
                subtask_info["confidence"] = 0.7
                classification["coding_subtask"] = subtask_info
                if ml_result.get("complexity"):
                    classification["complexity"] = ml_result["complexity"]
                logger.info(
                    "[_hybrid_classify] override: rule=%s -> ml=%s (ML recommends cloud)",
                    rule_subtask, ml_subtask,
                )
            else:
                # Otherwise keep rule result but note the disagreement
                logger.info(
                    "[_hybrid_classify] keep rule=%s (ml=%s, no cloud upgrade needed)",
                    rule_subtask, ml_subtask,
                )

        return classification

    # ------------------------------------------------------------------
    # MF Router refinement (RouteLLM preference-based scoring)
    # ------------------------------------------------------------------

    async def _mf_router_refine(
        self,
        decision: RoutingDecision,
        messages: List[Dict[str, str]],
    ) -> RoutingDecision:
        """Refine a Tier 1 decision using MF Router preference scoring.

        The MF Router predicts win_rate(strong, weak) based on Chatbot Arena
        preference data.  It can override the rule-based decision in two cases:

        1. **Upgrade**: Rule says LOCAL but MF says cloud (high win_rate).
           Only applied when rule confidence is not very high and the task
           is in the "complexity scaling" group (debugging, refactor).
        2. **Downgrade**: Rule says CLOUD_PREMIUM but MF says local (low
           win_rate).  Applied for economy — if the MF model thinks the
           weak model wins, we save cloud cost.

        Args:
            decision: Current Tier 1 routing decision.
            messages: Plain-dict messages for extracting last user text.

        Returns:
            Potentially modified RoutingDecision.
        """
        if self._mf_router is None or not self._mf_router.available:
            return decision

        # Extract last user message
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        if not last_user:
            return decision

        try:
            should_cloud, win_rate = await self._mf_router.should_route_to_cloud(
                last_user
            )
        except Exception as exc:
            logger.warning("[_mf_router_refine] MF Router error: %s", exc)
            return decision

        logger.info(
            "[_mf_router_refine] win_rate=%.4f threshold=%.2f "
            "mf_says=%s | current=%s/%s",
            win_rate,
            self._mf_router.threshold,
            "CLOUD" if should_cloud else "LOCAL",
            decision.target.value,
            decision.model,
        )

        # Case 1: MF says CLOUD but rule says LOCAL
        # Only upgrade for ambiguous cases (complexity-scaling subtasks)
        if (
            should_cloud
            and decision.target == RouteTarget.LOCAL
            and decision.coding_subtask in _COMPLEXITY_SCALING_SUBTASKS
        ):
            upgraded = RoutingDecision(
                target=RouteTarget.CLOUD_PREMIUM,
                model=self._pick_premium_model(),
                reason=(
                    f"MF Router upgrade: win_rate={win_rate:.3f} "
                    f"(>{self._mf_router.threshold:.2f}) for "
                    f"{decision.coding_subtask}"
                ),
                confidence=win_rate,
                tier=1,
                task_type=decision.task_type,
                coding_subtask=decision.coding_subtask,
                complexity="high",
            )
            logger.info(
                "[_mf_router_refine] UPGRADE: %s -> %s (win_rate=%.3f)",
                decision.model, upgraded.model, win_rate,
            )
            return upgraded

        # Case 2: MF says LOCAL but rule says CLOUD_PREMIUM
        # Downgrade to local to save cost.  Use a symmetric threshold:
        # if win_rate < (threshold - margin), the weak model suffices.
        # This adapts to different embedding backends (OpenAI range ~0.2-0.8,
        # Gemini range ~0.72-0.78).
        downgrade_threshold = self._mf_router.threshold - 0.03
        if (
            not should_cloud
            and decision.target == RouteTarget.CLOUD_PREMIUM
            and win_rate < downgrade_threshold
        ):
            downgraded = RoutingDecision(
                target=RouteTarget.LOCAL,
                model=self._primary_local,
                reason=(
                    f"MF Router downgrade: win_rate={win_rate:.3f} "
                    f"(<{downgrade_threshold:.2f}) — local model sufficient"
                ),
                confidence=1.0 - win_rate,
                tier=1,
                task_type=decision.task_type,
                coding_subtask=decision.coding_subtask,
                complexity=decision.complexity,
            )
            logger.info(
                "[_mf_router_refine] DOWNGRADE: %s -> %s (win_rate=%.3f)",
                decision.model, downgraded.model, win_rate,
            )
            return downgraded

        # No change — MF agrees with rule or not confident enough to override
        return decision

    # ------------------------------------------------------------------
    # Tier 0 - Force Rules
    # ------------------------------------------------------------------

    def _tier0_force_rules(
        self, classification: Dict[str, Any]
    ) -> Optional[RoutingDecision]:
        """Apply Tier 0 force rules.

        Checks (in order):
          1. ``[[tag]]`` force-route directive
          2. High-sensitivity content
          3. Budget exceeded

        Returns:
            RoutingDecision if a force rule fires, else None.
        """
        task_type = classification.get("task_type", "")
        coding_subtask = self._normalize_coding_subtask(
            classification.get("coding_subtask", "")
        )
        complexity = classification.get("complexity", "")

        # --- 1. Force route tag ---
        force_tag = classification.get("force_route")
        if force_tag:
            target, model = self._resolve_force_route(force_tag)
            logger.debug(
                "[_tier0_force_rules] force tag '%s' -> %s / %s",
                force_tag, target.value, model,
            )
            return RoutingDecision(
                target=target,
                model=model,
                reason=f"Force route tag [[{force_tag}]]",
                tier=0,
                task_type=task_type,
                coding_subtask=coding_subtask,
                complexity=complexity,
            )

        # --- 2. High sensitivity ---
        sensitivity = classification.get("sensitivity", {})
        if sensitivity.get("level") == "high":
            all_cloud = self._economy_models + self._premium_models
            selected = self._selector.select(
                task_info=classification,
                agent_type=None,
                available_models=all_cloud,
                optimize_for="quality",
                load_info=None,
            )
            target = self._classify_model_tier(selected)
            categories = sensitivity.get("categories", [])
            logger.debug(
                "[_tier0_force_rules] high sensitivity %s -> %s / %s",
                categories, target.value, selected,
            )
            return RoutingDecision(
                target=target,
                model=selected,
                reason=f"High sensitivity content: {categories}",
                tier=0,
                task_type=task_type,
                coding_subtask=coding_subtask,
                complexity=complexity,
            )

        # --- 3. Budget exceeded ---
        if self._budget_checker is not None:
            budget_status = self._budget_checker.check()
            if not budget_status.get("allowed", True):
                budget_reason = budget_status.get("reason", "Budget exceeded")
                logger.debug(
                    "[_tier0_force_rules] budget exceeded -> LOCAL | %s",
                    budget_reason,
                )
                return RoutingDecision(
                    target=RouteTarget.LOCAL,
                    model=self._primary_local,
                    reason=f"Budget exceeded: {budget_reason}",
                    tier=0,
                    task_type=task_type,
                    coding_subtask=coding_subtask,
                    complexity=complexity,
                )

        return None

    # ------------------------------------------------------------------
    # Tier 1 - Task-Based Routing
    # ------------------------------------------------------------------

    def _tier1_task_routing(
        self, classification: Dict[str, Any]
    ) -> RoutingDecision:
        """Apply Tier 1 task-based routing rules.

        Determines the routing target based on task_type, coding_subtask,
        and complexity.

        Returns:
            A RoutingDecision (always non-None).
        """
        task_type = classification.get("task_type", "qa")
        coding_subtask = self._normalize_coding_subtask(
            classification.get("coding_subtask", "")
        )
        complexity = classification.get("complexity", "medium")

        if task_type == "coding" and coding_subtask:
            target, model, reason = self._route_coding_task(
                coding_subtask, complexity
            )
        else:
            target, model, reason = self._route_general_task(
                task_type, complexity
            )

        return RoutingDecision(
            target=target,
            model=model,
            reason=reason,
            tier=1,
            task_type=task_type,
            coding_subtask=coding_subtask,
            complexity=complexity,
        )

    def _route_coding_task(
        self, subtask: str, complexity: str
    ) -> Tuple[RouteTarget, str, str]:
        """Route a coding task based on subtask and complexity.

        Returns:
            (target, model, reason)
        """
        if subtask in _LOCAL_CODING_SUBTASKS:
            return (
                RouteTarget.LOCAL,
                self._primary_local,
                f"Coding/{subtask} -> LOCAL",
            )

        if subtask in _PREMIUM_CODING_SUBTASKS:
            return (
                RouteTarget.CLOUD_PREMIUM,
                self._pick_premium_model(),
                f"Coding/{subtask} -> CLOUD_PREMIUM",
            )

        if subtask in _COMPLEXITY_SCALING_SUBTASKS:
            if complexity == "high":
                return (
                    RouteTarget.CLOUD_PREMIUM,
                    self._pick_premium_model(),
                    f"Coding/{subtask} (high complexity) -> CLOUD_PREMIUM",
                )
            # low or medium -> local
            return (
                RouteTarget.LOCAL,
                self._primary_local,
                f"Coding/{subtask} ({complexity} complexity) -> LOCAL",
            )

        # Unknown subtask fallback
        return (
            RouteTarget.LOCAL,
            self._primary_local,
            f"Coding/{subtask} (unknown subtask) -> LOCAL",
        )

    def _route_general_task(
        self, task_type: str, complexity: str
    ) -> Tuple[RouteTarget, str, str]:
        """Route a non-coding task based on type and complexity.

        Returns:
            (target, model, reason)
        """
        if task_type == "reasoning" and complexity == "high":
            return (
                RouteTarget.CLOUD_PREMIUM,
                self._pick_premium_model(),
                "Reasoning (high complexity) -> CLOUD_PREMIUM",
            )

        if task_type == "translation":
            return (
                RouteTarget.CLOUD_ECONOMY,
                self._pick_economy_model(),
                "Translation -> CLOUD_ECONOMY",
            )

        if task_type == "creative":
            return (
                RouteTarget.LOCAL,
                self._primary_local,
                "Creative -> LOCAL",
            )

        if task_type == "qa":
            if complexity == "high":
                return (
                    RouteTarget.CLOUD_ECONOMY,
                    self._pick_economy_model(),
                    "QA (high complexity) -> CLOUD_ECONOMY",
                )
            return (
                RouteTarget.LOCAL,
                self._primary_local,
                f"QA ({complexity} complexity) -> LOCAL",
            )

        # Default fallback for reasoning/medium, unknown types, etc.
        return (
            RouteTarget.LOCAL,
            self._primary_local,
            f"{task_type}/{complexity} -> LOCAL (default)",
        )

    # ------------------------------------------------------------------
    # Tier 2 - Load-Aware
    # ------------------------------------------------------------------

    def _tier2_load_aware(
        self, decision: RoutingDecision, local_queue_depth: int = 0
    ) -> RoutingDecision:
        """Apply Tier 2 load-aware overflow.

        If the decision targets LOCAL but the local queue is too deep,
        overflow to CLOUD_ECONOMY.

        Args:
            decision: Current routing decision.
            local_queue_depth: Number of pending requests in local queue.

        Returns:
            Potentially modified RoutingDecision.
        """
        if (
            decision.target == RouteTarget.LOCAL
            and local_queue_depth > self._overflow_threshold
        ):
            overflow_model = self._pick_economy_model()
            logger.info(
                "[_tier2_load_aware] local queue overflow (%d > %d) -> %s",
                local_queue_depth,
                self._overflow_threshold,
                overflow_model,
            )
            return RoutingDecision(
                target=RouteTarget.CLOUD_ECONOMY,
                model=overflow_model,
                reason=(
                    f"Local queue overflow ({local_queue_depth} > "
                    f"{self._overflow_threshold}) -> CLOUD_ECONOMY"
                ),
                confidence=decision.confidence,
                tier=2,
                task_type=decision.task_type,
                coding_subtask=decision.coding_subtask,
                complexity=decision.complexity,
            )

        return decision

    # ------------------------------------------------------------------
    # Session Pin
    # ------------------------------------------------------------------

    def _check_session_pin(
        self,
        decision: RoutingDecision,
        conversation_id: Optional[str],
    ) -> RoutingDecision:
        """Check and apply session pinning.

        If the conversation already has a pinned model, override the
        decision.  Otherwise, update tracking counters.

        Args:
            decision: Current routing decision.
            conversation_id: Conversation identifier (None = no pinning).

        Returns:
            Potentially overridden RoutingDecision.
        """
        if conversation_id is None:
            return decision

        pinned_model = self._session_pins.get(conversation_id)
        if pinned_model is not None:
            pinned_target = self._classify_model_tier(pinned_model)
            logger.debug(
                "[_check_session_pin] conv=%s pinned to %s",
                conversation_id, pinned_model,
            )
            return RoutingDecision(
                target=pinned_target,
                model=pinned_model,
                reason=f"Session pinned to {pinned_model}",
                confidence=decision.confidence,
                session_pinned=True,
                tier=decision.tier,
                task_type=decision.task_type,
                coding_subtask=decision.coding_subtask,
                complexity=decision.complexity,
            )

        self._update_session_tracking(conversation_id, decision.model)
        return decision

    def _update_session_tracking(
        self, conversation_id: str, model: str
    ) -> None:
        """Update per-conversation model tracking counters.

        When *pin_threshold* consecutive turns use the same model the
        conversation is pinned to that model.

        Args:
            conversation_id: Conversation identifier.
            model: Model used for the current turn.
        """
        last_model = self._session_last_model.get(conversation_id)

        if model == last_model:
            count = self._session_counts.get(conversation_id, 0) + 1
        else:
            count = 1

        self._session_counts[conversation_id] = count
        self._session_last_model[conversation_id] = model

        if count >= self._pin_threshold:
            self._session_pins[conversation_id] = model
            logger.info(
                "[_update_session_tracking] conv=%s pinned to %s "
                "after %d consecutive turns",
                conversation_id, model, count,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_coding_subtask(raw: Any) -> str:
        """Normalize a coding_subtask value to a plain string.

        The classifier may return the coding subtask as either a bare
        string (``"completion"``) or a dict
        (``{"subtask": "completion", "confidence": 0.5, ...}``).

        Args:
            raw: The raw value from ``classification["coding_subtask"]``.

        Returns:
            The subtask name as a string, or ``""`` if unavailable.
        """
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            return raw.get("subtask", "")
        return ""

    @staticmethod
    def _extract_messages(request: Any) -> List[Dict[str, str]]:
        """Convert request.messages (Pydantic models) to plain dicts.

        Handles the case where ``content`` may be a list (vision models)
        by joining into a single string.

        Args:
            request: Incoming request object with ``.messages``.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        messages: List[Dict[str, str]] = []
        raw_messages = getattr(request, "messages", None) or []

        for msg in raw_messages:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")

            if content is None:
                content = ""
            elif not isinstance(content, str):
                # Vision models may send a list of content parts
                content = str(content)

            messages.append({"role": role, "content": content})

        return messages

    def _pick_economy_model(self) -> str:
        """Return the preferred economy cloud model."""
        return self._economy_models[0] if self._economy_models else "gemini-2.5-flash"

    def _pick_premium_model(self) -> str:
        """Return the preferred premium cloud model."""
        return self._premium_models[0] if self._premium_models else "deepseek-r1"

    def _resolve_force_route(self, tag: str) -> Tuple[RouteTarget, str]:
        """Resolve a ``[[tag]]`` force-route directive to target + model.

        Args:
            tag: The tag string extracted by the classifier.

        Returns:
            (RouteTarget, concrete_model_name)
        """
        tag_lower = tag.lower().strip()

        # "local" always maps to the primary local model
        if tag_lower == "local":
            return RouteTarget.LOCAL, self._primary_local

        # Check local model names
        if tag_lower in (m.lower() for m in self._local_models):
            return RouteTarget.LOCAL, self._primary_local

        # Known aliases (static table)
        if tag_lower in _FORCE_ROUTE_ALIASES:
            return _FORCE_ROUTE_ALIASES[tag_lower]

        # Check dynamic economy / premium lists
        for model in self._economy_models:
            if tag_lower == model.lower():
                return RouteTarget.CLOUD_ECONOMY, model

        for model in self._premium_models:
            if tag_lower == model.lower():
                return RouteTarget.CLOUD_PREMIUM, model

        # Unknown tag: default to premium
        logger.warning(
            "[_resolve_force_route] unknown tag '%s', defaulting to "
            "CLOUD_PREMIUM / %s",
            tag, self._pick_premium_model(),
        )
        return RouteTarget.CLOUD_PREMIUM, self._pick_premium_model()

    def _classify_model_tier(self, model: str) -> RouteTarget:
        """Determine which RouteTarget a concrete model belongs to.

        Args:
            model: A concrete model name.

        Returns:
            The corresponding RouteTarget.
        """
        if model in self._economy_models:
            return RouteTarget.CLOUD_ECONOMY
        if model in self._premium_models:
            return RouteTarget.CLOUD_PREMIUM
        if model in self._local_models or model == self._primary_local:
            return RouteTarget.LOCAL
        # Fallback: treat unknown cloud models as premium
        return RouteTarget.CLOUD_PREMIUM

    def _record_decision(self, decision: RoutingDecision) -> None:
        """Append a decision to the bounded routing log.

        Trims the oldest entries when the log exceeds ``_max_log_size``.
        Also persists to SQLite via RoutingStore (fire-and-forget).

        Args:
            decision: The decision to record.
        """
        self._routing_log.append(decision)
        if len(self._routing_log) > self._max_log_size:
            self._routing_log = self._routing_log[-self._max_log_size:]

        # Persist to SQLite (fire-and-forget)
        if self._routing_store:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._routing_store.record_decision(asdict(decision))
                )
            except RuntimeError:
                pass  # No running event loop
