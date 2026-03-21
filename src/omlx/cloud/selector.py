"""Model Selector for OMLX cloud routing."""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger("omlx.cloud.selector")


class ModelSelector:
    """Model selector - Quality-Cost trade-off optimization."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Optional configuration dict. Expected structure:
                {
                    "cloud_models": [
                        {"name": "deepseek-r1", "cost_per_1k": 0.0014, ...},
                        ...
                    ],
                    "agent_profiles": {
                        "judge": {"preferred_models": ["deepseek-r1", ...]},
                        ...
                    }
                }
        """
        self.config = config or {}

        # Model configuration
        self.cloud_models = {
            model["name"]: model for model in self.config.get("cloud_models", [])
        }

        # Agent profiles
        self.agent_profiles = self.config.get("agent_profiles", {})

        # Default model ranking (quality-first)
        self.quality_ranking = [
            "deepseek-r1",
            "gpt-4o",
            "deepseek-v3",
            "glm-5",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "glm-4-flash",
        ]

        # Cost ranking (cost-first)
        self.cost_ranking = [
            "glm-4-flash",
            "gemini-2.5-flash",
            "deepseek-v3",
            "glm-5",
            "deepseek-r1",
            "gemini-2.5-pro",
            "gpt-4o",
        ]

        # Model content tolerance
        # lenient = permissive, moderate = middle, strict = restrictive
        # Sensitive content should route to lenient models to avoid refusals
        self.model_tolerance = {
            # Local models: no censorship, most lenient
            "qwen-1.7b": "lenient",
            # Gemini: relatively lenient
            "gemini-2.5-pro": "lenient",
            "gemini-2.5-flash": "lenient",
            "gemini-2-flash": "lenient",
            "gemini-2-pro": "lenient",
            # GPT: moderate
            "gpt-4o": "moderate",
            "gpt-5.2": "moderate",
            "gpt-5.1": "moderate",
            # DeepSeek: strict (Chinese model, NSFW/politics sensitive)
            "deepseek-r1": "strict",
            "deepseek-v3": "strict",
            # GLM: very strict (Chinese model, strictest censorship)
            "glm-5": "strict",
            "glm-4-plus": "strict",
            "glm-4-flash": "strict",
        }

        # Sensitivity categories each tolerance level can handle
        self.tolerance_capabilities = {
            "lenient": ["nsfw", "violence", "politics", "drugs"],  # Can handle almost everything
            "moderate": ["drugs", "violence"],  # Partial
            "strict": [],  # Rejects most sensitive content
        }

    def select(
        self,
        task_info: Dict,
        agent_type: Optional[str] = None,
        available_models: Optional[List[str]] = None,
        optimize_for: str = "quality",
        load_info: Optional[Dict[str, dict]] = None,
    ) -> str:
        """
        Select the optimal model.

        Args:
            task_info: Task information (from TaskClassifier).
            agent_type: Agent type (judge/builder/flash).
            available_models: List of available model names.
            optimize_for: Optimization target ("quality" / "cost" / "balanced").
            load_info: Model load information (from QueueManager.get_all_loads()).

        Returns:
            Selected model name.
        """
        logger.info(
            f"[select] starting model selection | task={task_info} | agent={agent_type} "
            f"| optimize={optimize_for} | available={available_models}"
        )

        # 0. Sensitive content check - filter unsuitable models
        sensitivity = task_info.get("sensitivity", {})
        sensitivity_level = sensitivity.get("level", "none")
        sensitivity_categories = sensitivity.get("categories", [])

        if sensitivity_level != "none" and available_models:
            filtered = self._filter_by_tolerance(available_models, sensitivity_categories)
            if filtered:
                logger.warning(
                    f"[select] sensitive content detected! level={sensitivity_level} "
                    f"categories={sensitivity_categories} | "
                    f"before={available_models} -> after={filtered}"
                )
                available_models = filtered
            else:
                logger.warning(
                    f"[select] sensitive content detected but no tolerant models available! "
                    f"keeping original list={available_models}"
                )

        # 1. Load-aware reranking (skip circuit-broken models, deprioritize high-load)
        if load_info and available_models:
            available_models = self._rerank_by_load(available_models, load_info)
            logger.debug(f"[select] after load reranking: {available_models}")

        # 2. If agent type specified, use agent profile preferences
        if agent_type and agent_type in self.agent_profiles:
            preferred_models = self.agent_profiles[agent_type].get("preferred_models", [])
            logger.debug(f"[select] agent={agent_type} preferred models: {preferred_models}")
            # Pick the first available preferred model
            for model in preferred_models:
                if not available_models or model in available_models:
                    logger.info(f"[select] agent mode hit: {model} (agent={agent_type})")
                    return model
            logger.warning(f"[select] agent={agent_type} preferred models unavailable, falling back to task routing")

        # 3. Task-based selection
        task_type = task_info.get("task_type", "qa")
        complexity = task_info.get("complexity", "medium")

        # 4. Sensitive content priority routing (bypass normal strategy for high level)
        if sensitivity_level == "high" and available_models:
            selected = self._select_for_sensitive(available_models, sensitivity_categories)
            if selected:
                logger.info(
                    f"[select] sensitive route: {selected} | "
                    f"tolerance={self.model_tolerance.get(selected, 'unknown')} | "
                    f"categories={sensitivity_categories}"
                )
                return selected

        # 5. Select by optimization target
        if optimize_for == "quality":
            selected = self._select_by_quality(task_type, complexity, available_models)
        elif optimize_for == "cost":
            selected = self._select_by_cost(task_type, complexity, available_models)
        else:  # balanced
            selected = self._select_balanced(task_type, complexity, available_models)

        logger.info(
            f"[select] final choice: {selected} | strategy={optimize_for} "
            f"| task={task_type} complexity={complexity}"
        )
        return selected

    def _rerank_by_load(self, models: List[str], load_info: Dict[str, dict]) -> List[str]:
        """Rerank by load: skip circuit-broken models, deprioritize high-load ones."""
        scored = []
        for model in models:
            info = load_info.get(model, {})
            # Skip circuit-broken models
            if info.get("circuit_state") == "open":
                logger.debug(f"[select] skipping circuit-broken model: {model}")
                continue
            # Load penalty: queue_depth * 2 + in_flight (lower is better)
            penalty = info.get("queue_depth", 0) * 2 + info.get("in_flight", 0)
            scored.append((model, penalty))

        if not scored:
            return models  # If all circuit-broken, keep original list

        scored.sort(key=lambda x: x[1])
        return [m for m, _ in scored]

    def _filter_by_tolerance(self, models: List[str], categories: List[str]) -> List[str]:
        """Filter out models that cannot handle the detected sensitive content."""
        result = []
        for model in models:
            tolerance = self.model_tolerance.get(model, "moderate")
            capabilities = self.tolerance_capabilities.get(tolerance, [])
            # Model can handle all detected sensitivity categories
            if all(cat in capabilities for cat in categories):
                result.append(model)
        return result

    def _select_for_sensitive(self, models: List[str], categories: List[str]) -> Optional[str]:
        """Select the best model for sensitive content (prefer lenient models)."""
        # Sort by tolerance: lenient > moderate > strict
        tolerance_order = {"lenient": 0, "moderate": 1, "strict": 2}

        scored = []
        for model in models:
            tolerance = self.model_tolerance.get(model, "moderate")
            capabilities = self.tolerance_capabilities.get(tolerance, [])
            can_handle = all(cat in capabilities for cat in categories)
            score = tolerance_order.get(tolerance, 1)
            scored.append((model, score, can_handle))

        # First pick models that can handle the content, then sort by tolerance
        capable = [(m, s) for m, s, c in scored if c]
        if capable:
            capable.sort(key=lambda x: x[1])
            return capable[0][0]

        # None can handle it - return the most lenient
        scored.sort(key=lambda x: x[1])
        return scored[0][0] if scored else None

    def _select_by_quality(
        self, task_type: str, complexity: str, available_models: Optional[List[str]]
    ) -> str:
        """Quality-first model selection."""

        # High complexity -> high quality models
        if complexity == "high":
            candidates = ["deepseek-r1", "gpt-4o", "gemini-2.5-pro"]
        elif complexity == "medium":
            candidates = ["deepseek-v3", "glm-5", "gemini-2.5-pro"]
        else:
            candidates = ["gemini-2.5-flash", "glm-4-flash"]

        # Pick the first available candidate
        for model in candidates:
            if not available_models or model in available_models:
                return model

        # Fallback: pick the first available model
        if available_models:
            return available_models[0]

        # Final fallback
        return "gemini-2.5-flash"

    def _select_by_cost(
        self, task_type: str, complexity: str, available_models: Optional[List[str]]
    ) -> str:
        """Cost-first model selection."""

        # Low complexity -> low cost models
        if complexity == "low":
            candidates = ["glm-4-flash", "gemini-2.5-flash"]
        elif complexity == "medium":
            candidates = ["gemini-2.5-flash", "glm-5", "deepseek-v3"]
        else:
            candidates = ["glm-5", "deepseek-v3", "deepseek-r1"]

        # Pick the first available candidate
        for model in candidates:
            if not available_models or model in available_models:
                return model

        # Fallback: pick the first available model
        if available_models:
            return available_models[0]

        return "glm-4-flash"

    def _select_balanced(
        self, task_type: str, complexity: str, available_models: Optional[List[str]]
    ) -> str:
        """Balanced model selection."""

        # Reasoning tasks -> quality first
        if task_type == "reasoning":
            return self._select_by_quality(task_type, complexity, available_models)
        # Simple tasks -> cost first
        elif complexity == "low":
            return self._select_by_cost(task_type, complexity, available_models)
        # Everything else -> mid-tier models
        else:
            candidates = ["gemini-2.5-flash", "glm-5", "deepseek-v3"]
            for model in candidates:
                if not available_models or model in available_models:
                    return model

            # Fallback: pick the first available model
            if available_models:
                return available_models[0]

            return "gemini-2.5-flash"

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a given model and token counts.

        Args:
            model: Model name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        if model not in self.cloud_models:
            return 0.0

        model_config = self.cloud_models[model]
        cost_per_1k = model_config.get("cost_per_1k", 0.001)

        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * cost_per_1k
