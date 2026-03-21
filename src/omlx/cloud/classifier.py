"""Task Classifier for OMLX cloud routing."""

from enum import Enum
from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger("omlx.cloud.classifier")


class CodingSubtask(str, Enum):
    """Fine-grained coding subtask categories."""

    COMPLETION = "completion"
    SIMPLE_FIX = "simple_fix"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    REFACTOR = "refactor"
    EXPLANATION = "explanation"


class TaskClassifier:
    """Task classifier - analyzes task type, complexity, and priority."""

    def __init__(self):
        # Task type keywords
        self.task_keywords = {
            "reasoning": ["分析", "推理", "证明", "解释", "为什么", "analyze", "reason", "explain", "why"],
            "coding": ["代码", "实现", "写", "修复", "bug", "code", "implement", "write", "fix"],
            "translation": ["翻译", "translate", "中文", "英文", "chinese", "english"],
            "creative": ["创意", "设计", "故事", "creative", "design", "story"],
            "qa": ["问答", "什么", "如何", "what", "how", "question"],
        }

        # Complexity indicators
        self.complexity_indicators = {
            "high": ["复杂", "深入", "详细", "系统", "架构", "complex", "detailed", "architecture"],
            "low": ["简单", "快速", "概述", "simple", "quick", "brief"],
        }

        # Coding subtask keywords (Chinese + English)
        self.coding_subtask_keywords = {
            CodingSubtask.COMPLETION: [
                "写", "实现", "创建", "添加", "新增",
                "write", "implement", "create", "add", "new function", "new class",
            ],
            CodingSubtask.SIMPLE_FIX: [
                "修复", "修改", "改一下", "小问题",
                "fix", "patch", "typo", "bug fix",
            ],
            CodingSubtask.DOCUMENTATION: [
                "文档", "注释", "readme", "docstring",
                "document", "comment", "jsdoc",
            ],
            CodingSubtask.DEBUGGING: [
                "调试", "排查", "为什么报错", "内存泄漏",
                "debug", "trace", "why error", "stack trace", "memory leak",
            ],
            CodingSubtask.ARCHITECTURE: [
                "架构", "设计", "系统设计", "分布式", "微服务",
                "architecture", "system design", "distributed", "microservice", "scalab",
            ],
            CodingSubtask.SECURITY: [
                "安全", "漏洞", "注入", "xss", "csrf", "渗透",
                "security", "vulnerability", "injection", "pentest",
            ],
            CodingSubtask.REFACTOR: [
                "重构", "优化", "清理",
                "refactor", "restructure", "cleanup", "optimize code", "dead code",
            ],
            CodingSubtask.EXPLANATION: [
                "解释", "什么意思", "怎么理解",
                "explain", "what does", "how does", "walkthrough",
            ],
        }

        # Sensitivity keywords (Chinese + English)
        self.sensitivity_keywords = {
            "nsfw": [
                "色情", "性爱", "做爱", "裸体", "裸照", "成人", "情色", "调情",
                "约炮", "一夜情", "性感", "胸部", "屁股", "下体", "阴茎", "阴道",
                "自慰", "口交", "肛交", "高潮", "勃起", "射精", "潮吹",
                "porn", "sex", "nude", "naked", "erotic", "nsfw", "xxx",
                "orgasm", "masturbat", "fetish", "hentai", "boobs", "dick",
                "pussy", "cock", "blowjob", "anal", "cum", "horny",
                "slutty", "kinky", "foreplay", "threesome",
            ],
            "violence": [
                "杀人", "自杀", "暴力", "虐待", "谋杀", "血腥", "残忍",
                "砍头", "分尸", "酷刑", "屠杀", "强奸", "强暴",
                "kill", "murder", "suicide", "torture", "gore", "violent",
                "rape", "assault", "massacre", "slaughter", "dismember",
            ],
            "politics": [
                "习近平", "共产党", "六四", "天安门", "台独", "藏独", "疆独",
                "法轮功", "民主运动", "政治犯", "维权", "翻墙",
                "tiananmen", "falun gong", "uyghur", "tibet independence",
                "xinjiang", "ccp",
            ],
            "drugs": [
                "毒品", "大麻", "可卡因", "海洛因", "冰毒", "摇头丸", "迷幻",
                "嗑药", "吸毒", "贩毒",
                "cocaine", "heroin", "meth", "mdma", "lsd", "weed",
                "marijuana", "drug use", "overdose",
            ],
        }

    def classify(self, messages: List[Dict]) -> Dict:
        """
        Classify a task based on conversation messages.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            {
                "task_type": "reasoning/coding/translation/creative/qa",
                "complexity": "high/medium/low",
                "priority": 0/1/2,
                "sensitivity": {...},
                "features": {...},
                "force_route": str (optional, only if [[tag]] detected)
            }
        """
        # Extract last user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return self._default_classification()

        last_message = user_messages[-1].get("content", "")

        # 0. Detect forced routing tag [[model/backend]]
        force_route = self._detect_force_route_tag(last_message)
        if force_route:
            last_message = force_route["clean_message"]
            # Update message content (strip the tag)
            user_messages[-1]["content"] = last_message

        # 1. Task type
        task_type = self._detect_task_type(last_message)

        # 2. Complexity
        complexity = self._detect_complexity(last_message, messages)

        # 3. Priority (default, can be overridden by the request)
        priority = 1

        # 4. Sensitivity detection
        sensitivity = self._detect_sensitivity(last_message)

        # 5. Features
        features = {
            "has_code": bool(re.search(r"```", last_message)),
            "message_length": len(last_message),
            "conversation_length": len(messages),
            "requires_context": len(messages) > 2,
        }

        result = {
            "task_type": task_type,
            "complexity": complexity,
            "priority": priority,
            "sensitivity": sensitivity,
            "features": features,
        }

        # Include coding subtask classification when task is coding
        if task_type == "coding":
            result["coding_subtask"] = self.classify_coding_subtask(messages)

        # Include forced route if a tag was detected
        if force_route:
            result["force_route"] = force_route["target"]
            logger.info(f"[classify] forced route tag: [[{force_route['target']}]]")

        return result

    def classify_coding_subtask(self, messages: List[Dict]) -> Dict:
        """Classify coding subtask with confidence.

        Analyzes conversation messages to determine a fine-grained coding subtask
        category. Uses keyword matching, message length, code block presence, and
        conversation length as signals.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            {
                "subtask": CodingSubtask value,
                "confidence": float (0-1),
                "signals": list of matched signals
            }
        """
        # Gather all user message text
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return {
                "subtask": CodingSubtask.COMPLETION.value,
                "confidence": 0.1,
                "signals": [],
            }

        last_message = user_messages[-1].get("content", "")
        all_user_text = " ".join(msg.get("content", "") for msg in user_messages)
        text_lower = last_message.lower()
        all_text_lower = all_user_text.lower()

        # 1. Score each subtask by keyword matches
        scores: Dict[CodingSubtask, int] = {}
        matched_signals: Dict[CodingSubtask, List[str]] = {}

        for subtask, keywords in self.coding_subtask_keywords.items():
            matched = [kw for kw in keywords if kw in text_lower]
            # Also check full conversation for weaker signals
            context_matched = [kw for kw in keywords if kw in all_text_lower and kw not in matched]
            total = len(matched) + len(context_matched) * 0.5
            scores[subtask] = total
            signals = []
            if matched:
                signals.extend([f"keyword:{kw}" for kw in matched])
            if context_matched:
                signals.extend([f"context:{kw}" for kw in context_matched])
            matched_signals[subtask] = signals

        # 2. Contextual boosting
        has_code = bool(re.search(r"```", last_message))
        message_length = len(last_message)
        conversation_length = len(messages)

        # Long messages lean toward debugging / architecture
        if message_length > 2000:
            scores[CodingSubtask.DEBUGGING] = scores.get(CodingSubtask.DEBUGGING, 0) + 1
            scores[CodingSubtask.ARCHITECTURE] = scores.get(CodingSubtask.ARCHITECTURE, 0) + 1
            matched_signals.setdefault(CodingSubtask.DEBUGGING, []).append("long_message")
            matched_signals.setdefault(CodingSubtask.ARCHITECTURE, []).append("long_message")

        # Code blocks lean toward debugging / fix / completion
        if has_code:
            scores[CodingSubtask.DEBUGGING] = scores.get(CodingSubtask.DEBUGGING, 0) + 1
            scores[CodingSubtask.SIMPLE_FIX] = scores.get(CodingSubtask.SIMPLE_FIX, 0) + 0.5
            scores[CodingSubtask.COMPLETION] = scores.get(CodingSubtask.COMPLETION, 0) + 0.5
            matched_signals.setdefault(CodingSubtask.DEBUGGING, []).append("has_code_block")
            matched_signals.setdefault(CodingSubtask.SIMPLE_FIX, []).append("has_code_block")
            matched_signals.setdefault(CodingSubtask.COMPLETION, []).append("has_code_block")

        # Long conversations lean toward debugging / architecture
        if conversation_length > 5:
            scores[CodingSubtask.DEBUGGING] = scores.get(CodingSubtask.DEBUGGING, 0) + 1
            scores[CodingSubtask.ARCHITECTURE] = scores.get(CodingSubtask.ARCHITECTURE, 0) + 0.5
            matched_signals.setdefault(CodingSubtask.DEBUGGING, []).append("long_conversation")
            matched_signals.setdefault(CodingSubtask.ARCHITECTURE, []).append("long_conversation")

        # 3. Pick the winning subtask
        max_score = max(scores.values()) if scores else 0

        if max_score == 0:
            # No signals at all: default to COMPLETION with low confidence
            logger.info("[coding_subtask] no signals matched, default=completion")
            return {
                "subtask": CodingSubtask.COMPLETION.value,
                "confidence": 0.2,
                "signals": [],
            }

        winning_subtask = max(scores, key=scores.get)
        winning_signals = matched_signals.get(winning_subtask, [])

        # 4. Compute confidence based on score magnitude and signal diversity
        # Base confidence from keyword match count (capped at 1.0)
        keyword_signals = [s for s in winning_signals if s.startswith("keyword:")]
        context_signals = [s for s in winning_signals if s.startswith("context:")]
        boost_signals = [s for s in winning_signals if not s.startswith("keyword:") and not s.startswith("context:")]

        confidence = min(1.0, 0.3 + len(keyword_signals) * 0.2 + len(context_signals) * 0.1 + len(boost_signals) * 0.1)

        # Penalize if the runner-up is close
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            gap_ratio = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            if gap_ratio < 0.3:
                confidence *= 0.8  # Reduce confidence when scores are close

        confidence = round(confidence, 2)

        logger.info(
            f"[coding_subtask] subtask={winning_subtask.value} | confidence={confidence} | "
            f"scores={{{', '.join(f'{k.value}:{v}' for k, v in scores.items() if v > 0)}}} | "
            f"signals={winning_signals}"
        )

        return {
            "subtask": winning_subtask.value,
            "confidence": confidence,
            "signals": winning_signals,
        }

    def _detect_task_type(self, text: str) -> str:
        """Detect task type from text content."""
        text_lower = text.lower()

        # Score each task type by keyword matches
        scores = {}
        for task_type, keywords in self.task_keywords.items():
            matched = [kw for kw in keywords if kw in text_lower]
            scores[task_type] = len(matched)
            if matched:
                logger.debug(f"[classify] {task_type}: matched keywords {matched} (score={len(matched)})")

        # Return the highest scoring type
        if max(scores.values()) > 0:
            result = max(scores, key=scores.get)
            logger.info(f"[classify] task_type={result} | score distribution: {scores}")
            return result
        else:
            logger.info(f"[classify] no keyword match, default=qa | input first 50 chars: {text_lower[:50]}...")
            return "qa"  # Default to Q&A

    def _detect_complexity(self, text: str, messages: List[Dict]) -> str:
        """Detect task complexity."""
        text_lower = text.lower()

        # 1. Keyword detection
        high_matched = [kw for kw in self.complexity_indicators["high"] if kw in text_lower]
        low_matched = [kw for kw in self.complexity_indicators["low"] if kw in text_lower]
        high_score = len(high_matched)
        low_score = len(low_matched)

        # 2. Length detection
        if len(text) > 500:
            high_score += 1
            logger.debug(f"[complexity] message_length={len(text)} > 500, high_score+1")

        # 3. Context length
        if len(messages) > 5:
            high_score += 1
            logger.debug(f"[complexity] conversation_turns={len(messages)} > 5, high_score+1")

        # Determine result
        if high_score > low_score:
            result = "high"
        elif low_score > high_score:
            result = "low"
        else:
            result = "medium"

        logger.info(
            f"[complexity] result={result} | high={high_score}(words:{high_matched}) "
            f"low={low_score}(words:{low_matched}) | msg_len={len(text)}, turns={len(messages)}"
        )
        return result

    def _detect_force_route_tag(self, text: str) -> Optional[Dict]:
        """Detect forced routing tag [[target]].

        Supported formats:
            [[gemini]] hello  -> force route to gemini
            [[deepseek]] analyze this -> force route to deepseek
            [[gpt]] / [[glm]] / [[local]] etc.

        Returns:
            None or {"target": "gemini", "clean_message": "hello"}
        """
        match = re.match(r"^\s*\[\[(\w[\w\-\.]*)\]\]\s*(.*)", text, re.DOTALL)
        if not match:
            return None

        target = match.group(1).lower().strip()
        clean_message = match.group(2).strip()

        logger.info(f"[tag_route] detected [[{target}]] | original: {text[:60]}...")
        return {"target": target, "clean_message": clean_message or text}

    def _detect_sensitivity(self, text: str) -> Dict:
        """Detect content sensitivity level.

        Returns:
            {
                "level": "none" / "low" / "high",
                "categories": ["nsfw", "violence", ...],
                "details": {"nsfw": ["keyword1", ...], ...}
            }
        """
        text_lower = text.lower()
        detected = {}

        for category, keywords in self.sensitivity_keywords.items():
            matched = [kw for kw in keywords if kw in text_lower]
            if matched:
                detected[category] = matched

        if not detected:
            logger.debug("[sensitivity] no sensitive content detected")
            return {"level": "none", "categories": [], "details": {}}

        # Determine sensitivity level
        total_hits = sum(len(v) for v in detected.values())
        categories = list(detected.keys())

        # high: many keyword hits OR multiple categories OR nsfw/violence
        if total_hits >= 3 or len(categories) >= 2 or "nsfw" in categories or "violence" in categories:
            level = "high"
        else:
            level = "low"

        logger.warning(
            f"[sensitivity] level={level} | categories={categories} | "
            f"hit_count={total_hits} | details={detected}"
        )
        return {"level": level, "categories": categories, "details": detected}

    def _default_classification(self) -> Dict:
        """Return default classification when no user messages found."""
        return {
            "task_type": "qa",
            "complexity": "medium",
            "priority": 1,
            "sensitivity": {"level": "none", "categories": [], "details": {}},
            "features": {},
        }
