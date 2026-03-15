"""
ContextPilot Adapter for ThunderOMLX.

Provides context indexing and request optimization for improved cache hit rates
in multi-user scenarios with shared system prompts.

Reference: ThunderOMLX prefix_cache.py style
"""

from dataclasses import dataclass, field
import hashlib
import logging
from typing import Dict, List, Optional, Any, TypedDict

logger = logging.getLogger(__name__)


class OptimizedRequest(TypedDict):
    """Result of request optimization containing reordered messages and metadata."""
    messages: List[Dict[str, str]]
    message_boundaries: List[int]
    context_refs: Dict[str, "ContextBlock"]
    system_prompt_hash: Optional[str]  # sha256[:16] of system prompt for cluster identification


@dataclass
class ContextBlock:
    """
    Semantic-level context block with content hash for deduplication.

    Attributes:
        content: The text content of this block
        role: The role (system/user/assistant) of this message
        hash: SHA256 hash prefix (16 chars) of content
        ref_id: Unique reference ID in format ctx_{hash}
    """
    content: str
    role: str
    hash: str
    ref_id: str

    @classmethod
    def from_message(cls, content: str, role: str = "user") -> "ContextBlock":
        """
        Create a ContextBlock from message content and role.

        Args:
            content: The message content string
            role: The role of the message (default: "user")

        Returns:
            A new ContextBlock instance with computed hash
        """
        hash_val = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return cls(
            content=content,
            role=role,
            hash=hash_val,
            ref_id=f"ctx_{hash_val}"
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "role": self.role,
            "hash": self.hash,
            "ref_id": self.ref_id
        }


class ContextIndex:
    """
    Global deduplication index for context blocks.

    Maintains a mapping of hash -> ContextBlock to enable content deduplication
    across multiple requests.

    Attributes:
        blocks: Dictionary mapping hash to ContextBlock
    """

    def __init__(self) -> None:
        """Initialize an empty context index."""
        self._blocks: Dict[str, ContextBlock] = {}
        logger.debug("ContextIndex initialized")

    @property
    def blocks(self) -> Dict[str, ContextBlock]:
        """Get the blocks dictionary (read-only view)."""
        return self._blocks.copy()

    def add_or_get(self, content: str, role: str) -> ContextBlock:
        """
        Add a new block or return existing one if content already indexed.

        Args:
            content: The message content to index
            role: The role of the message

        Returns:
            Either existing ContextBlock (if hash matches) or newly created one
        """
        block = ContextBlock.from_message(content, role)

        if block.hash in self._blocks:
            existing = self._blocks[block.hash]
            logger.debug(f"Cache hit for hash={block.hash}, ref_id={block.ref_id}")
            return existing

        self._blocks[block.hash] = block
        logger.debug(f"Added new block: hash={block.hash}, ref_id={block.ref_id}")
        return block

    def get_by_hash(self, hash_val: str) -> Optional[ContextBlock]:
        """
        Retrieve a block by its hash.

        Args:
            hash_val: The 16-char hash to look up

        Returns:
            ContextBlock if found, None otherwise
        """
        return self._blocks.get(hash_val)

    def get_by_ref_id(self, ref_id: str) -> Optional[ContextBlock]:
        """
        Retrieve a block by its reference ID.

        Args:
            ref_id: The reference ID in format ctx_{hash}

        Returns:
            ContextBlock if found, None otherwise
        """
        if not ref_id.startswith("ctx_"):
            return None
        hash_val = ref_id[4:]  # Remove "ctx_" prefix
        return self.get_by_hash(hash_val)

    def clear(self) -> None:
        """Clear all indexed blocks."""
        self._blocks.clear()
        logger.debug("ContextIndex cleared")

    def __len__(self) -> int:
        """Return the number of indexed blocks."""
        return len(self._blocks)

    def __contains__(self, hash_val: str) -> bool:
        """Check if a hash exists in the index."""
        return hash_val in self._blocks


class ContextPilotAdapter:
    """
    Main adapter for context optimization in LLM requests.

    Implements request reordering to maximize prefix cache hits by placing
    common prefixes (typically system prompts) at the beginning of requests.

    Attributes:
        context_index: The global ContextIndex for deduplication
    """

    def __init__(
        self,
        context_index: Optional[ContextIndex] = None,
        tokenizer: Any = None,
    ) -> None:
        """
        Initialize the adapter with optional context index and tokenizer.

        Args:
            context_index: Optional ContextIndex instance (creates new one if None)
            tokenizer: Optional tokenizer for precise message boundary computation.
                       If None, message_boundaries will be empty.
        """
        self._context_index = context_index or ContextIndex()
        self._tokenizer = tokenizer
        logger.debug(
            f"ContextPilotAdapter initialized "
            f"(tokenizer={'yes' if tokenizer else 'no'})"
        )

    @property
    def context_index(self) -> ContextIndex:
        """Get the context index."""
        return self._context_index

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set tokenizer for precise message boundary computation."""
        self._tokenizer = tokenizer
        logger.debug("ContextPilotAdapter: tokenizer set")

    def optimize_request(
        self,
        messages: List[Dict[str, Any]],
        previous_requests: Optional[List[List[Dict[str, Any]]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> OptimizedRequest:
        """
        Optimize a request by reordering messages for better cache utilization.

        The optimization strategy:
        1. Parse messages into ContextBlocks (indexed for deduplication)
        2. Find longest common prefix with previous requests
        3. Reorder: common prefix first, new content last
        4. Compute precise message boundaries (if tokenizer available)
        5. Compute system prompt fingerprint

        Args:
            messages: List of message dicts with 'content' and 'role' keys
            previous_requests: List of previous message lists for comparison
            prompt_token_ids: Pre-computed token IDs for boundary validation

        Returns:
            OptimizedRequest with reordered messages and metadata

        Note:
            Fail-safe: Returns original messages if optimization fails
        """
        try:
            return self._do_optimize(messages, previous_requests or [], prompt_token_ids)
        except Exception as e:
            logger.warning(f"Optimization failed, returning original messages: {e}")
            return self._create_fallback_result(messages)

    def _do_optimize(
        self,
        messages: List[Dict[str, Any]],
        previous_requests: List[List[Dict[str, Any]]],
        prompt_token_ids: Optional[List[int]] = None,
    ) -> OptimizedRequest:
        """Internal optimization implementation."""

        # Step 1: Parse messages into ContextBlocks
        current_blocks = self._parse_messages(messages)

        if not current_blocks:
            return self._create_fallback_result(messages)

        # Step 2: Find maximum common prefix length
        max_prefix_len = 0
        for prev_msgs in previous_requests:
            prefix_len = self._compute_prefix_len(messages, prev_msgs)
            max_prefix_len = max(max_prefix_len, prefix_len)

        # Step 3: Reorder messages
        reordered_messages = self._reorder_messages(
            messages, current_blocks, max_prefix_len
        )

        # Step 4: Build context_refs
        context_refs = {block.ref_id: block for block in current_blocks}

        # Step 5: Compute system prompt fingerprint for cluster identification
        system_prompt_hash = self._compute_system_prompt_hash(messages)

        # Step 6: Compute precise message boundaries using tokenizer
        message_boundaries = self._extract_message_boundaries(
            reordered_messages, prompt_token_ids
        )

        logger.debug(
            f"Optimized request: {len(messages)} messages, "
            f"prefix_len={max_prefix_len}, refs={len(context_refs)}, "
            f"system_hash={system_prompt_hash or 'none'}, "
            f"boundaries={message_boundaries[:3]}{'...' if len(message_boundaries) > 3 else ''}"
        )

        return OptimizedRequest(
            messages=reordered_messages,
            message_boundaries=message_boundaries,
            context_refs=context_refs,
            system_prompt_hash=system_prompt_hash,
        )

    def _parse_messages(self, messages: List[Dict[str, Any]]) -> List[ContextBlock]:
        """
        Parse messages into ContextBlocks and add to index.

        Args:
            messages: List of message dicts

        Returns:
            List of ContextBlock instances
        """
        blocks: List[ContextBlock] = []

        for msg in messages:
            content = self._extract_content(msg)
            role = self._extract_role(msg)

            if content is not None:
                block = self._context_index.add_or_get(content, role)
                blocks.append(block)
            else:
                logger.warning(f"Message missing content, skipping: {msg}")

        return blocks

    def _extract_content(self, msg: Dict[str, Any]) -> Optional[str]:
        """Extract content from a message dict."""
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle multi-modal messages (list of content parts)
            text_parts = [
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return " ".join(text_parts) if text_parts else None
        return None

    def _extract_role(self, msg: Dict[str, Any]) -> str:
        """Extract role from a message dict with default fallback."""
        role = msg.get("role", "user")
        return str(role) if role else "user"

    def _compute_prefix_len(
        self,
        msgs1: List[Dict[str, Any]],
        msgs2: List[Dict[str, Any]]
    ) -> int:
        """
        Compute the length of common prefix between two message lists.

        Two messages match if both content and role are identical.

        Args:
            msgs1: First message list
            msgs2: Second message list

        Returns:
            Length of common prefix (number of matching messages from start)
        """
        prefix_len = 0
        for m1, m2 in zip(msgs1, msgs2):
            if self._messages_equal(m1, m2):
                prefix_len += 1
            else:
                break
        return prefix_len

    def _messages_equal(self, m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
        """Check if two messages have same content and role."""
        content1 = self._extract_content(m1)
        content2 = self._extract_content(m2)
        role1 = self._extract_role(m1)
        role2 = self._extract_role(m2)

        return content1 == content2 and role1 == role2

    def _reorder_messages(
        self,
        messages: List[Dict[str, Any]],
        blocks: List[ContextBlock],
        common_prefix_len: int
    ) -> List[Dict[str, Any]]:
        """
        Reorder messages to place common prefix first.

        Strategy:
        - Messages [0:common_prefix_len] are already in optimal position
        - For Phase 1, we maintain order but could swap in future phases

        Args:
            messages: Original message list
            blocks: Parsed ContextBlocks
            common_prefix_len: Length of common prefix

        Returns:
            Reordered message list
        """
        # Phase 1: Simple pass-through with validation
        # Phase 2 will implement more sophisticated reordering

        if common_prefix_len == 0:
            # No common prefix, return as-is
            return list(messages)

        # For OpenClaw scenario (shared system prompts), prefix is already first
        # Just ensure messages are properly formatted
        return [self._normalize_message(msg) for msg in messages]

    def _normalize_message(self, msg: Dict[str, Any]) -> Dict[str, str]:
        """Normalize a message dict to standard format."""
        content = self._extract_content(msg) or ""
        role = self._extract_role(msg)
        return {"content": content, "role": role}

    def _extract_message_boundaries(
        self,
        messages: List[Dict[str, Any]],
        prompt_token_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Extract precise message boundaries using incremental chat template.

        Uses apply_chat_template() on progressively longer message prefixes
        to determine exact token ranges for each message, including all
        special tokens (<|im_start|>, role tags, etc.).

        Args:
            messages: List of message dicts with 'role' and 'content'
            prompt_token_ids: Pre-computed token IDs for boundary validation

        Returns:
            List of cumulative token counts marking each message boundary.
            boundaries[i] = end token position of message i (exclusive).
        """
        if not messages or self._tokenizer is None:
            return []

        boundaries: List[int] = []
        actual_token_count = len(prompt_token_ids) if prompt_token_ids else 0

        try:
            has_chat_template = hasattr(self._tokenizer, 'apply_chat_template')

            for i in range(len(messages)):
                msg_prefix = messages[:i + 1]

                if has_chat_template:
                    try:
                        template_text = self._tokenizer.apply_chat_template(
                            msg_prefix,
                            tokenize=False,
                            add_generation_prompt=(i == len(messages) - 1),
                        )
                        prefix_tokens = self._tokenizer.encode(template_text)
                        current_token_count = len(prefix_tokens)
                    except (TypeError, Exception):
                        # Fallback if template doesn't support partial call
                        break
                else:
                    text = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in msg_prefix
                    ) + "\nassistant:"
                    prefix_tokens = self._tokenizer.encode(text)
                    current_token_count = len(prefix_tokens)

                boundaries.append(current_token_count)

            # Validate monotonicity
            if boundaries:
                for j in range(1, len(boundaries)):
                    if boundaries[j] <= boundaries[j - 1]:
                        logger.warning(
                            f"Non-monotonic boundary at msg {j}: "
                            f"{boundaries[j]} <= {boundaries[j-1]}, falling back"
                        )
                        return []

            # Align last boundary to actual token count
            if boundaries and actual_token_count > 0:
                if abs(boundaries[-1] - actual_token_count) > 5:
                    logger.warning(
                        f"Boundary mismatch: boundaries[-1]={boundaries[-1]} "
                        f"vs actual={actual_token_count}, adjusting"
                    )
                boundaries[-1] = actual_token_count

            return boundaries
        except Exception as e:
            logger.warning(f"Failed to extract message boundaries: {e}")
            return []

    def _compute_system_prompt_hash(
        self,
        messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Compute SHA256 fingerprint of the system prompt for cluster identification.

        Returns:
            First 16 hex chars of SHA256 hash of system prompt content,
            or None if no system prompt found.
        """
        for msg in messages:
            if self._extract_role(msg) == "system":
                content = self._extract_content(msg)
                if content:
                    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return None

    def _create_fallback_result(
        self,
        messages: List[Dict[str, Any]]
    ) -> OptimizedRequest:
        """Create a fail-safe result with original messages."""
        return OptimizedRequest(
            messages=list(messages),
            message_boundaries=[],
            context_refs={},
            system_prompt_hash=self._compute_system_prompt_hash(messages),
        )


# ============================================================================
# Unit Tests (run with: python -m pytest adapter.py -v)
# ============================================================================

if __name__ == "__main__":
    """Quick self-test when run directly."""

    print("=" * 60)
    print("ContextPilot Adapter Self-Test")
    print("=" * 60)

    # Test 1: ContextBlock creation
    print("\n[Test 1] ContextBlock creation...")
    block = ContextBlock.from_message("Hello, world!", "user")
    assert block.hash == hashlib.sha256("Hello, world!".encode()).hexdigest()[:16]
    assert block.ref_id == f"ctx_{block.hash}"
    print(f"  ✓ block.hash = {block.hash}")
    print(f"  ✓ block.ref_id = {block.ref_id}")

    # Test 2: ContextIndex deduplication
    print("\n[Test 2] ContextIndex deduplication...")
    index = ContextIndex()
    block1 = index.add_or_get("Shared content", "system")
    block2 = index.add_or_get("Shared content", "system")
    assert block1 is block2, "Should return same block for identical content"
    assert len(index) == 1, "Index should have only 1 entry"
    print(f"  ✓ Deduplication works: block1 is block2 = {block1 is block2}")

    # Test 3: ContextPilotAdapter basic optimization
    print("\n[Test 3] ContextPilotAdapter optimization...")
    adapter = ContextPilotAdapter(index)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    result = adapter.optimize_request(messages, previous_requests=[])
    assert len(result["messages"]) == 2
    assert len(result["context_refs"]) == 2
    print(f"  ✓ Optimized {len(messages)} messages")
    print(f"  ✓ Created {len(result['context_refs'])} context refs")

    # Test 4: Prefix detection
    print("\n[Test 4] Common prefix detection...")
    prev_request = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Previous question"},
    ]

    new_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "New question"},
    ]

    result = adapter.optimize_request(new_messages, previous_requests=[prev_request])
    print(f"  ✓ Processed request with previous context")

    # Test 5: Fail-safe behavior
    print("\n[Test 5] Fail-safe behavior...")
    bad_messages = [{"role": "user"}]  # Missing content
    result = adapter.optimize_request(bad_messages)
    assert result["messages"] == bad_messages
    print("  ✓ Returns original on parse issues")

    # Test 6: Edge case - empty messages
    print("\n[Test 6] Edge case - empty messages...")
    result = adapter.optimize_request([])
    assert result["messages"] == []
    print("  ✓ Handles empty message list")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
