"""Semantic Cache - Keyword-based Jaccard similarity caching for non-streaming requests

Design decisions:
- Uses keyword Jaccard similarity (no embedding dependency)
- Threshold 0.85 (conservative, avoid wrong answers)
- Only for non-streaming requests (streaming responses can't be cached)
- 4h TTL, max 500 entries, LRU eviction

Ported from ClawGate to ThunderOMLX with inline sqlite3 storage
(replaces SQLiteStore dependency).
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("omlx.cloud.semantic_cache")

# Common stop words (Chinese + English)
_STOP_WORDS = frozenset({
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "but", "or", "not", "no", "if", "then", "so", "too", "very",
    "this", "that", "it", "i", "me", "my", "you", "your", "he", "she",
    "we", "they", "what", "how", "when", "where", "which", "who", "why",
    # Chinese
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们",
    "这", "那", "和", "与", "而", "但", "或", "也", "都", "就",
    "还", "又", "不", "没", "有", "要", "会", "能", "可以",
    "一个", "一些", "什么", "怎么", "如何", "为什么", "哪个",
    "请", "帮", "用", "把", "给", "让", "吗", "呢", "吧",
})

# Pattern to split text into tokens
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class SemanticCacheDB:
    """Lightweight sqlite3 storage backend for SemanticCache.

    Replaces the shared SQLiteStore with direct sqlite3 calls,
    scoped specifically to semantic_cache table operations.
    """

    def __init__(self, db_path: str = "~/.omlx/data/semantic_cache.db"):
        expanded = os.path.expanduser(db_path)
        self.db_path = Path(expanded)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                keywords TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_cache_hash "
            "ON semantic_cache(query_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_cache_model "
            "ON semantic_cache(model)"
        )
        conn.commit()
        conn.close()

    def get_all(self, model: Optional[str] = None) -> List[Dict]:
        """Get all non-expired entries for similarity search"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if model:
            cursor.execute(
                "SELECT * FROM semantic_cache WHERE model = ? AND expires_at > datetime('now') "
                "ORDER BY hit_count DESC",
                (model,),
            )
        else:
            cursor.execute(
                "SELECT * FROM semantic_cache WHERE expires_at > datetime('now') "
                "ORDER BY hit_count DESC"
            )

        rows = cursor.fetchall()
        conn.close()

        result = []
        for row in rows:
            entry = dict(row)
            entry["keywords"] = json.loads(entry["keywords"])
            entry["response"] = json.loads(entry["response"])
            result.append(entry)
        return result

    def store(
        self,
        query_hash: str,
        query_text: str,
        keywords: List[str],
        response: Dict,
        model: str,
        ttl_hours: int = 4,
    ):
        """Store a semantic cache entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        expires_at = (
            datetime.utcnow() + timedelta(hours=ttl_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT OR REPLACE INTO semantic_cache (
                query_hash, query_text, keywords, response, model, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                query_hash,
                query_text,
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
                model,
                expires_at,
            ),
        )
        conn.commit()
        conn.close()

    def bump_hit(self, query_hash: str):
        """Increment hit count for a cache entry"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
            (query_hash,),
        )
        conn.commit()
        conn.close()

    def cleanup(self, max_size: int = 500) -> int:
        """Remove expired + LRU eviction beyond max_size"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Remove expired
        cursor.execute("DELETE FROM semantic_cache WHERE expires_at <= datetime('now')")
        expired_count = cursor.rowcount

        # LRU eviction: keep top max_size by (hit_count DESC, created_at DESC)
        cursor.execute(
            """
            DELETE FROM semantic_cache WHERE id NOT IN (
                SELECT id FROM semantic_cache
                ORDER BY hit_count DESC, created_at DESC
                LIMIT ?
            )
            """,
            (max_size,),
        )
        evicted_count = cursor.rowcount

        conn.commit()
        conn.close()
        return expired_count + evicted_count


class SemanticCache:
    """Keyword-based semantic cache with Jaccard similarity matching"""

    def __init__(
        self,
        db: SemanticCacheDB,
        threshold: float = 0.85,
        max_size: int = 500,
        ttl_hours: int = 4,
    ):
        self.db = db
        self.threshold = threshold
        self.max_size = max_size
        self.ttl_hours = ttl_hours

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text (tokenize + remove stop words)"""
        tokens = _TOKEN_PATTERN.findall(text.lower())
        # Filter stop words and very short tokens
        keywords = {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}
        return keywords

    @staticmethod
    def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
        """Compute Jaccard similarity between two keyword sets"""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def lookup(self, query: str, model: str) -> Optional[Dict]:
        """Look up a query in the semantic cache

        Returns cached response if a similar query (Jaccard >= threshold) exists.
        """
        query_keywords = self.extract_keywords(query)
        if not query_keywords:
            return None

        # Check exact hash first (fast path)
        query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
        entries = self.db.get_all(model=model)

        best_match = None
        best_similarity = 0.0

        for entry in entries:
            # Exact hash match
            if entry["query_hash"] == query_hash:
                self.db.bump_hit(entry["query_hash"])
                logger.info(
                    f"[SemanticCache] Exact HIT: hash={query_hash[:8]}..."
                )
                return entry

            # Jaccard similarity check
            cached_keywords = set(entry["keywords"])
            similarity = self.jaccard_similarity(query_keywords, cached_keywords)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match and best_similarity >= self.threshold:
            self.db.bump_hit(best_match["query_hash"])
            logger.info(
                f"[SemanticCache] Similarity HIT: "
                f"score={best_similarity:.3f} >= {self.threshold}"
            )
            return best_match

        return None

    def store(self, query: str, model: str, response: Dict):
        """Store a query-response pair in the semantic cache"""
        keywords = self.extract_keywords(query)
        if not keywords:
            return

        query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

        self.db.store(
            query_hash=query_hash,
            query_text=query[:500],  # truncate for storage
            keywords=sorted(keywords),
            response=response,
            model=model,
            ttl_hours=self.ttl_hours,
        )

        # Periodic cleanup
        self.db.cleanup(max_size=self.max_size)

        logger.debug(
            f"[SemanticCache] Stored: hash={query_hash[:8]}... "
            f"keywords={len(keywords)} model={model}"
        )
