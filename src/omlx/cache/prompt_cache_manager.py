# SPDX-License-Identifier: Apache-2.0
"""Named prompt cache management with KVTC compression.

Provides persistent, named KV cache storage for frequently reused prompts
(system prompts, RAG document context, coding assistant instructions).

Storage layout:
    ~/.omlx/cache/prompt_caches/
        {name}.safetensors   - KV cache data (KVTC-compressed or raw)
        {name}.json          - Metadata (model, tokens, compression info)

Compression: KVTC achieves 4-8x over raw safetensors, vs ~2-3x for lz4.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .kvtc_calibration_store import KVTCCalibrationStore
from .kvtc_codec import KVTCCodecConfig, KVTCSharedCalibration

logger = logging.getLogger(__name__)


class PromptCacheManager:
    """Named prompt cache management with optional KVTC compression.

    Each named cache stores per-layer KV pairs as safetensors, with optional
    KVTC transform-coding compression for 4-8x disk savings.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        calibration_store: Optional[KVTCCalibrationStore] = None,
    ):
        if cache_dir is None:
            cache_dir = Path.home() / ".omlx" / "cache" / "prompt_caches"
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._calibration_store = calibration_store or KVTCCalibrationStore()

    def _data_path(self, name: str) -> Path:
        return self._dir / f"{name}.safetensors"

    def _meta_path(self, name: str) -> Path:
        return self._dir / f"{name}.json"

    async def save(
        self,
        name: str,
        cache_data: List,
        model_name: str,
        token_count: int,
        prompt_text: str = "",
        compress: bool = True,
        kvtc_config: Optional[KVTCCodecConfig] = None,
    ) -> Dict[str, Any]:
        """Save a named prompt cache with optional KVTC compression.

        Args:
            name: Unique cache name (e.g., 'system-coding-v1').
            cache_data: Per-layer KV pairs: List[(keys, values)] where each
                is mx.array or np.ndarray of shape [heads, tokens, dim] or
                [heads*tokens, dim].
            model_name: Model identifier for calibration lookup.
            token_count: Number of prompt tokens cached.
            prompt_text: Original prompt text (stored in metadata only).
            compress: Whether to apply KVTC compression.
            kvtc_config: Custom KVTC config (uses defaults if None).

        Returns:
            Dict with name, file_size, compression_ratio, token_count, etc.
        """
        t0 = time.time()
        data_path = self._data_path(name)
        meta_path = self._meta_path(name)

        # Normalize cache_data to consistent 2D [rows, head_dim] per layer.
        # SSD blocks may store mixed formats:
        #   4D [1, heads, tokens, dim]  → squeeze → [heads, tokens, dim]
        #   3D [heads, tokens, dim]     → reshape → [heads*tokens, dim]
        #   2D [tokens, heads*dim]      → reshape → [heads*tokens, dim]
        # We detect head_dim from 3D tensors and reshape 2D ones to match.
        cache_data = _normalize_cache_shapes(cache_data)

        # Calculate uncompressed size
        raw_size = 0
        for layer_data in cache_data:
            if isinstance(layer_data, (tuple, list)) and len(layer_data) == 2:
                k, v = layer_data
                k_arr = np.array(k) if not isinstance(k, np.ndarray) else k
                v_arr = np.array(v) if not isinstance(v, np.ndarray) else v
                raw_size += k_arr.nbytes + v_arr.nbytes

        arrays: Dict[str, mx.array] = {}
        compression_ratio = 1.0
        calibration_fingerprint = None

        if compress:
            # KVTC shared calibration requires uniform feature dims across layers.
            # Non-uniform models (e.g., distilled with mixed GQA) fall back to raw.
            key_dims = set()
            val_dims = set()
            for ld in cache_data:
                if isinstance(ld, (tuple, list)) and len(ld) == 2:
                    key_dims.add(ld[0].shape[-1])
                    val_dims.add(ld[1].shape[-1])
            if len(key_dims) > 1 or len(val_dims) > 1:
                logger.warning(
                    "Non-uniform feature dims (keys=%s, values=%s) — "
                    "falling back to uncompressed save for '%s'",
                    key_dims, val_dims, name,
                )
                compress = False

        if compress:
            calibration = self._calibration_store.get_or_fit(
                model_name, cache_data, kvtc_config,
            )
            calibration_fingerprint = calibration.fingerprint()

            for i, layer_data in enumerate(cache_data):
                if isinstance(layer_data, (tuple, list)) and len(layer_data) == 2:
                    k, v = layer_data
                    k_np = k.astype(np.float32) if isinstance(k, np.ndarray) else np.array(k).astype(np.float32)
                    v_np = v.astype(np.float32) if isinstance(v, np.ndarray) else np.array(v).astype(np.float32)

                    # Store original shapes for reconstruction
                    # (already 2D after normalization above)
                    k_orig_shape = k_np.shape
                    v_orig_shape = v_np.shape

                    enc_k, enc_v = calibration.encode(k_np, v_np)

                    # Save encoded components as separate arrays
                    _save_encoded_layer(arrays, f"layer_{i}_keys", enc_k, k_orig_shape)
                    _save_encoded_layer(arrays, f"layer_{i}_values", enc_v, v_orig_shape)
        else:
            # Raw save (no compression)
            for i, layer_data in enumerate(cache_data):
                if isinstance(layer_data, (tuple, list)) and len(layer_data) == 2:
                    k, v = layer_data
                    arrays[f"layer_{i}_keys"] = mx.array(np.array(k))
                    arrays[f"layer_{i}_values"] = mx.array(np.array(v))

        mx.save_safetensors(str(data_path), arrays)

        file_size = data_path.stat().st_size
        compression_ratio = raw_size / max(1, file_size)
        encode_time = (time.time() - t0) * 1000

        # Save metadata
        meta = {
            "name": name,
            "model_name": model_name,
            "token_count": token_count,
            "num_layers": len(cache_data),
            "compressed": compress,
            "compression_ratio": round(compression_ratio, 2),
            "file_size_bytes": file_size,
            "raw_size_bytes": raw_size,
            "encode_time_ms": round(encode_time, 1),
            "calibration_fingerprint": calibration_fingerprint,
            "prompt_text_preview": prompt_text[:200] if prompt_text else "",
            "created_at": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(
            "Prompt cache saved: name=%s model=%s tokens=%d ratio=%.1fx time=%.0fms",
            name, model_name, token_count, compression_ratio, encode_time,
        )

        return {
            "name": name,
            "model": model_name,
            "token_count": token_count,
            "file_size_bytes": file_size,
            "compression_ratio": compression_ratio,
            "encode_time_ms": encode_time,
        }

    async def load(
        self,
        name: str,
        decompress: bool = True,
    ) -> Tuple[Optional[List], Optional[Dict]]:
        """Load a named prompt cache, optionally decompressing with KVTC.

        Returns:
            (cache_data, metadata) or (None, None) if not found.
            cache_data: List[(mx.array, mx.array)] per layer.
        """
        data_path = self._data_path(name)
        meta_path = self._meta_path(name)

        if not data_path.exists() or not meta_path.exists():
            return None, None

        t0 = time.time()
        meta = json.loads(meta_path.read_text())
        arrays = mx.load(str(data_path))
        num_layers = meta["num_layers"]

        cache_data = []

        if meta.get("compressed", False) and decompress:
            calibration = self._calibration_store.load(meta["model_name"])
            if calibration is None:
                logger.error(
                    "Cannot decompress %s: calibration missing for %s",
                    name, meta["model_name"],
                )
                return None, meta

            for i in range(num_layers):
                enc_k = _load_encoded_layer(arrays, f"layer_{i}_keys")
                enc_v = _load_encoded_layer(arrays, f"layer_{i}_values")

                if enc_k is None or enc_v is None:
                    logger.warning("Layer %d missing from cache %s", i, name)
                    continue

                orig_shape_k = enc_k[-1]  # Last element is original shape
                orig_shape_v = enc_v[-1]
                enc_k_data = enc_k[:-1]
                enc_v_data = enc_v[:-1]

                dec_k = calibration.keys.decode(enc_k_data)
                dec_v = calibration.values.decode(enc_v_data)

                # Reshape back to original shape
                k_shape = tuple(int(v) for v in np.array(orig_shape_k).tolist())
                v_shape = tuple(int(v) for v in np.array(orig_shape_v).tolist())
                dec_k = dec_k.reshape(k_shape)
                dec_v = dec_v.reshape(v_shape)

                cache_data.append((mx.array(dec_k), mx.array(dec_v)))
        else:
            # Raw load
            for i in range(num_layers):
                k_key = f"layer_{i}_keys"
                v_key = f"layer_{i}_values"
                if k_key in arrays and v_key in arrays:
                    cache_data.append((arrays[k_key], arrays[v_key]))

        load_time = (time.time() - t0) * 1000
        meta["load_time_ms"] = round(load_time, 1)

        logger.info(
            "Prompt cache loaded: name=%s layers=%d time=%.0fms",
            name, len(cache_data), load_time,
        )

        return cache_data, meta

    def list_caches(self) -> List[Dict[str, Any]]:
        """List all saved prompt caches with metadata."""
        results = []
        for meta_path in sorted(self._dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text())
                results.append({
                    "name": meta.get("name", meta_path.stem),
                    "model": meta.get("model_name", "unknown"),
                    "token_count": meta.get("token_count", 0),
                    "file_size_bytes": meta.get("file_size_bytes", 0),
                    "compression_ratio": meta.get("compression_ratio", 1.0),
                    "compressed": meta.get("compressed", False),
                    "created_at": meta.get("created_at", 0),
                })
            except Exception as e:
                logger.warning("Skipping corrupt cache metadata %s: %s", meta_path, e)
        return results

    def delete(self, name: str) -> bool:
        """Delete a named prompt cache. Returns True if deleted."""
        deleted = False
        for path in (self._data_path(name), self._meta_path(name)):
            if path.exists():
                path.unlink()
                deleted = True
        if deleted:
            logger.info("Prompt cache deleted: name=%s", name)
        return deleted

    def stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for all prompt caches."""
        caches = self.list_caches()
        total_size = sum(c.get("file_size_bytes", 0) for c in caches)
        return {
            "total_count": len(caches),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self._dir),
        }

    def exists(self, name: str) -> bool:
        """Check if a named cache exists."""
        return self._data_path(name).exists() and self._meta_path(name).exists()


# ---------------------------------------------------------------------------
# Shape normalisation: make all KV layers consistently 2D [rows, head_dim]
# ---------------------------------------------------------------------------

def _normalize_cache_shapes(
    cache_data: List,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Normalize per-layer (keys, values) to 2D [rows, head_dim].

    SSD cache blocks may store layers in mixed formats:
    - 4D ``[1, heads, tokens, dim]`` → squeeze → 3D
    - 3D ``[heads, tokens, dim]``    → reshape → ``[heads*tokens, dim]``
    - 2D ``[tokens, heads*dim]``     → reshape → ``[heads*tokens, dim]``

    We detect ``head_dim`` from the 3D tensors (last axis) and use it to
    reshape any 2D tensors whose last axis is a multiple of ``head_dim``.
    """
    # Pass 1: convert to numpy, squeeze leading batch dims, collect shapes.
    stage1: List[Tuple[np.ndarray, np.ndarray]] = []
    key_3d_dims: List[int] = []
    val_3d_dims: List[int] = []

    for layer_data in cache_data:
        if not (isinstance(layer_data, (tuple, list)) and len(layer_data) == 2):
            continue
        k, v = layer_data
        k_arr = np.array(k) if not isinstance(k, np.ndarray) else np.array(k)
        v_arr = np.array(v) if not isinstance(v, np.ndarray) else np.array(v)
        while k_arr.ndim > 3:
            k_arr = k_arr.squeeze(0)
        while v_arr.ndim > 3:
            v_arr = v_arr.squeeze(0)
        if k_arr.ndim == 3:
            key_3d_dims.append(k_arr.shape[-1])
        if v_arr.ndim == 3:
            val_3d_dims.append(v_arr.shape[-1])
        stage1.append((k_arr, v_arr))

    # Determine canonical head_dim from 3D tensors (most common last dim).
    k_head_dim = _most_common(key_3d_dims) if key_3d_dims else None
    v_head_dim = _most_common(val_3d_dims) if val_3d_dims else None

    logger.debug(
        "Shape normalization: %d layers, k_head_dim=%s, v_head_dim=%s",
        len(stage1), k_head_dim, v_head_dim,
    )

    # Pass 2: flatten everything to 2D [rows, head_dim].
    normalized: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, (k_arr, v_arr) in enumerate(stage1):
        k_before = k_arr.shape
        v_before = v_arr.shape
        k_arr = _flatten_to_2d(k_arr, k_head_dim)
        v_arr = _flatten_to_2d(v_arr, v_head_dim)
        if i < 3 or k_arr.shape[-1] != (k_head_dim or k_arr.shape[-1]):
            logger.debug(
                "  layer %d: k %s→%s, v %s→%s",
                i, k_before, k_arr.shape, v_before, v_arr.shape,
            )
        normalized.append((k_arr, v_arr))

    # Final check
    k_dims = {n[0].shape[-1] for n in normalized}
    v_dims = {n[1].shape[-1] for n in normalized}
    logger.info(
        "Normalized shapes: key_dims=%s, val_dims=%s (%d layers)",
        k_dims, v_dims, len(normalized),
    )

    return normalized


def _flatten_to_2d(arr: np.ndarray, head_dim: int | None) -> np.ndarray:
    """Flatten an array to 2D ``[rows, feature_dim]``.

    - 3D ``[heads, tokens, dim]`` → ``[heads*tokens, dim]``
    - 2D ``[tokens, total_dim]``  → reshape to ``[tokens*heads, head_dim]``
      when ``total_dim`` is a multiple of ``head_dim``
    """
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2 and head_dim is not None:
        total = arr.shape[-1]
        if total != head_dim and total % head_dim == 0:
            n_heads = total // head_dim
            # [tokens, n_heads*head_dim] → [tokens, n_heads, head_dim]
            #  → [n_heads, tokens, head_dim] → [n_heads*tokens, head_dim]
            tokens = arr.shape[0]
            return arr.reshape(tokens, n_heads, head_dim).transpose(1, 0, 2).reshape(-1, head_dim)
        return arr
    # 1D or 0D — unlikely but return as-is
    return arr


def _most_common(values: List[int]) -> int:
    """Return the most frequent value."""
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Encoding helpers: serialize KVTC encoded data into safetensors arrays
# ---------------------------------------------------------------------------

def _save_encoded_layer(
    arrays: Dict[str, mx.array],
    prefix: str,
    encoded: tuple,
    orig_shape: tuple,
) -> None:
    """Serialize KVTC-encoded data into the arrays dict for safetensors."""
    payloads, shifts, scales, q_shapes, flat_shape = encoded

    # Save number of blocks
    arrays[f"{prefix}_n_blocks"] = mx.array(np.array([len(payloads)], dtype=np.int32))
    arrays[f"{prefix}_orig_shape"] = mx.array(np.array(orig_shape, dtype=np.int32))
    arrays[f"{prefix}_flat_shape"] = mx.array(flat_shape)

    for b_idx in range(len(payloads)):
        arrays[f"{prefix}_b{b_idx}_payload"] = mx.array(payloads[b_idx])
        arrays[f"{prefix}_b{b_idx}_shifts"] = mx.array(shifts[b_idx])
        arrays[f"{prefix}_b{b_idx}_scales"] = mx.array(scales[b_idx])
        arrays[f"{prefix}_b{b_idx}_qshape"] = mx.array(q_shapes[b_idx])


def _load_encoded_layer(
    arrays: Dict[str, mx.array],
    prefix: str,
) -> Optional[tuple]:
    """Deserialize KVTC-encoded data from safetensors arrays."""
    n_key = f"{prefix}_n_blocks"
    if n_key not in arrays:
        return None

    n_blocks = int(np.array(arrays[n_key])[0])
    orig_shape = arrays[f"{prefix}_orig_shape"]
    flat_shape = arrays[f"{prefix}_flat_shape"]

    payloads = []
    shifts = []
    scales = []
    q_shapes = []

    for b_idx in range(n_blocks):
        payloads.append(np.array(arrays[f"{prefix}_b{b_idx}_payload"]))
        shifts.append(np.array(arrays[f"{prefix}_b{b_idx}_shifts"]))
        scales.append(np.array(arrays[f"{prefix}_b{b_idx}_scales"]))
        q_shapes.append(np.array(arrays[f"{prefix}_b{b_idx}_qshape"]))

    # Return as (payloads, shifts, scales, q_shapes, flat_shape, orig_shape)
    return (
        tuple(payloads),
        tuple(shifts),
        tuple(scales),
        tuple(q_shapes),
        np.array(flat_shape),
        np.array(orig_shape),  # Extra: original shape for reconstruction
    )
