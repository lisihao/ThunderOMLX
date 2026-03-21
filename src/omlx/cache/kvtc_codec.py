# SPDX-License-Identifier: Apache-2.0
"""KVTC (KV Cache Transform Coding) codec for ThunderOMLX.

Ported from FlashMLX implementation based on arXiv 2511.01815 (ICLR 2026).

Three-stage lossy compression pipeline:
  1. PCA decorrelation - fit shared basis, project to lower-rank coefficients
  2. DP bit allocation - dynamic programming to assign per-block precision
  3. Affine quantization + DEFLATE entropy coding

Designed for prompt-cache persistence (disk/network compression), not runtime
memory compression. Achieves 4-8x compression with <1% perplexity loss.

Dependencies: mlx.core (for array conversion), numpy (core computation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import zlib
from typing import Optional, Sequence

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class KVTCCodecConfig:
    """Codec knobs for KVTC calibration and encoding."""

    energy: float = 0.995
    rank: Optional[int] = None
    max_rank: int = 64
    bits: int = 4
    group_size: int = 64
    sample_limit: int = 4096
    seed: int = 0
    allowed_block_sizes: tuple[int, ...] = (1, 16, 64, 256, 1024)
    allowed_bits: tuple[int, ...] = (0, 2, 4, 8)
    scale_overhead_bits: int = 64
    zero_bit_penalty_bits: int = 8
    zero_bit_energy_fraction: float = 0.015


@dataclass
class KVTCTransformPlan:
    """Shared transform and bit-allocation plan for a single tensor family."""

    mean: np.ndarray
    basis: np.ndarray
    block_meta: np.ndarray
    config: KVTCCodecConfig

    @property
    def state(self):
        return self.mean, self.basis, self.block_meta

    @property
    def meta_state(self):
        return json.dumps(asdict(self.config))

    @classmethod
    def from_state(cls, state, meta_state):
        if isinstance(meta_state, str):
            config = KVTCCodecConfig(**json.loads(meta_state))
        else:
            config = KVTCCodecConfig(**dict(meta_state))
        mean, basis, block_meta = state
        return cls(
            mean=_to_numpy(mean).astype(np.float32, copy=False),
            basis=_to_numpy(basis).astype(np.float32, copy=False),
            block_meta=_to_numpy(block_meta).astype(np.int32, copy=False),
            config=config,
        )

    def encode(self, x: np.ndarray):
        x = _to_numpy(x).astype(np.float32, copy=False)
        coeffs = project(x, self.mean, self.basis)
        payloads = []
        shifts = []
        scales = []
        q_shapes = []

        for start, width, bits in self.block_meta:
            block = coeffs[:, start : start + width]
            if int(bits) == 0:
                payloads.append(np.zeros(1, dtype=np.uint8))
                shifts.append(np.zeros(1, dtype=np.float32))
                scales.append(np.zeros(1, dtype=np.float32))
                q_shapes.append(np.asarray(block.shape, dtype=np.int32))
                continue
            group_size = min(self.config.group_size, int(width))
            payload, block_shifts, block_scales, q_shape = quantize_groups(
                block, int(bits), group_size
            )
            payloads.append(payload)
            shifts.append(block_shifts)
            scales.append(block_scales)
            q_shapes.append(np.asarray(q_shape, dtype=np.int32))

        return (
            tuple(payloads),
            tuple(shifts),
            tuple(scales),
            tuple(q_shapes),
            np.asarray(x.shape, dtype=np.int32),
        )

    def decode(self, encoded):
        if len(encoded) == 4:
            payloads, scales, q_shapes, orig_shape = encoded
            shifts = tuple(np.zeros_like(_to_numpy(scale), dtype=np.float32) for scale in scales)
        else:
            payloads, shifts, scales, q_shapes, orig_shape = encoded
        coeffs = np.zeros((int(orig_shape[0]), self.basis.shape[1]), dtype=np.float32)

        for idx, (start, width, bits) in enumerate(self.block_meta):
            if int(bits) == 0:
                continue
            group_size = min(self.config.group_size, int(width))
            block = dequantize_groups(
                payloads[idx],
                q_shapes[idx],
                shifts[idx],
                scales[idx],
                int(bits),
                group_size,
            )
            coeffs[:, start : start + width] = block

        reconstructed = reconstruct(coeffs, self.mean, self.basis)
        return reconstructed.reshape(tuple(int(v) for v in _to_numpy(orig_shape).tolist()))

    def fingerprint(self) -> str:
        """Stable identifier for sharing calibration across caches."""
        h = hashlib.sha1()
        h.update(np.asarray(self.mean.shape, dtype=np.int32).tobytes())
        h.update(np.asarray(self.basis.shape, dtype=np.int32).tobytes())
        h.update(np.asarray(self.block_meta.shape, dtype=np.int32).tobytes())
        h.update(self.mean.tobytes())
        h.update(self.basis.tobytes())
        h.update(self.block_meta.tobytes())
        h.update(self.meta_state.encode("utf-8"))
        return h.hexdigest()


@dataclass
class KVTCSharedCalibration:
    """Shared key/value calibration used by a prompt-cache group."""

    keys: KVTCTransformPlan
    values: KVTCTransformPlan

    @property
    def state(self):
        return {"keys": self.keys.state, "values": self.values.state}

    @property
    def meta_state(self):
        return {
            "keys": self.keys.meta_state,
            "values": self.values.meta_state,
        }

    @classmethod
    def from_state(cls, state, meta_state):
        return cls(
            keys=KVTCTransformPlan.from_state(state["keys"], meta_state["keys"]),
            values=KVTCTransformPlan.from_state(
                state["values"], meta_state["values"]
            ),
        )

    def fingerprint(self) -> str:
        return hashlib.sha1(
            (self.keys.fingerprint() + "::" + self.values.fingerprint()).encode("utf-8")
        ).hexdigest()

    def encode(self, keys, values):
        return self.keys.encode(keys), self.values.encode(values)

    def decode(self, encoded_keys, encoded_values):
        return self.keys.decode(encoded_keys), self.values.decode(encoded_values)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        if hasattr(x, "astype"):
            try:
                return np.asarray(x.astype(mx.float32))
            except Exception:
                pass
        if hasattr(x, "tolist"):
            return np.asarray(x.tolist())
        raise


def _subsample_rows(x: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if x.shape[0] <= limit:
        return x
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=limit, replace=False))
    return x[indices]


# ---------------------------------------------------------------------------
# PCA fitting and projection
# ---------------------------------------------------------------------------

def _randomized_svd(M: np.ndarray, n_components: int, n_oversamples: int = 10, seed: int = 0):
    """Fast randomized SVD via Halko-Martinsson-Tropp algorithm.

    Computes only the top ``n_components`` singular values/vectors,
    much faster than full SVD for low-rank approximations.
    """
    rng = np.random.default_rng(seed)
    n_random = min(n_components + n_oversamples, M.shape[1])
    Omega = rng.standard_normal((M.shape[1], n_random)).astype(np.float32)
    Y = M @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ M
    Uhat, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Uhat, s, Vt


def fit_pca_basis(x, config: KVTCCodecConfig):
    """Fit a PCA basis for a 2D array."""
    x = _to_numpy(x).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")

    x_fit = _subsample_rows(x, config.sample_limit, config.seed)
    mean = x_fit.mean(axis=0, keepdims=True)
    centered = x_fit - mean

    dim = centered.shape[1]
    target_rank = config.max_rank if config.rank is None else min(config.rank, config.max_rank)

    if target_rank < dim:
        # Randomized SVD — only compute top target_rank components
        _, singular_values, vt = _randomized_svd(centered, target_rank, seed=config.seed)
    else:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    if config.rank is not None:
        rank = max(1, min(config.rank, vt.shape[0], config.max_rank))
    else:
        total = np.sum(singular_values**2)
        if total == 0:
            rank = 1
        else:
            energy = np.cumsum(singular_values**2) / total
            rank = int(np.searchsorted(energy, config.energy)) + 1
            rank = max(1, min(rank, vt.shape[0], config.max_rank))

    basis = vt[:rank].T.astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), basis


def project(x, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = _to_numpy(x).astype(np.float32, copy=False)
    return (x - mean) @ basis


def reconstruct(coeffs, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coeffs = _to_numpy(coeffs).astype(np.float32, copy=False)
    return coeffs @ basis.T + mean


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_groups(x, bits: int, group_size: int):
    """Per-group affine quantization plus DEFLATE compression."""
    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")
    x = _to_numpy(x).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")

    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"Unsupported bit-width: {bits}")

    n_cols = x.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    q = np.zeros_like(x, dtype=np.int8)
    shifts = np.zeros(n_groups, dtype=np.float32)
    scales = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        chunk = x[:, start:end]
        shift = float(np.mean(chunk))
        centered = chunk - shift
        scale = float(np.max(np.abs(centered)))
        scale = scale / qmax if scale > 0 else 1.0
        shifts[g] = shift
        scales[g] = scale
        q[:, start:end] = np.clip(
            np.rint(centered / scale), -qmax - 1, qmax
        ).astype(np.int8)

    payload = zlib.compress(q.tobytes(), level=9)
    return (
        np.frombuffer(payload, dtype=np.uint8).copy(),
        shifts,
        scales,
        np.asarray(q.shape, dtype=np.int32),
    )


def _quantize_groups_raw(x, bits: int, group_size: int):
    """Quantize groups without DEFLATE (used by DP planner)."""
    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")
    x = _to_numpy(x).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {x.shape}")

    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"Unsupported bit-width: {bits}")

    n_cols = x.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    q = np.zeros_like(x, dtype=np.int8)
    shifts = np.zeros(n_groups, dtype=np.float32)
    scales = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        chunk = x[:, start:end]
        shift = float(np.mean(chunk))
        centered = chunk - shift
        scale = float(np.max(np.abs(centered)))
        scale = scale / qmax if scale > 0 else 1.0
        shifts[g] = shift
        scales[g] = scale
        q[:, start:end] = np.clip(
            np.rint(centered / scale), -qmax - 1, qmax
        ).astype(np.int8)

    return q, shifts, scales, np.asarray(q.shape, dtype=np.int32)


def dequantize_groups(
    payload, q_shape, shifts, scales, bits: int, group_size: int
) -> np.ndarray:
    if bits < 2 or bits > 8:
        raise ValueError(f"Unsupported bit-width: {bits}")

    raw = zlib.decompress(_to_numpy(payload).astype(np.uint8, copy=False).tobytes())
    q_shape = tuple(int(v) for v in _to_numpy(q_shape).tolist())
    shifts = _to_numpy(shifts).astype(np.float32, copy=False)
    q = np.frombuffer(raw, dtype=np.int8).reshape(q_shape)
    q = q.astype(np.float32, copy=False)

    n_cols = q.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    out = np.zeros_like(q, dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        out[:, start:end] = q[:, start:end] * float(scales[g]) + float(shifts[g])
    return out


# ---------------------------------------------------------------------------
# DP bit allocation
# ---------------------------------------------------------------------------

def _collect_rows(matrices: Sequence[np.ndarray], config: KVTCCodecConfig) -> np.ndarray:
    rows = []
    for i, matrix in enumerate(matrices):
        mat = _to_numpy(matrix).astype(np.float32, copy=False)
        if mat.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {mat.shape}")
        sampled = _subsample_rows(
            mat,
            max(1, config.sample_limit // max(1, len(matrices))),
            config.seed + i,
        )
        rows.append(sampled)
    if not rows:
        raise ValueError("No matrices were provided for calibration")
    return np.concatenate(rows, axis=0)


def _quantize_block_error(block: np.ndarray, bits: int, group_size: int) -> float:
    if bits == 0:
        return float(np.sum(block * block))
    q, shifts, scales, q_shape = _quantize_groups_raw(block, bits, group_size)
    recon = np.zeros_like(q, dtype=np.float32)
    q_float = q.astype(np.float32, copy=False)
    n_cols = q.shape[1]
    n_groups = (n_cols + group_size - 1) // group_size
    for g in range(n_groups):
        start = g * group_size
        end = min(n_cols, start + group_size)
        recon[:, start:end] = q_float[:, start:end] * float(scales[g]) + float(shifts[g])
    diff = block - recon
    return float(np.sum(diff * diff))


def _estimate_block_cost(block: np.ndarray, bits: int, group_size: int, config: KVTCCodecConfig) -> int:
    if bits == 0:
        penalty_bytes = int(np.ceil(config.zero_bit_penalty_bits / 8))
        return max(1, penalty_bytes)
    q, _, _, q_shape = _quantize_groups_raw(block, bits, group_size)
    n_groups = int(np.ceil(q_shape[1] / group_size))
    payload_bytes = int(np.ceil(q.size * bits / 8.0))
    return int(payload_bytes + n_groups * max(1, config.scale_overhead_bits // 8))


def plan_bit_allocation(coeffs: np.ndarray, config: KVTCCodecConfig) -> np.ndarray:
    """Compute a DP-based blockwise precision plan.

    The plan is stored as ``(start, width, bits)`` rows in ascending order.

    Note: The DP budget is per-coefficient (not per-sample), so we subsample
    the coefficient matrix to a small number of representative rows before
    running the planner. This ensures cost estimates fit within budget
    regardless of input size, while PCA fitting can use the full sample.
    """
    coeffs = _to_numpy(coeffs).astype(np.float32, copy=False)
    if coeffs.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {coeffs.shape}")

    # Subsample rows for DP planning — the planner only needs representative
    # cost/error estimates, not all samples. Budget is O(rank), not O(samples).
    _DP_SAMPLE_LIMIT = 8
    if coeffs.shape[0] > _DP_SAMPLE_LIMIT:
        coeffs = _subsample_rows(coeffs, _DP_SAMPLE_LIMIT, config.seed + 9999)

    rank = coeffs.shape[1]
    if rank == 0:
        return np.zeros((0, 3), dtype=np.int32)

    allowed_block_sizes = sorted({int(s) for s in config.allowed_block_sizes if int(s) > 0})
    allowed_block_sizes = [s for s in allowed_block_sizes if s <= rank]
    if not allowed_block_sizes:
        allowed_block_sizes = [1]

    allowed_bits = sorted({int(b) for b in config.allowed_bits if int(b) >= 0})
    if 0 not in allowed_bits:
        allowed_bits = [0] + allowed_bits

    budget = max(
        1,
        int(rank * max(1, config.bits + config.scale_overhead_bits // 8)),
    )

    n_widths = len(allowed_block_sizes)
    n_bits = len(allowed_bits)
    n_rows = coeffs.shape[0]
    scale_oh = max(1, config.scale_overhead_bits // 8)
    zero_cost = max(1, int(np.ceil(config.zero_bit_penalty_bits / 8)))

    # --- Pre-compute error/cost tables for all unique (start, width, bits) ---
    # This replaces ~3.9M _quantize_block_error calls with ~rank×n_widths×n_bits
    error_table = np.full((rank, n_widths, n_bits), np.inf, dtype=np.float64)
    cost_table = np.full((rank, n_widths, n_bits), budget + 1, dtype=np.int32)
    energy_table = np.zeros((rank, n_widths), dtype=np.float64)

    for w_idx, width in enumerate(allowed_block_sizes):
        for start in range(rank - width + 1):
            block = coeffs[:, start : start + width]
            block_energy = float(np.sum(block * block))
            energy_table[start, w_idx] = block_energy
            group_size = min(config.group_size, width)

            for b_idx, bits in enumerate(allowed_bits):
                if bits == 0:
                    error_table[start, w_idx, b_idx] = block_energy
                    cost_table[start, w_idx, b_idx] = zero_cost
                else:
                    error_table[start, w_idx, b_idx] = _quantize_block_error(
                        block, bits, group_size
                    )
                    n_groups = int(np.ceil(width / group_size))
                    payload_bytes = int(np.ceil(n_rows * width * bits / 8.0))
                    cost_table[start, w_idx, b_idx] = payload_bytes + n_groups * scale_oh

    total_energy = float(np.sum(coeffs * coeffs))
    energy_thresh = config.zero_bit_energy_fraction * total_energy

    # --- DP with table lookups (no function calls in inner loop) ---
    best_error = np.full((rank + 1, budget + 1), np.inf, dtype=np.float64)
    prev_i = np.full((rank + 1, budget + 1), -1, dtype=np.int32)
    prev_b = np.full((rank + 1, budget + 1), -1, dtype=np.int32)
    choice_width = np.zeros((rank + 1, budget + 1), dtype=np.int32)
    choice_bits = np.zeros((rank + 1, budget + 1), dtype=np.int32)

    best_error[0, :] = 0.0

    for i in range(1, rank + 1):
        for b in range(0, budget + 1):
            if b > 0 and best_error[i, b - 1] <= best_error[i, b]:
                best_error[i, b] = best_error[i, b - 1]
                prev_i[i, b] = prev_i[i, b - 1]
                prev_b[i, b] = prev_b[i, b - 1]
                choice_width[i, b] = choice_width[i, b - 1]
                choice_bits[i, b] = choice_bits[i, b - 1]

            for w_idx in range(n_widths):
                width = allowed_block_sizes[w_idx]
                if width > i:
                    continue
                start = i - width
                be = energy_table[start, w_idx]

                for b_idx in range(n_bits):
                    bits = allowed_bits[b_idx]
                    if bits == 0 and total_energy > 0 and be > energy_thresh:
                        continue
                    cost = int(cost_table[start, w_idx, b_idx])
                    if cost > b:
                        continue
                    error = error_table[start, w_idx, b_idx]

                    candidate = best_error[start, b - cost] + error
                    if candidate < best_error[i, b]:
                        best_error[i, b] = candidate
                        prev_i[i, b] = start
                        prev_b[i, b] = b - cost
                        choice_width[i, b] = width
                        choice_bits[i, b] = bits

    # Reconstruct the plan from the strongest budget.
    b = budget
    while b > 0 and choice_width[rank, b] == 0 and prev_i[rank, b] < 0:
        b -= 1

    blocks = []
    i = rank
    while i > 0:
        if choice_width[i, b] == 0 and prev_i[i, b] < 0:
            b -= 1
            if b < 0:
                break
            continue
        width = int(choice_width[i, b])
        bits = int(choice_bits[i, b])
        start = int(prev_i[i, b])
        if width <= 0 or start < 0:
            break
        blocks.append((start, width, bits))
        next_i = int(prev_i[i, b])
        next_b = int(prev_b[i, b])
        i = next_i
        b = next_b

    if not blocks:
        return np.asarray([(0, rank, 0)], dtype=np.int32)

    blocks.sort(key=lambda row: row[0])
    return np.asarray(blocks, dtype=np.int32)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def fit_transform_plan(matrices: Sequence[np.ndarray], config: KVTCCodecConfig) -> KVTCTransformPlan:
    """Fit a shared plan from one or more 2D calibration matrices."""
    combined = _collect_rows(matrices, config)
    mean, basis = fit_pca_basis(combined, config)
    coeffs = project(combined, mean, basis)
    block_meta = plan_bit_allocation(coeffs, config)
    return KVTCTransformPlan(mean=mean, basis=basis, block_meta=block_meta, config=config)


def fit_shared_calibration(
    key_matrices: Sequence[np.ndarray],
    value_matrices: Sequence[np.ndarray],
    config: KVTCCodecConfig,
) -> KVTCSharedCalibration:
    """Fit a shared key/value calibration that can be reused across layers."""
    key_plan = fit_transform_plan(key_matrices, config)
    value_plan = fit_transform_plan(value_matrices, config)
    return KVTCSharedCalibration(keys=key_plan, values=value_plan)


def encode_tensor(x, plan: KVTCTransformPlan):
    """Encode a 2D matrix with a fitted transform plan."""
    return plan.encode(x)


def decode_tensor(encoded, plan: KVTCTransformPlan) -> np.ndarray:
    """Decode a tensor encoded with :func:`encode_tensor`."""
    return plan.decode(encoded)


# ---------------------------------------------------------------------------
# Binary serialization helpers for SSD cache integration (Phase 3)
# ---------------------------------------------------------------------------

import io
import pickle


def encode_tensor_to_bytes(x: np.ndarray, plan: KVTCTransformPlan) -> bytes:
    """Encode a 2D numpy array and serialize the result to bytes.

    Uses pickle + zlib for the encoded tuple (payloads, shifts, scales,
    q_shapes, orig_shape).  The result is a compact binary blob suitable
    for SSD storage.
    """
    encoded = plan.encode(x)
    buf = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(buf, level=1)  # fast compression on already-quantized data


def decode_tensor_from_bytes(data: bytes, plan: KVTCTransformPlan) -> np.ndarray:
    """Deserialize bytes produced by :func:`encode_tensor_to_bytes` and decode."""
    buf = zlib.decompress(data)
    encoded = pickle.loads(buf)  # noqa: S301 — trusted internal data
    return plan.decode(encoded)


def encode_block_to_bytes(
    cache_data_numpy: list[tuple[np.ndarray, np.ndarray]],
    calibration: "KVTCSharedCalibration",
) -> bytes:
    """Encode all layers of a KV cache block into a single bytes blob.

    Args:
        cache_data_numpy: Per-layer list of (keys_2d, values_2d) numpy arrays.
            Each array should be 2D [tokens*heads, head_dim] or at least
            reshapeable to 2D.
        calibration: Shared KVTC calibration fitted for this model.

    Returns:
        Compressed bytes blob containing all encoded layers.
    """
    encoded_layers = []
    for keys_np, values_np in cache_data_numpy:
        # Flatten to 2D if needed (e.g. [heads, tokens, dim] → [heads*tokens, dim])
        if keys_np.ndim > 2:
            keys_np = keys_np.reshape(-1, keys_np.shape[-1])
        if values_np.ndim > 2:
            values_np = values_np.reshape(-1, values_np.shape[-1])
        enc_k = calibration.keys.encode(keys_np.astype(np.float32, copy=False))
        enc_v = calibration.values.encode(values_np.astype(np.float32, copy=False))
        encoded_layers.append((enc_k, enc_v))

    buf = pickle.dumps(encoded_layers, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(buf, level=1)


def decode_block_from_bytes(
    data: bytes,
    calibration: "KVTCSharedCalibration",
    original_shapes: list[tuple[tuple, tuple]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode a bytes blob produced by :func:`encode_block_to_bytes`.

    Args:
        data: Compressed bytes blob.
        calibration: Same calibration used for encoding.
        original_shapes: Optional list of ((keys_shape), (values_shape)) per layer
            to reshape decoded 2D arrays back to original shapes.

    Returns:
        Per-layer list of (keys, values) numpy arrays.
    """
    buf = zlib.decompress(data)
    encoded_layers = pickle.loads(buf)  # noqa: S301 — trusted internal data

    decoded_layers = []
    for idx, (enc_k, enc_v) in enumerate(encoded_layers):
        dec_k = calibration.keys.decode(enc_k)
        dec_v = calibration.values.decode(enc_v)
        if original_shapes and idx < len(original_shapes):
            k_shape, v_shape = original_shapes[idx]
            if k_shape:
                dec_k = dec_k.reshape(k_shape)
            if v_shape:
                dec_v = dec_v.reshape(v_shape)
        decoded_layers.append((dec_k, dec_v))

    return decoded_layers
