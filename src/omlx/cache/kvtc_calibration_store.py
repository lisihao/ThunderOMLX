# SPDX-License-Identifier: Apache-2.0
"""Per-model KVTC calibration persistence.

Calibration data (PCA basis + DP bit-allocation plan) is expensive to compute
(SVD + DP over sample data) but stable per model architecture. This store
persists calibrations to disk so they are computed only once per model.

Storage: ~/.omlx/cache/kvtc_calibrations/{model_hash}.safetensors
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

from .kvtc_codec import (
    KVTCCodecConfig,
    KVTCSharedCalibration,
    KVTCTransformPlan,
    fit_shared_calibration,
)

logger = logging.getLogger(__name__)


def _model_key(model_name: str) -> str:
    """Normalize model name to a filesystem-safe key."""
    return hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]


class KVTCCalibrationStore:
    """Per-model KVTC calibration persistence.

    Calibrations are saved as safetensors files containing the PCA basis,
    mean vectors, and DP block-meta arrays for both keys and values.
    """

    def __init__(self, calibrations_dir: Optional[Path] = None):
        if calibrations_dir is None:
            calibrations_dir = Path.home() / ".omlx" / "cache" / "kvtc_calibrations"
        self._dir = Path(calibrations_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, model_name: str) -> Path:
        return self._dir / f"{_model_key(model_name)}.safetensors"

    def _meta_path_for(self, model_name: str) -> Path:
        return self._dir / f"{_model_key(model_name)}.json"

    def save(self, model_name: str, calibration: KVTCSharedCalibration) -> Path:
        """Save calibration to disk as safetensors + JSON metadata."""
        path = self._path_for(model_name)
        meta_path = self._meta_path_for(model_name)

        # Build safetensors-compatible dict of arrays
        arrays: Dict[str, mx.array] = {}
        for side in ("keys", "values"):
            plan: KVTCTransformPlan = getattr(calibration, side)
            arrays[f"{side}_mean"] = mx.array(plan.mean)
            arrays[f"{side}_basis"] = mx.array(plan.basis)
            arrays[f"{side}_block_meta"] = mx.array(plan.block_meta)

        mx.save_safetensors(str(path), arrays)

        # Save metadata (config + fingerprint + model name)
        meta = {
            "model_name": model_name,
            "fingerprint": calibration.fingerprint(),
            "keys_config": json.loads(calibration.keys.meta_state),
            "values_config": json.loads(calibration.values.meta_state),
            "saved_at": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(
            "KVTC calibration saved: model=%s fingerprint=%s path=%s",
            model_name, meta["fingerprint"], path,
        )
        return path

    def load(self, model_name: str) -> Optional[KVTCSharedCalibration]:
        """Load calibration from disk. Returns None if not found."""
        path = self._path_for(model_name)
        meta_path = self._meta_path_for(model_name)

        if not path.exists() or not meta_path.exists():
            return None

        try:
            arrays = mx.load(str(path))
            meta = json.loads(meta_path.read_text())

            keys_config = KVTCCodecConfig(**meta["keys_config"])
            values_config = KVTCCodecConfig(**meta["values_config"])

            keys_plan = KVTCTransformPlan(
                mean=np.array(arrays["keys_mean"]),
                basis=np.array(arrays["keys_basis"]),
                block_meta=np.array(arrays["keys_block_meta"]).astype(np.int32),
                config=keys_config,
            )
            values_plan = KVTCTransformPlan(
                mean=np.array(arrays["values_mean"]),
                basis=np.array(arrays["values_basis"]),
                block_meta=np.array(arrays["values_block_meta"]).astype(np.int32),
                config=values_config,
            )

            calibration = KVTCSharedCalibration(keys=keys_plan, values=values_plan)

            # Verify fingerprint
            expected = meta.get("fingerprint", "")
            actual = calibration.fingerprint()
            if expected and expected != actual:
                logger.warning(
                    "KVTC calibration fingerprint mismatch for %s: "
                    "expected=%s actual=%s — re-fitting recommended",
                    model_name, expected, actual,
                )
                return None

            logger.info(
                "KVTC calibration loaded: model=%s fingerprint=%s",
                model_name, actual,
            )
            return calibration

        except Exception as e:
            logger.error("Failed to load KVTC calibration for %s: %s", model_name, e)
            return None

    def get_or_fit(
        self,
        model_name: str,
        sample_cache_data: List,
        config: Optional[KVTCCodecConfig] = None,
    ) -> KVTCSharedCalibration:
        """Load existing calibration or fit from sample data.

        Args:
            model_name: Model identifier for cache lookup.
            sample_cache_data: List of (keys, values) tuples per layer,
                where each is an mx.array or np.ndarray of shape
                [heads, tokens, dim]. Multiple layers are flattened for
                calibration.
            config: Codec config. Uses defaults if None.

        Returns:
            Fitted or loaded KVTCSharedCalibration.
        """
        existing = self.load(model_name)
        if existing is not None:
            return existing

        if config is None:
            config = KVTCCodecConfig()

        t0 = time.time()

        # Flatten cache_data into 2D matrices for calibration
        key_matrices: List[np.ndarray] = []
        value_matrices: List[np.ndarray] = []

        for layer_keys, layer_values in sample_cache_data:
            k = np.array(layer_keys) if not isinstance(layer_keys, np.ndarray) else layer_keys
            v = np.array(layer_values) if not isinstance(layer_values, np.ndarray) else layer_values
            # Reshape [heads, tokens, dim] -> [heads*tokens, dim]
            if k.ndim == 3:
                k = k.reshape(-1, k.shape[-1])
            if v.ndim == 3:
                v = v.reshape(-1, v.shape[-1])
            key_matrices.append(k.astype(np.float32))
            value_matrices.append(v.astype(np.float32))

        calibration = fit_shared_calibration(key_matrices, value_matrices, config)

        elapsed = time.time() - t0
        logger.info(
            "KVTC calibration fitted: model=%s fingerprint=%s elapsed=%.2fs",
            model_name, calibration.fingerprint(), elapsed,
        )

        self.save(model_name, calibration)
        return calibration

    def list_calibrations(self) -> List[Dict[str, Any]]:
        """List all saved calibrations with metadata."""
        results = []
        for meta_path in sorted(self._dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text())
                safetensors_path = meta_path.with_suffix(".safetensors")
                meta["file_size_bytes"] = (
                    safetensors_path.stat().st_size if safetensors_path.exists() else 0
                )
                results.append(meta)
            except Exception as e:
                logger.warning("Skipping corrupt calibration %s: %s", meta_path, e)
        return results

    def delete(self, model_name: str) -> bool:
        """Delete calibration for a model. Returns True if deleted."""
        path = self._path_for(model_name)
        meta_path = self._meta_path_for(model_name)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        if deleted:
            logger.info("KVTC calibration deleted: model=%s", model_name)
        return deleted
