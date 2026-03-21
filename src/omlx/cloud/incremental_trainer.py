"""Incremental training for MF Router using production routing data.

Extracts the core training logic from scripts/train_mf_incremental.py into
an importable module.  Mixes production preference pairs (from RoutingStore)
with the original Chatbot Arena dataset to fine-tune the MF Router checkpoint.
"""

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from omlx.cloud.preference_labeler import convert_to_training_format

logger = logging.getLogger("omlx.cloud.incremental_trainer")

# ---------------------------------------------------------------------------
# Constants (mirrored from train_mf_gemini.py)
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "gpt-4-1106-preview": 21, "gpt-4-0613": 20, "gpt-4-0314": 19,
    "gpt-4-0125-preview": 18, "gpt-4-turbo-2024-04-09": 22,
    "gpt-3.5-turbo-0613": 15, "gpt-3.5-turbo-0125": 13,
    "claude-3-opus-20240229": 2, "claude-3-sonnet-20240229": 3,
    "claude-3-haiku-20240307": 1, "claude-2.1": 0, "claude-2.0": 4,
    "command-r-plus": 6, "command-r": 5, "dbrx-instruct": 7,
    "gemini-pro": 8, "gemini-pro-dev-api": 9,
    "gemma-1.1-7b-it": 10, "gemma-7b-it": 11,
    "llama-2-13b-chat": 25, "llama-2-70b-chat": 26, "llama-2-7b-chat": 27,
    "mixtral-8x7b-instruct-v0.1": 36, "mistral-7b-instruct-v0.2": 35,
    "mistral-7b-instruct": 34, "mistral-large-2402": 37, "mistral-medium": 38,
    "phi-3-mini-4k-instruct": 42, "phi-3-small-8k-instruct": 43,
    "qwen1.5-72b-chat": 46, "qwen1.5-7b-chat": 48, "qwen1.5-110b-chat": 44,
    "yi-34b-chat": 61, "vicuna-13b": 55, "vicuna-33b": 56, "vicuna-7b": 57,
    "zephyr-7b-beta": 63, "codellama-70b-instruct": 24,
    "deepseek-llm-67b-chat": 23, "starling-lm-7b-beta": 52,
    "tulu-2-dpo-70b": 53, "openchat-3.5-0106": 40,
    "wizardlm-70b": 60, "wizardlm-13b": 59, "chatglm3-6b": 12,
    "llama-3-70b-instruct": 30, "llama-3-8b-instruct": 31,
    "gemma-1.1-2b-it": 32, "olmo-7b-instruct": 41, "pplx-70b-online": 39,
    "gpt-4o-2024-05-13": 16, "gemini-1.5-pro-api-0409-preview": 17,
    "phi-3-medium-4k-instruct": 33, "snowflake-arctic-instruct": 51,
    "reka-flash-21b-20240226-online": 49,
    "qwen1.5-14b-chat": 45, "qwen1.5-32b-chat": 47,
    "mistral-7b-instruct-v0.1": 29, "snorkel-mistral-pairrm-dpo": 50,
    "llama-2-chat": 28, "yi-1.5-34b-chat": 62,
    "llama-3-70b-instruct-nitro": 54, "reka-flash": 58,
}
ID_TO_MODEL = {v: k for k, v in MODEL_IDS.items()}
NUM_MODELS = 64
STRONG_ID = 21  # gpt-4-1106-preview
WEAK_ID = 36    # mixtral-8x7b-instruct-v0.1
EMBED_DIM = 3072  # gemini-embedding-001


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    routing_db: str = os.path.expanduser("~/.omlx/routing.db")
    arena_embeddings: str = "models/mf-router/cache/gemini_embeddings.npy"
    arena_metadata: str = "models/mf-router/cache/gemini_metadata.npz"
    old_checkpoint: str = "models/mf-router/model_gemini.safetensors"
    output: str = "models/mf-router/model_gemini_v2.safetensors"
    mix_ratio: float = 0.3
    epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 64
    hidden_dim: int = 128
    alpha: float = 0.1
    seed: int = 42
    min_pairs: int = 100
    regression_threshold: float = 0.01


@dataclass
class TrainingResult:
    status: str = "skipped"  # "accepted" / "rejected" / "skipped" / "error"
    old_accuracy: float = 0.0
    new_accuracy: float = 0.0
    production_pairs: int = 0
    arena_pairs: int = 0
    total_pairs: int = 0
    elapsed_seconds: float = 0.0
    output_path: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Step 1: Load production data
# ---------------------------------------------------------------------------

def load_production_data(db_path: str) -> List[Dict[str, Any]]:
    """Load labeled preference pairs from the routing database.

    Uses synchronous sqlite3 since this is a CLI script.

    Args:
        db_path: Path to the routing SQLite database.

    Returns:
        List of row dicts with decision_id, prompt_text, embedding (bytes),
        mf_win_rate, target, model, pair_label, outcome_status.
    """
    expanded = os.path.expanduser(db_path)
    if not Path(expanded).exists():
        logger.warning("Routing database not found at %s", expanded)
        return []

    query = """
        SELECT decision_id, prompt_text, embedding, mf_win_rate,
               target, model, pair_label, outcome_status
        FROM routing_decisions
        WHERE pair_label IS NOT NULL
          AND embedding IS NOT NULL
          AND length(embedding) > 0
        ORDER BY timestamp DESC
    """
    try:
        conn = sqlite3.connect(expanded)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        logger.info("Loaded %d labeled pairs from production database", len(rows))
        return rows
    except sqlite3.Error as exc:
        logger.error("Database error reading %s: %s", expanded, exc)
        return []


def convert_production_to_pairs(
    raw_pairs: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Convert raw database rows to train_mf_numpy-compatible format.

    Calls convert_to_training_format() then restructures into
    (embeddings_array, pairs_list) where pairs have model names and idx.

    Args:
        raw_pairs: Rows from load_production_data().

    Returns:
        (embeddings, pairs) where embeddings is (N, 3072) float32 and
        pairs is list of {model_a, model_b, winner, idx}.
    """
    formatted = convert_to_training_format(raw_pairs)

    if not formatted:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []

    embeddings_list = []
    pairs_list = []

    for i, item in enumerate(formatted):
        emb = item["embedding"]
        if emb.shape != (EMBED_DIM,):
            logger.warning(
                "Skipping pair %d: embedding shape %s != (%d,)",
                i, emb.shape, EMBED_DIM,
            )
            continue

        model_a_name = ID_TO_MODEL.get(item["model_a_id"])
        model_b_name = ID_TO_MODEL.get(item["model_b_id"])
        if model_a_name is None or model_b_name is None:
            logger.warning("Skipping pair %d: unknown model ID", i)
            continue

        idx = len(embeddings_list)
        embeddings_list.append(emb)
        pairs_list.append({
            "model_a": model_a_name,
            "model_b": model_b_name,
            "winner": item["winner"],
            "idx": idx,
        })

    if not embeddings_list:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []

    embeddings_array = np.stack(embeddings_list).astype(np.float32)
    logger.info(
        "Converted %d production pairs (embeddings shape: %s)",
        len(pairs_list), embeddings_array.shape,
    )
    return embeddings_array, pairs_list


# ---------------------------------------------------------------------------
# Step 2: Load Arena data
# ---------------------------------------------------------------------------

def load_arena_data(
    embeddings_path: str, metadata_path: str,
) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """Load cached Chatbot Arena embeddings and metadata.

    Args:
        embeddings_path: Path to gemini_embeddings.npy.
        metadata_path: Path to gemini_metadata.npz.

    Returns:
        (embeddings, pairs) or (None, None) if files are missing.
    """
    if not Path(embeddings_path).exists():
        logger.warning("Arena embeddings not found: %s", embeddings_path)
        return None, None

    if not Path(metadata_path).exists():
        logger.warning("Arena metadata not found: %s", metadata_path)
        return None, None

    try:
        embeddings = np.load(embeddings_path)
        metadata = np.load(metadata_path, allow_pickle=True)

        model_a_arr = metadata["model_a"]
        model_b_arr = metadata["model_b"]
        winner_arr = metadata["winner"]
        idx_arr = metadata["idx"]

        pairs = []
        for i in range(len(model_a_arr)):
            pairs.append({
                "model_a": str(model_a_arr[i]),
                "model_b": str(model_b_arr[i]),
                "winner": str(winner_arr[i]),
                "idx": int(idx_arr[i]),
            })

        logger.info(
            "Loaded Arena data: %d embeddings, %d pairs",
            len(embeddings), len(pairs),
        )
        return embeddings, pairs

    except Exception as exc:
        logger.error("Failed to load Arena data: %s", exc)
        return None, None


def save_arena_metadata(pairs: List[Dict[str, Any]], metadata_path: str) -> None:
    """Save Arena pairs metadata as npz for future reuse.

    Args:
        pairs: List of {model_a, model_b, winner, idx} dicts.
        metadata_path: Output .npz path.
    """
    model_a = np.array([p["model_a"] for p in pairs], dtype=object)
    model_b = np.array([p["model_b"] for p in pairs], dtype=object)
    winner = np.array([p["winner"] for p in pairs], dtype=object)
    idx = np.array([p["idx"] for p in pairs], dtype=np.int32)

    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(metadata_path, model_a=model_a, model_b=model_b, winner=winner, idx=idx)
    logger.info("Saved arena metadata to %s", metadata_path)


# ---------------------------------------------------------------------------
# Step 3: Mix datasets
# ---------------------------------------------------------------------------

def mix_datasets(
    prod_embeddings: np.ndarray,
    prod_pairs: List[Dict[str, Any]],
    arena_embeddings: Optional[np.ndarray],
    arena_pairs: Optional[List[Dict[str, Any]]],
    mix_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Mix production and arena datasets based on the target ratio.

    Production data occupies ``mix_ratio`` fraction of the combined set;
    arena data occupies ``1 - mix_ratio``.

    Args:
        prod_embeddings: (N_prod, 3072) float32.
        prod_pairs: Production pairs list.
        arena_embeddings: (N_arena, 3072) float32 or None.
        arena_pairs: Arena pairs list or None.
        mix_ratio: Fraction of combined set that is production data.
        seed: Random seed for reproducible sampling.

    Returns:
        (combined_embeddings, combined_pairs) ready for training.
    """
    has_arena = (
        arena_embeddings is not None
        and arena_pairs is not None
        and len(arena_pairs) > 0
    )
    has_prod = len(prod_pairs) > 0

    if not has_arena and not has_prod:
        logger.error("Both production and arena data are empty")
        return np.zeros((0, EMBED_DIM), dtype=np.float32), []

    if not has_arena:
        logger.warning("Arena data unavailable, using production data only")
        return prod_embeddings, prod_pairs

    if not has_prod:
        logger.warning("Production data empty, using Arena data only")
        return arena_embeddings, arena_pairs

    rng = np.random.RandomState(seed)

    num_prod = len(prod_pairs)

    if mix_ratio <= 0.0:
        num_arena_sample = len(arena_pairs)
    elif mix_ratio >= 1.0:
        return prod_embeddings, prod_pairs
    else:
        num_arena_sample = int(num_prod * (1.0 - mix_ratio) / mix_ratio)

    num_arena_sample = min(num_arena_sample, len(arena_pairs))
    logger.info(
        "Mixing: %d production + %d arena (sampled from %d) | ratio=%.2f",
        num_prod, num_arena_sample, len(arena_pairs), mix_ratio,
    )

    arena_sample_indices = rng.choice(
        len(arena_pairs), num_arena_sample, replace=False,
    )
    sampled_arena_pairs = [arena_pairs[i] for i in arena_sample_indices]

    unique_arena_emb_idx = sorted(set(p["idx"] for p in sampled_arena_pairs))
    arena_emb_remap = {old: new for new, old in enumerate(unique_arena_emb_idx)}
    sampled_arena_embs = arena_embeddings[unique_arena_emb_idx]

    combined_embeddings = np.vstack([prod_embeddings, sampled_arena_embs])

    combined_pairs = list(prod_pairs)
    prod_emb_offset = len(prod_embeddings)
    for pair in sampled_arena_pairs:
        combined_pairs.append({
            "model_a": pair["model_a"],
            "model_b": pair["model_b"],
            "winner": pair["winner"],
            "idx": prod_emb_offset + arena_emb_remap[pair["idx"]],
        })

    rng.shuffle(combined_pairs)

    logger.info(
        "Combined dataset: %d pairs, %d embeddings",
        len(combined_pairs), len(combined_embeddings),
    )
    return combined_embeddings, combined_pairs


# ---------------------------------------------------------------------------
# Step 4: Training (with warm start support)
# ---------------------------------------------------------------------------

def train_mf_incremental(
    pairs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    init_weights: Optional[Dict[str, np.ndarray]] = None,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    alpha: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Train MF Router with optional warm start from existing checkpoint.

    Same architecture as train_mf_numpy() in train_mf_gemini.py but accepts
    pre-trained weights for fine-tuning.

    Args:
        pairs: List of {model_a, model_b, winner, idx}.
        embeddings: (N, embed_dim) float32 array.
        init_weights: Dict with P.weight, text_proj.0.weight,
            classifier.0.weight for warm start. None for random init.
        hidden_dim: Model embedding dimension.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
        batch_size: Training batch size.
        alpha: Noise injection scale for embeddings.
        seed: Random seed.

    Returns:
        (best_weights, best_test_accuracy) tuple.
    """
    rng = np.random.RandomState(seed)
    embed_dim = embeddings.shape[1]

    if init_weights is not None:
        logger.info("Warm start: loading weights from checkpoint")
        P = init_weights["P.weight"].copy()
        text_proj = init_weights["text_proj.0.weight"].copy()
        classifier = init_weights["classifier.0.weight"].copy()

        if P.shape != (NUM_MODELS, hidden_dim):
            raise ValueError(
                f"P.weight shape {P.shape} != expected ({NUM_MODELS}, {hidden_dim})"
            )
        if text_proj.shape != (hidden_dim, embed_dim):
            raise ValueError(
                f"text_proj shape {text_proj.shape} != expected ({hidden_dim}, {embed_dim})"
            )
        if classifier.shape != (1, hidden_dim):
            raise ValueError(
                f"classifier shape {classifier.shape} != expected (1, {hidden_dim})"
            )
    else:
        logger.info("Cold start: random weight initialization")
        scale_p = np.sqrt(2.0 / (NUM_MODELS + hidden_dim))
        P = rng.randn(NUM_MODELS, hidden_dim).astype(np.float32) * scale_p

        scale_proj = np.sqrt(2.0 / (hidden_dim + embed_dim))
        text_proj = rng.randn(hidden_dim, embed_dim).astype(np.float32) * scale_proj

        scale_cls = np.sqrt(2.0 / (1 + hidden_dim))
        classifier = rng.randn(1, hidden_dim).astype(np.float32) * scale_cls

    adam_state = {}
    for name in ["P", "text_proj", "classifier"]:
        adam_state[name] = {"m": 0.0, "v": 0.0, "t": 0}

    def sigmoid(x):
        x = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-x))

    def l2_normalize(x, axis=-1, eps=1e-8):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)

    def adam_update(param, grad, state, lr_t, wd):
        state["t"] += 1
        t = state["t"]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grad = grad + wd * param
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
        m_hat = state["m"] / (1 - beta1 ** t)
        v_hat = state["v"] / (1 - beta2 ** t)
        param -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        return param

    data_indices = list(range(len(pairs)))

    logger.info(
        "Training MF model: %d pairs, embed_dim=%d, hidden=%d, epochs=%d, lr=%.1e",
        len(pairs), embed_dim, hidden_dim, epochs, lr,
    )

    rng.shuffle(data_indices)
    split = int(len(data_indices) * 0.95)
    train_idx = data_indices[:split]
    test_idx = data_indices[split:]

    best_test_acc = 0.0
    best_weights = {
        "P.weight": P.copy(),
        "text_proj.0.weight": text_proj.copy(),
        "classifier.0.weight": classifier.copy(),
    }

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        num_batches = math.ceil(len(train_idx) / batch_size)

        for b in range(num_batches):
            batch_start = b * batch_size
            batch_end = min(batch_start + batch_size, len(train_idx))
            batch_indices = train_idx[batch_start:batch_end]
            bs = len(batch_indices)
            if bs == 0:
                continue

            dP = np.zeros_like(P)
            d_text_proj = np.zeros_like(text_proj)
            d_classifier = np.zeros_like(classifier)
            batch_loss = 0.0

            for idx in batch_indices:
                pair = pairs[idx]
                winner_str = pair["winner"]
                model_a_id = MODEL_IDS[pair["model_a"]]
                model_b_id = MODEL_IDS[pair["model_b"]]
                prompt_idx = pair["idx"]

                if winner_str == "model_a":
                    win_id, lose_id = model_a_id, model_b_id
                else:
                    win_id, lose_id = model_b_id, model_a_id

                emb = embeddings[prompt_idx].copy()
                if alpha > 0:
                    emb += alpha * rng.randn(embed_dim).astype(np.float32)

                prompt_proj = text_proj @ emb

                p_win = l2_normalize(P[win_id])
                p_lose = l2_normalize(P[lose_id])

                logit_win = float((classifier @ (p_win * prompt_proj)).item())
                logit_lose = float((classifier @ (p_lose * prompt_proj)).item())

                diff = logit_win - logit_lose
                prob = sigmoid(diff)
                loss = -np.log(prob + 1e-8)
                batch_loss += loss

                if diff > 0:
                    epoch_correct += 1
                epoch_total += 1

                d_diff = prob - 1.0
                d_logit_win = d_diff
                d_logit_lose = -d_diff

                product_win = p_win * prompt_proj
                product_lose = p_lose * prompt_proj
                d_classifier += d_logit_win * product_win.reshape(1, -1)
                d_classifier += d_logit_lose * product_lose.reshape(1, -1)

                cls_vec = classifier.flatten()
                d_product_win = d_logit_win * cls_vec
                d_product_lose = d_logit_lose * cls_vec

                dP[win_id] += d_product_win * prompt_proj
                dP[lose_id] += d_product_lose * prompt_proj

                d_prompt_proj = d_product_win * p_win + d_product_lose * p_lose
                d_text_proj += np.outer(d_prompt_proj, emb)

            dP /= bs
            d_text_proj /= bs
            d_classifier /= bs

            P = adam_update(P, dP, adam_state["P"], lr, weight_decay)
            text_proj = adam_update(
                text_proj, d_text_proj, adam_state["text_proj"], lr, weight_decay,
            )
            classifier = adam_update(
                classifier, d_classifier, adam_state["classifier"], lr, weight_decay,
            )

            epoch_loss += batch_loss

        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_correct = 0
            if test_idx:
                for idx in test_idx:
                    pair = pairs[idx]
                    model_a_id = MODEL_IDS[pair["model_a"]]
                    model_b_id = MODEL_IDS[pair["model_b"]]
                    prompt_idx = pair["idx"]

                    if pair["winner"] == "model_a":
                        win_id, lose_id = model_a_id, model_b_id
                    else:
                        win_id, lose_id = model_b_id, model_a_id

                    emb = embeddings[prompt_idx]
                    pp = text_proj @ emb
                    p_w = l2_normalize(P[win_id])
                    p_l = l2_normalize(P[lose_id])
                    lw = float((classifier @ (p_w * pp)).item())
                    ll = float((classifier @ (p_l * pp)).item())
                    if lw > ll:
                        test_correct += 1

                test_acc = test_correct / len(test_idx)
            else:
                test_acc = 0.0

            logger.info(
                "Epoch %3d/%d | loss=%.4f | train_acc=%.3f | test_acc=%.3f",
                epoch + 1, epochs, avg_loss, train_acc, test_acc,
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_weights = {
                    "P.weight": P.copy(),
                    "text_proj.0.weight": text_proj.copy(),
                    "classifier.0.weight": classifier.copy(),
                }

    logger.info("Best test accuracy: %.3f", best_test_acc)
    return best_weights, best_test_acc


# ---------------------------------------------------------------------------
# Step 5: Evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    weights: Dict[str, np.ndarray],
    pairs: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> float:
    """Evaluate accuracy of a weight set on given pairs.

    Runs a pure forward pass with no training.

    Args:
        weights: Dict with P.weight, text_proj.0.weight, classifier.0.weight.
        pairs: List of {model_a, model_b, winner, idx}.
        embeddings: (N, embed_dim) float32 array.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if not pairs:
        return 0.0

    P = weights["P.weight"]
    text_proj = weights["text_proj.0.weight"]
    classifier_w = weights["classifier.0.weight"]

    def l2_normalize(x, eps=1e-8):
        norm = np.linalg.norm(x)
        return x / (norm + eps)

    correct = 0
    for pair in pairs:
        model_a_id = MODEL_IDS.get(pair["model_a"])
        model_b_id = MODEL_IDS.get(pair["model_b"])
        if model_a_id is None or model_b_id is None:
            continue

        prompt_idx = pair["idx"]
        if prompt_idx >= len(embeddings):
            continue

        if pair["winner"] == "model_a":
            win_id, lose_id = model_a_id, model_b_id
        else:
            win_id, lose_id = model_b_id, model_a_id

        emb = embeddings[prompt_idx]
        pp = text_proj @ emb
        p_w = l2_normalize(P[win_id])
        p_l = l2_normalize(P[lose_id])
        lw = float((classifier_w @ (p_w * pp)).item())
        ll = float((classifier_w @ (p_l * pp)).item())
        if lw > ll:
            correct += 1

    return correct / len(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Step 6: Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Load MF Router weights from a safetensors file.

    Args:
        path: Path to the .safetensors checkpoint.

    Returns:
        Dict with P.weight, text_proj.0.weight, classifier.0.weight arrays.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        KeyError: If required tensors are missing.
    """
    from safetensors.numpy import load_file

    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    tensors = load_file(path)
    required = {"P.weight", "text_proj.0.weight", "classifier.0.weight"}
    missing = required - set(tensors.keys())
    if missing:
        raise KeyError(f"Missing tensors in checkpoint: {missing}")

    logger.info(
        "Loaded checkpoint from %s: P=%s text_proj=%s classifier=%s",
        path,
        tensors["P.weight"].shape,
        tensors["text_proj.0.weight"].shape,
        tensors["classifier.0.weight"].shape,
    )
    return tensors


def save_weights(weights: Dict[str, np.ndarray], output_path: str) -> None:
    """Save weights as safetensors.

    Args:
        weights: Dict of numpy arrays to save.
        output_path: Destination file path.
    """
    from safetensors.numpy import save_file

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(weights, output_path)

    total_params = sum(w.size for w in weights.values())
    file_size = os.path.getsize(output_path)
    logger.info(
        "Saved to %s | %d params | %d bytes",
        output_path, total_params, file_size,
    )
    for name, w in weights.items():
        logger.info("  %s: shape=%s dtype=%s", name, w.shape, w.dtype)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_training_pipeline(config: TrainingConfig) -> TrainingResult:
    """Run the full incremental training pipeline.

    Executes steps 1-7 from the original script without CLI or print
    statements.  Returns a :class:`TrainingResult` describing what happened.

    Args:
        config: All paths and hyper-parameters.

    Returns:
        TrainingResult with status, accuracies, counts, and timing.
    """
    try:
        t_start = time.monotonic()

        # --- Step 1: Load and convert production data ---
        logger.info("Step 1: Loading production data from %s", config.routing_db)
        raw_prod = load_production_data(config.routing_db)

        if not raw_prod:
            logger.warning("No production data found. Nothing to train on.")
            return TrainingResult(status="skipped")

        prod_embeddings, prod_pairs = convert_production_to_pairs(raw_prod)

        if not prod_pairs:
            logger.warning("No usable production pairs after conversion.")
            return TrainingResult(status="skipped")

        # --- Step 2: Load Arena data ---
        logger.info("Step 2: Loading Arena data")
        arena_embeddings, arena_pairs = load_arena_data(
            config.arena_embeddings, config.arena_metadata,
        )

        # --- Step 3: Mix datasets ---
        logger.info("Step 3: Mixing datasets")

        use_arena = len(prod_pairs) >= config.min_pairs and arena_pairs is not None
        if len(prod_pairs) < config.min_pairs:
            logger.warning(
                "Production pairs (%d) < min_pairs (%d). "
                "Using Arena data only if available.",
                len(prod_pairs), config.min_pairs,
            )
            if arena_pairs is not None and arena_embeddings is not None:
                combined_embs = arena_embeddings
                combined_pairs = arena_pairs
            else:
                logger.warning("No Arena data either. Training on small production set.")
                combined_embs = prod_embeddings
                combined_pairs = prod_pairs
        else:
            combined_embs, combined_pairs = mix_datasets(
                prod_embeddings, prod_pairs,
                arena_embeddings, arena_pairs,
                mix_ratio=config.mix_ratio,
                seed=config.seed,
            )

        if not combined_pairs:
            logger.error("No training data available after mixing.")
            return TrainingResult(status="skipped")

        # --- Step 4: Load old checkpoint for warm start ---
        logger.info("Step 4: Loading old checkpoint")

        init_weights = None
        old_weights = None
        if Path(config.old_checkpoint).exists():
            try:
                old_weights = load_checkpoint(config.old_checkpoint)
                init_weights = old_weights
            except (KeyError, ValueError) as exc:
                logger.warning("Could not load old checkpoint: %s. Using cold start.", exc)
        else:
            logger.warning(
                "Old checkpoint not found: %s. Using cold start.",
                config.old_checkpoint,
            )

        # --- Step 5: Train ---
        logger.info("Step 5: Training")

        new_weights, new_acc = train_mf_incremental(
            combined_pairs,
            combined_embs,
            init_weights=init_weights,
            hidden_dim=config.hidden_dim,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=1e-5,
            batch_size=config.batch_size,
            alpha=config.alpha,
            seed=config.seed,
        )

        # --- Step 6: Validate against old checkpoint ---
        logger.info("Step 6: Validation")

        old_acc = 0.0
        if old_weights is not None:
            old_acc = evaluate_checkpoint(old_weights, combined_pairs, combined_embs)
            logger.info("Old checkpoint accuracy on combined data: %.3f", old_acc)
        else:
            logger.info("No old checkpoint to compare against")

        delta = new_acc - old_acc
        rejected = old_weights is not None and new_acc < old_acc - config.regression_threshold

        if rejected:
            logger.warning(
                "REJECTED: new accuracy %.3f < old accuracy %.3f - %.3f threshold. "
                "Keeping old checkpoint.",
                new_acc, old_acc, config.regression_threshold,
            )
            status = "rejected"
        else:
            save_weights(new_weights, config.output)
            status = "accepted"

        # --- Step 7: Build result ---
        elapsed = time.monotonic() - t_start
        arena_count = len(arena_pairs) if arena_pairs else 0

        return TrainingResult(
            status=status,
            old_accuracy=old_acc,
            new_accuracy=new_acc,
            production_pairs=len(prod_pairs),
            arena_pairs=arena_count,
            total_pairs=len(combined_pairs),
            elapsed_seconds=elapsed,
            output_path=config.output if status == "accepted" else "",
        )

    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        return TrainingResult(status="error", error=str(exc))
