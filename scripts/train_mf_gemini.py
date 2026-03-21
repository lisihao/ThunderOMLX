#!/usr/bin/env python3
"""Retrain RouteLLM MF Router with Gemini embeddings.

Replaces OpenAI text-embedding-3-small (1536-dim) with
Google gemini-embedding-001 (3072-dim) so we can use the existing
Gemini API key for inference without needing OpenAI.

Steps:
  1. Load Chatbot Arena preference data from HuggingFace
  2. Extract first-turn prompts
  3. Compute Gemini embeddings (3072-dim) in batches
  4. Train MF model (P + text_proj + classifier)
  5. Save weights as safetensors

Usage:
  python scripts/train_mf_gemini.py --gemini-key AIzaSy...
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# RouteLLM MODEL_IDS (64 arena models)
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
NUM_MODELS = 64


# ---------------------------------------------------------------------------
# Step 1: Load and prepare battle data
# ---------------------------------------------------------------------------

def load_battles():
    """Load Chatbot Arena preference data from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading lmsys/lmsys-arena-human-preference-55k ...")
    ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train")
    logger.info("Loaded %d rows", len(ds))
    return ds


def extract_prompts_and_pairs(ds):
    """Extract first-turn prompts and pairwise data.

    Returns:
        prompts: list of str (one per row, even ties)
        pairs: list of dict {model_a, model_b, winner, idx}
    """
    prompts = []
    pairs = []
    skipped_ties = 0
    skipped_unknown = 0

    for i, row in enumerate(ds):
        # Extract first turn from JSON-encoded prompt
        try:
            prompt_turns = json.loads(row["prompt"])
            if isinstance(prompt_turns, list) and len(prompt_turns) > 0:
                first_turn = prompt_turns[0]
            else:
                first_turn = str(prompt_turns)
        except (json.JSONDecodeError, TypeError):
            first_turn = str(row["prompt"])

        # Truncate very long prompts (embedding API limit)
        if len(first_turn) > 8000:
            first_turn = first_turn[:8000]

        prompts.append(first_turn)

        # Skip ties
        if row["winner_tie"] == 1:
            skipped_ties += 1
            continue

        # Check model IDs exist
        model_a = row["model_a"]
        model_b = row["model_b"]
        if model_a not in MODEL_IDS or model_b not in MODEL_IDS:
            skipped_unknown += 1
            continue

        # Skip same-model battles
        if model_a == model_b:
            continue

        winner = "model_a" if row["winner_model_a"] == 1 else "model_b"
        pairs.append({
            "model_a": model_a,
            "model_b": model_b,
            "winner": winner,
            "idx": i,
        })

    logger.info(
        "Extracted %d prompts, %d pairs (skipped %d ties, %d unknown models)",
        len(prompts), len(pairs), skipped_ties, skipped_unknown,
    )
    return prompts, pairs


# ---------------------------------------------------------------------------
# Step 2: Compute Gemini embeddings
# ---------------------------------------------------------------------------

def compute_gemini_embeddings(prompts, api_key, batch_size=50, cache_path=None):
    """Compute gemini-embedding-001 embeddings for all prompts.

    Uses batched API calls for efficiency. Caches to disk to resume
    on interruption.

    Args:
        prompts: list of str
        api_key: Gemini API key
        batch_size: prompts per API call (max ~100)
        cache_path: path to save/load partial results

    Returns:
        np.ndarray of shape (len(prompts), 3072)
    """
    import urllib.request

    embedding_dim = 3072
    total = len(prompts)

    # Try to load from cache (supports partial resume)
    start_batch = 0
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape == (total, embedding_dim):
            # Check how many are non-zero (already computed)
            nonzero_mask = np.any(cached != 0, axis=1)
            nonzero_count = int(np.sum(nonzero_mask))
            if nonzero_count == total:
                logger.info("Loaded complete cached embeddings from %s", cache_path)
                return cached
            logger.info("Resuming from cache: %d/%d already computed", nonzero_count, total)
            embeddings = cached
            # Find first batch that has zeros
            start_batch = nonzero_count // batch_size
        else:
            logger.info("Cache shape mismatch (%s != (%d,%d)), starting fresh",
                        cached.shape, total, embedding_dim)
            embeddings = np.zeros((total, embedding_dim), dtype=np.float32)
    else:
        embeddings = np.zeros((total, embedding_dim), dtype=np.float32)

    processed = start_batch * batch_size
    errors = 0
    t0 = time.monotonic()

    num_batches = math.ceil(total / batch_size)
    logger.info(
        "Computing Gemini embeddings: %d prompts, %d batches (starting from batch %d)",
        total, num_batches, start_batch,
    )

    for batch_idx in range(start_batch, num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_texts = prompts[start:end]

        # Build batch request
        requests_list = []
        for text in batch_texts:
            requests_list.append({
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text or " "}]},
            })

        payload = json.dumps({"requests": requests_list}).encode()

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/gemini-embedding-001:batchEmbedContents?key={api_key}"
        )

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        retries = 5
        for attempt in range(retries):
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                result = json.loads(resp.read())

                batch_embeddings = result.get("embeddings", [])
                for j, emb_data in enumerate(batch_embeddings):
                    values = emb_data.get("values", [])
                    if len(values) == embedding_dim:
                        embeddings[start + j] = np.array(values, dtype=np.float32)
                    else:
                        logger.warning("Unexpected dim %d at index %d", len(values), start + j)

                processed += len(batch_texts)
                break

            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str:
                    # Rate limited - exponential backoff with longer waits
                    wait = min(60, 5 * (2 ** attempt))
                    logger.warning("Rate limited (429), waiting %ds (attempt %d/%d)",
                                   wait, attempt + 1, retries)
                    time.sleep(wait)
                elif attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Batch %d attempt %d failed: %s, retrying in %ds",
                                   batch_idx, attempt, exc, wait)
                    time.sleep(wait)
                else:
                    logger.error("Batch %d failed after %d attempts: %s", batch_idx, retries, exc)
                    errors += len(batch_texts)

        # Progress logging every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.monotonic() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.0f prompts/s | errors: %d",
                processed, total, processed / total * 100, rate, errors,
            )

        # Incremental save every 50 batches for resume support
        if cache_path and (batch_idx + 1) % 50 == 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info("Checkpoint saved to %s", cache_path)

        # Rate limiting: Gemini free tier is 1500 RPM
        # batch_size=50 * 30 batches/min = 1500 items/min
        # ~2 seconds between batches
        if batch_idx < num_batches - 1:
            time.sleep(2.0)

    elapsed = time.monotonic() - t0
    logger.info(
        "Embedding complete: %d/%d in %.1fs (%.0f/s), %d errors",
        processed, total, elapsed, processed / elapsed if elapsed > 0 else 0, errors,
    )

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Saved embedding cache to %s", cache_path)

    return embeddings


# ---------------------------------------------------------------------------
# Step 3: Train MF model (numpy-only, no PyTorch required)
# ---------------------------------------------------------------------------

def train_mf_numpy(
    pairs,
    embeddings,
    hidden_dim=128,
    epochs=100,
    lr=3e-4,
    weight_decay=1e-5,
    batch_size=64,
    alpha=0.1,
    seed=42,
):
    """Train MF Router using numpy (Adam optimizer).

    Implements the same architecture as RouteLLM MFModel_Train
    but without PyTorch dependency.

    Architecture:
        score(model, prompt) = classifier @ (L2norm(P[model]) * text_proj @ embedding)
        loss = BCE(score(winner) - score(loser), 1)

    Note: embed_dim is 3072 for gemini-embedding-001.

    Args:
        pairs: list of {model_a, model_b, winner, idx}
        embeddings: (N, embed_dim) array
        hidden_dim: model embedding dimension (128)
        epochs: training epochs
        lr: learning rate
        weight_decay: L2 regularization
        batch_size: training batch size
        alpha: noise injection scale
        seed: random seed

    Returns:
        dict of trained weights {name: ndarray}
    """
    rng = np.random.RandomState(seed)
    embed_dim = embeddings.shape[1]  # 768 for Gemini

    # Initialize weights (Xavier/Glorot)
    scale_p = np.sqrt(2.0 / (NUM_MODELS + hidden_dim))
    P = rng.randn(NUM_MODELS, hidden_dim).astype(np.float32) * scale_p

    scale_proj = np.sqrt(2.0 / (hidden_dim + embed_dim))
    text_proj = rng.randn(hidden_dim, embed_dim).astype(np.float32) * scale_proj

    scale_cls = np.sqrt(2.0 / (1 + hidden_dim))
    classifier = rng.randn(1, hidden_dim).astype(np.float32) * scale_cls

    # Adam optimizer state
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
        """Apply Adam update with weight decay."""
        state["t"] += 1
        t = state["t"]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Weight decay
        grad = grad + wd * param

        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)

        m_hat = state["m"] / (1 - beta1 ** t)
        v_hat = state["v"] / (1 - beta2 ** t)

        param -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        return param

    # Build index arrays for batch sampling
    data_indices = list(range(len(pairs)))

    logger.info(
        "Training MF model: %d pairs, embed_dim=%d, hidden=%d, epochs=%d",
        len(pairs), embed_dim, hidden_dim, epochs,
    )

    # Split train/test
    rng.shuffle(data_indices)
    split = int(len(data_indices) * 0.95)
    train_idx = data_indices[:split]
    test_idx = data_indices[split:]

    best_test_acc = 0.0
    best_weights = None

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

            # Zero gradients
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

                # Get prompt embedding + noise
                emb = embeddings[prompt_idx].copy()
                if alpha > 0:
                    emb += alpha * rng.randn(embed_dim).astype(np.float32)

                # Forward pass
                # prompt_proj = text_proj @ emb  (hidden_dim,)
                prompt_proj = text_proj @ emb

                # Model embeddings (L2 normalized)
                p_win = l2_normalize(P[win_id])    # (hidden_dim,)
                p_lose = l2_normalize(P[lose_id])  # (hidden_dim,)

                # Logits
                logit_win = float((classifier @ (p_win * prompt_proj)).item())
                logit_lose = float((classifier @ (p_lose * prompt_proj)).item())

                # BCE loss: -log(sigmoid(logit_win - logit_lose))
                diff = logit_win - logit_lose
                prob = sigmoid(diff)
                loss = -np.log(prob + 1e-8)
                batch_loss += loss

                # Accuracy
                if diff > 0:
                    epoch_correct += 1
                epoch_total += 1

                # Backward pass (manual gradients)
                # d_loss/d_diff = sigmoid(diff) - 1 = prob - 1
                d_diff = prob - 1.0  # negative because we want to increase diff

                # d_diff/d_logit_win = 1, d_diff/d_logit_lose = -1
                d_logit_win = d_diff
                d_logit_lose = -d_diff

                # d_logit/d_classifier = (p * prompt_proj).T
                product_win = p_win * prompt_proj
                product_lose = p_lose * prompt_proj
                d_classifier += d_logit_win * product_win.reshape(1, -1)
                d_classifier += d_logit_lose * product_lose.reshape(1, -1)

                # d_logit/d_(p*prompt_proj) = classifier.T
                cls_vec = classifier.flatten()  # (hidden_dim,)

                # d_logit_win/d_p_win = cls_vec * prompt_proj (element-wise)
                # d_logit_win/d_prompt_proj = cls_vec * p_win
                d_product_win = d_logit_win * cls_vec
                d_product_lose = d_logit_lose * cls_vec

                # Gradient for P (ignoring L2 norm gradient for simplicity)
                dP[win_id] += d_product_win * prompt_proj
                dP[lose_id] += d_product_lose * prompt_proj

                # Gradient for text_proj
                # prompt_proj = text_proj @ emb
                # d_text_proj += d_prompt_proj @ emb.T
                d_prompt_proj = d_product_win * p_win + d_product_lose * p_lose
                d_text_proj += np.outer(d_prompt_proj, emb)

            # Average gradients
            dP /= bs
            d_text_proj /= bs
            d_classifier /= bs

            # Adam updates
            P = adam_update(P, dP, adam_state["P"], lr, weight_decay)
            text_proj = adam_update(text_proj, d_text_proj, adam_state["text_proj"], lr, weight_decay)
            classifier = adam_update(classifier, d_classifier, adam_state["classifier"], lr, weight_decay)

            epoch_loss += batch_loss

        # Epoch stats
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0

        # Test accuracy
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_correct = 0
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

            test_acc = test_correct / len(test_idx) if test_idx else 0

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
    return best_weights or {
        "P.weight": P,
        "text_proj.0.weight": text_proj,
        "classifier.0.weight": classifier,
    }


# ---------------------------------------------------------------------------
# Step 4: Save weights
# ---------------------------------------------------------------------------

def save_weights(weights, output_path):
    """Save weights as safetensors."""
    from safetensors.numpy import save_file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MF Router with Gemini embeddings")
    parser.add_argument("--gemini-key", required=True, help="Gemini API key")
    parser.add_argument("--output", default="models/mf-router/model_gemini.safetensors")
    parser.add_argument("--cache-dir", default="models/mf-router/cache")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.monotonic()

    # Step 1: Load battle data
    ds = load_battles()
    prompts, pairs = extract_prompts_and_pairs(ds)

    # Step 2: Compute Gemini embeddings
    cache_path = os.path.join(args.cache_dir, "gemini_embeddings.npy")
    embeddings = compute_gemini_embeddings(
        prompts, args.gemini_key,
        batch_size=50,
        cache_path=cache_path,
    )

    # Step 3: Train
    weights = train_mf_numpy(
        pairs, embeddings,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=args.seed,
    )

    # Step 4: Save
    save_weights(weights, args.output)

    elapsed = time.monotonic() - t_start
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
