#!/usr/bin/env python3
"""CLI entry point for MF Router incremental training.

Thin wrapper around omlx.cloud.incremental_trainer — all training logic
lives in the importable module so both this script and AutoTrainer share
the same code.

Usage:
    python scripts/train_mf_incremental.py \\
        --routing-db ~/.omlx/routing.db \\
        --arena-embeddings models/mf-router/cache/gemini_embeddings.npy \\
        --arena-metadata models/mf-router/cache/gemini_metadata.npz \\
        --old-checkpoint models/mf-router/model_gemini.safetensors \\
        --output models/mf-router/model_gemini_v2.safetensors \\
        --mix-ratio 0.3 \\
        --epochs 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Add project root to path so we can import omlx
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from omlx.cloud.incremental_trainer import (
    TrainingConfig,
    TrainingResult,
    convert_production_to_pairs,
    evaluate_checkpoint,
    load_arena_data,
    load_checkpoint,
    load_production_data,
    run_training_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental MF Router training with production data",
    )
    parser.add_argument(
        "--routing-db",
        default=os.path.expanduser("~/.omlx/routing.db"),
        help="Path to the RoutingStore SQLite database",
    )
    parser.add_argument(
        "--arena-embeddings",
        default="models/mf-router/cache/gemini_embeddings.npy",
        help="Path to cached Arena embedding .npy file",
    )
    parser.add_argument(
        "--arena-metadata",
        default="models/mf-router/cache/gemini_metadata.npz",
        help="Path to cached Arena metadata .npz file",
    )
    parser.add_argument(
        "--old-checkpoint",
        default="models/mf-router/model_gemini.safetensors",
        help="Path to existing MF Router checkpoint for warm start",
    )
    parser.add_argument(
        "--output",
        default="models/mf-router/model_gemini_v2.safetensors",
        help="Path for the new checkpoint output",
    )
    parser.add_argument("--mix-ratio", type=float, default=0.3,
                        help="Fraction of combined data from production (0.0-1.0)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Model embedding hidden dimension")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Noise injection scale for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-pairs", type=int, default=20,
                        help="Minimum production pairs required for mixed training")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show dataset stats without training")
    parser.add_argument("--regression-threshold", type=float, default=0.01,
                        help="Max accuracy drop allowed before rejecting new checkpoint")
    return parser.parse_args()


def print_dry_run(args: argparse.Namespace) -> None:
    """Load data and print stats without training."""
    raw_prod = load_production_data(args.routing_db)
    if not raw_prod:
        print("No production data found.")
        return

    prod_embeddings, prod_pairs = convert_production_to_pairs(raw_prod)
    arena_embeddings, arena_pairs = load_arena_data(
        args.arena_embeddings, args.arena_metadata,
    )
    arena_count = len(arena_pairs) if arena_pairs else 0

    old_weights = None
    if Path(args.old_checkpoint).exists():
        try:
            old_weights = load_checkpoint(args.old_checkpoint)
        except (KeyError, ValueError) as exc:
            logger.warning("Could not load old checkpoint: %s", exc)

    print()
    print("=" * 55)
    print("  Incremental Training - DRY RUN")
    print("=" * 55)
    print(f"  Production pairs:    {len(prod_pairs)}")
    print(f"  Arena pairs:         {arena_count}")
    print(f"  Mix ratio:           {args.mix_ratio:.2f}")
    print(f"  Warm start:          {'yes' if old_weights else 'no'}")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Learning rate:       {args.lr:.1e}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Output:              {args.output}")
    print("-" * 55)

    label_counts: Dict[str, int] = {}
    for row in raw_prod:
        lbl = row.get("pair_label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print("  Production label distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {lbl:30s} {cnt:6d}")

    if old_weights is not None and prod_pairs:
        old_acc = evaluate_checkpoint(old_weights, prod_pairs, prod_embeddings)
        print(f"  Old checkpoint accuracy:  {old_acc:.3f}")

    print("=" * 55)


def print_result(result: TrainingResult) -> None:
    """Print human-readable training summary."""
    print()
    print("=" * 55)
    print("  Incremental Training Summary")
    print("=" * 55)
    print(f"  Production pairs:    {result.production_pairs}")
    print(f"  Arena pairs:         {result.arena_pairs}")
    print(f"  Total training:      {result.total_pairs}")
    print("-" * 55)
    print(f"  Old checkpoint acc:  {result.old_accuracy:.3f}")
    print(f"  New checkpoint acc:  {result.new_accuracy:.3f}")
    delta = result.new_accuracy - result.old_accuracy
    print(f"  Delta:               {delta:+.3f}")
    print(f"  Status:              {result.status.upper()}")
    print("-" * 55)
    if result.status == "accepted":
        print(f"  Output: {result.output_path}")
    else:
        print(f"  Output: (not saved, old checkpoint retained)")
    print(f"  Elapsed: {result.elapsed_seconds:.1f}s")
    print("=" * 55)


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print_dry_run(args)
        return

    config = TrainingConfig(
        routing_db=args.routing_db,
        arena_embeddings=args.arena_embeddings,
        arena_metadata=args.arena_metadata,
        old_checkpoint=args.old_checkpoint,
        output=args.output,
        mix_ratio=args.mix_ratio,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        seed=args.seed,
        min_pairs=args.min_pairs,
        regression_threshold=args.regression_threshold,
    )

    result = run_training_pipeline(config)
    print_result(result)

    if result.status == "error":
        logger.error("Training failed: %s", result.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
