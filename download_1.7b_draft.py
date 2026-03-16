#!/usr/bin/env python3
"""
Download Qwen3-1.7B draft model for Speculative Decoding.
"""

from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    model_id = "Qwen/Qwen3-1.7B-MLX-4bit"
    local_dir = Path.home() / ".omlx" / "models" / "Qwen3-1.7B-MLX-4bit"

    print(f"📥 Downloading {model_id}...")
    print(f"📂 Target directory: {local_dir}")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print(f"\n✅ Model downloaded to: {local_dir}")

    # List downloaded files
    print("\n📋 Downloaded files:")
    for file in sorted(local_dir.glob("*")):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name:<30} {size_mb:>8.2f} MB")

if __name__ == "__main__":
    main()
