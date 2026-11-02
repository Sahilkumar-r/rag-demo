"""Chunk the docs and compute embeddings: python scripts/build_index.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragdemo.retrieval import build_index  # noqa: E402

if __name__ == "__main__":
    n = build_index(ROOT / "data" / "docs", ROOT / "data" / "index")
    print(f"indexed {n} chunks")
