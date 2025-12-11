from __future__ import annotations

from pathlib import Path
import sys

# Ensure we can import src.* when running as `python scripts/build_indexes.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vectorstores import (  # type: ignore  # noqa: E402
    build_sig_examples_store,
    build_med_knowledge_store,
)


def main() -> None:
    print("Building sig examples vector store...")
    build_sig_examples_store(rebuild=True)
    print("Done building sig examples store.")

    print("Building medical knowledge vector store...")
    build_med_knowledge_store(rebuild=True)
    print("Done building medical knowledge store.")


if __name__ == "__main__":  # pragma: no cover
    main()
