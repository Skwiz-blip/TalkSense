"""
train_entry.py
Small CLI entrypoint to run trainer.train_continuous().

This mirrors how a backend service would spawn the Python training worker:
the last stdout line is JSON metrics.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from trainer import train_continuous


def _default_run_id() -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{ts}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--org-id", default="demo")
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--pretrained", default="bert-base-multilingual-cased")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    train_continuous(
        run_id=args.run_id,
        org_id=args.org_id,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        pretrained=args.pretrained,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

