#!/usr/bin/env python3
"""
Тренування моделі та експорт артефактів для CI / Quality Gate:
  - model.pkl
  - metrics.json (поля accuracy, f1 та детальні метрики)
  - confusion_matrix.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_data", required=True, help="Шлях до train.csv")
    p.add_argument("--test_data", required=True, help="Шлях до test.csv")
    p.add_argument("--c_param", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--max_features", type=int, default=5000)
    p.add_argument("--author", type=str, default="CI")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
