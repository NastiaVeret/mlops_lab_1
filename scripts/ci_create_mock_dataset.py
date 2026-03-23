#!/usr/bin/env python3
"""Генерує синтетичний CSV для CML, якщо немає data/row/dataset.csv (~80 рядків, баланс класів)."""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path("data/row/dataset.csv")

POS = [
    "good movie",
    "great acting",
    "I loved it",
    "best film",
    "excellent story",
    "highly recommend",
    "not bad at all",
    "enjoyed every minute",
    "fantastic",
    "amazing cast",
]
NEG = [
    "bad movie",
    "terrible plot",
    "waste of time",
    "boring",
    "awful acting",
    "worst film",
    "poor script",
    "disappointing",
    "could not finish",
    "hated it",
]


def main() -> None:
    rows: list[tuple[str, str]] = [("review", "sentiment")]
    for i in range(40):
        rows.append((f"{POS[i % len(POS)]} review {i}", "positive"))
        rows.append((f"{NEG[i % len(NEG)]} review {i}", "negative"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {OUT} ({len(rows) - 1} rows)")


if __name__ == "__main__":
    main()
