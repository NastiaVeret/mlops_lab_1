#!/usr/bin/env python3
"""
Перевірка DAG: DagBag без import_errors (синтаксис + імпорти).
Запускати з кореня репозиторію.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    airflow_home = root / ".airflow_ci"
    airflow_home.mkdir(exist_ok=True)
    os.environ["AIRFLOW_HOME"] = str(airflow_home)
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    os.environ.setdefault("AIRFLOW_ML_PROJECT_DIR", "/opt/airflow/ml_project")

    dag_folder = str(root / "dags")
    # Імпорт після налаштування змінних середовища Airflow
    # pylint: disable-next=import-outside-toplevel
    from airflow.models import DagBag  # noqa: E402

    bag = DagBag(dag_folder=dag_folder, include_examples=False)
    if bag.import_errors:
        for path, err in sorted(bag.import_errors.items()):
            print(f"IMPORT ERROR {path}:\n{err}", file=sys.stderr)
        return 1

    print(f"OK: завантажено {len(bag.dags)} DAG(s): {sorted(bag.dags.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
