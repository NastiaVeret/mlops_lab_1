"""Спільне налаштування CI/локального pytest: Airflow очікує AIRFLOW_HOME до першого імпорту."""

from __future__ import annotations

import os
from pathlib import Path


def pytest_configure(config) -> None:
    root = Path(__file__).resolve().parent.parent
    airflow_home = root / ".airflow_ci"
    airflow_home.mkdir(exist_ok=True)
    os.environ["AIRFLOW_HOME"] = str(airflow_home)
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    os.environ.setdefault("AIRFLOW_ML_PROJECT_DIR", str(root))
