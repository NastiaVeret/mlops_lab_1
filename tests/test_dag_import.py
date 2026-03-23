"""4.2: перевірка DAG на помилки імпорту (DagBag) — зручно ганяти в CI."""

from pathlib import Path

from airflow.models import DagBag

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dag_import() -> None:
    dag_folder = str(REPO_ROOT / "dags")
    dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"DAG import errors:\n{dag_bag.import_errors}"
