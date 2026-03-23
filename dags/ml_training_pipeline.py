"""
DAG: перевірка даних/DVC → dvc repro prepare/train → evaluate_model (XCom) → BranchPythonOperator
за accuracy → MLflow Registry (Staging) або зупинка.

4.3: метрики з `metrics.json` потрапляють у XCom через `evaluate_model`; `check_accuracy` тягне їх
`ti.xcom_pull(task_ids='evaluate_model')` і повертає `register_model` / `stop_pipeline`.

Поріг accuracy за замовчуванням 0.85 (методичка); опційно Airflow Variable `ml_min_accuracy`.
Tracking URI MLflow — локальний `<репозиторій>/mlruns` (узгоджено з src/train.py).
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule

# У docker-compose репозиторій змонтовано сюди (не плутати з ML_PROJECT_HOST_PATH для DockerOperator).
ML_ROOT = os.environ.get("AIRFLOW_ML_PROJECT_DIR", "/opt/airflow/ml_project").rstrip("/")
EXPERIMENT_NAME = "sentiment_analysis_stages"
REGISTRY_MODEL_NAME = "SentimentClassifier"
METRICS_PATH = Path(ML_ROOT) / "metrics.json"
DATASET_PATH = Path(ML_ROOT) / "data" / "row" / "dataset.csv"


def _min_accuracy_threshold(**context) -> float:
    var = Variable.get("ml_min_accuracy", default_var="")
    if var not in (None, ""):
        return float(var)
    params = context.get("params") or {}
    if "min_accuracy" in params:
        return float(params["min_accuracy"])
    dag = context.get("dag")
    if dag and dag.params and "min_accuracy" in dag.params:
        return float(dag.params["min_accuracy"])
    return 0.85


def evaluate_model(**kwargs) -> dict:
    """Читає metrics.json після train і передає словник у XCom для BranchPythonOperator."""
    if not METRICS_PATH.is_file():
        kwargs["ti"].log.error("Файл metrics.json не знайдено: %s", METRICS_PATH)
        return {}
    with METRICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def check_accuracy(**kwargs):
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model")
    if not metrics or "accuracy" not in metrics:
        ti.log.error("Немає метрик або поля accuracy після evaluate_model")
        return "stop_pipeline"
    threshold = _min_accuracy_threshold(**kwargs)
    ti.log.info("accuracy=%.4f поріг=%.4f", metrics["accuracy"], threshold)
    if metrics["accuracy"] > threshold:
        return "register_model"
    return "stop_pipeline"


def register_model_to_staging(**context) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = f"file:{Path(ML_ROOT) / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Експеримент MLflow '{EXPERIMENT_NAME}' не знайдено")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Немає run у експерименті")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    ti = context["ti"]
    ti.log.info("register_model uri=%s name=%s", model_uri, REGISTRY_MODEL_NAME)

    mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_MODEL_NAME)
    client.transition_model_version_stage(
        name=REGISTRY_MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    ti.log.info("Model Registry: %s v%s → Staging", REGISTRY_MODEL_NAME, mv.version)


def _poke_dvc_ready() -> bool:
    if not DATASET_PATH.is_file() or DATASET_PATH.stat().st_size == 0:
        return False
    root = Path(ML_ROOT)
    if not (root / ".dvc").is_dir():
        return False
    proc = subprocess.run(
        ["dvc", "version"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="DVC prepare/train → XCom metrics → accuracy branch → MLflow Staging або stop",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "dvc", "mlflow"],
    params={"min_accuracy": 0.85},
    doc_md=__doc__,
) as dag:
    wait_raw_dataset = FileSensor(
        task_id="wait_raw_dataset",
        filepath=str(DATASET_PATH),
        poke_interval=30,
        timeout=60 * 60,
        mode="poke",
    )

    dvc_repo_ready = PythonSensor(
        task_id="dvc_repo_ready",
        python_callable=_poke_dvc_ready,
        poke_interval=30,
        timeout=60 * 30,
        mode="poke",
    )

    dvc_status_check = BashOperator(
        task_id="dvc_status_check",
        bash_command=(
            f'cd "{ML_ROOT}" && echo "=== dvc status ===" && dvc status && '
            f'echo "=== dvc data status (якщо доступно) ===" && (dvc data status 2>/dev/null || true)'
        ),
    )

    dvc_repro_prepare = BashOperator(
        task_id="dvc_repro_prepare",
        bash_command=f'cd "{ML_ROOT}" && dvc repro prepare',
    )

    dvc_repro_train = BashOperator(
        task_id="dvc_repro_train",
        bash_command=f'cd "{ML_ROOT}" && dvc repro train',
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    check_accuracy_branch = BranchPythonOperator(
        task_id="check_accuracy",
        python_callable=check_accuracy,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_to_staging,
    )

    stop_pipeline = BashOperator(
        task_id="stop_pipeline",
        bash_command=(
            f'echo "Модель не пройшла поріг accuracy; див. {METRICS_PATH}, '
            'DAG params min_accuracy або Variable ml_min_accuracy"'
        ),
    )

    pipeline_done = EmptyOperator(
        task_id="pipeline_done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    (
        wait_raw_dataset
        >> dvc_repo_ready
        >> dvc_status_check
        >> dvc_repro_prepare
        >> dvc_repro_train
        >> evaluate_model_task
        >> check_accuracy_branch
    )
    check_accuracy_branch >> register_model >> pipeline_done
    check_accuracy_branch >> stop_pipeline >> pipeline_done
