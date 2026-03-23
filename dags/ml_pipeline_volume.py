"""
Пайплайн ML через монтування репозиторію в контейнер Airflow (/opt/airflow/ml_project).
Задачі виконуються в тому ж середовищі, що й scheduler (Python + залежності з requirements-airflow.txt).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

ML_ROOT = "/opt/airflow/ml_project"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline_volume_mount",
    default_args=default_args,
    description="Prepare + train з кодом на змонтованому томі",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "volume"],
) as dag:
    prepare = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"python {ML_ROOT}/src/prepare.py "
            f"{ML_ROOT}/data/row/dataset.csv "
            f"{ML_ROOT}/data/prepared"
        ),
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {ML_ROOT} && "
            "python src/train.py "
            "--train_data data/prepared/train.csv "
            "--test_data data/prepared/test.csv "
            "--c_param 1.0 --max_iter 100 --max_features 5000"
        ),
    )

    prepare >> train
