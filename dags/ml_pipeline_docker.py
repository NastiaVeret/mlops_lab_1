"""
Пайплайн ML через DockerOperator: кожна задача в окремому контейнері образу mlops-lab:latest.

Потрібно:
  1. Зібрати образ: docker compose --profile ml-image build ml-image
  2. У .env задати ML_PROJECT_HOST_PATH — абсолютний шлях до цього репозиторію на машині з Docker
     (інакше bind-mount у дочірній контейнер не побачить ваші data/ та src/).
"""
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

ML_PROJECT_HOST_PATH = os.environ.get("ML_PROJECT_HOST_PATH", "").strip()

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

docker_mounts = (
    [Mount(source=ML_PROJECT_HOST_PATH, target="/app", type="bind")]
    if ML_PROJECT_HOST_PATH
    else []
)

with DAG(
    dag_id="ml_pipeline_docker_operator",
    default_args=default_args,
    description="Prepare + train у контейнері mlops-lab (DockerOperator)",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "docker"],
) as dag:
    guard = BashOperator(
        task_id="require_host_path_for_docker_bind",
        bash_command=(
            'if [ -z "$ML_PROJECT_HOST_PATH" ]; then '
            'echo "Встановіть ML_PROJECT_HOST_PATH у .env (абсолютний шлях до репозиторію)"; exit 1; '
            "fi"
        ),
        env={"ML_PROJECT_HOST_PATH": ML_PROJECT_HOST_PATH},
    )

    prepare_docker = DockerOperator(
        task_id="prepare_in_ml_container",
        image="mlops-lab:latest",
        api_version="auto",
        auto_remove="force",
        mount_tmp_dir=False,
        docker_url="unix://var/run/docker.sock",
        working_dir="/app",
        command=(
            "python src/prepare.py "
            "/app/data/row/dataset.csv /app/data/prepared"
        ),
        mounts=docker_mounts,
    )

    train_docker = DockerOperator(
        task_id="train_in_ml_container",
        image="mlops-lab:latest",
        api_version="auto",
        auto_remove="force",
        mount_tmp_dir=False,
        docker_url="unix://var/run/docker.sock",
        working_dir="/app",
        command=(
            "python src/train.py "
            "--train_data /app/data/prepared/train.csv "
            "--test_data /app/data/prepared/test.csv "
            "--c_param 1.0 --max_iter 100 --max_features 5000"
        ),
        mounts=docker_mounts,
    )

    guard >> prepare_docker >> train_docker
