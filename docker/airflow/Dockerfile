FROM apache/airflow:2.10.4-python3.11

USER airflow
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
# PyPI downloads of large wheels (e.g. scipy) often hit the default ~15s read timeout on slow links.
ENV PIP_DEFAULT_TIMEOUT=300
RUN pip install --no-cache-dir --retries 10 -r /tmp/requirements-airflow.txt
