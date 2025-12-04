FROM apache/airflow:2.10.5-python3.11

RUN pip install --no-cache-dir uv

WORKDIR /opt/airflow/project

RUN mkdir -p /opt/airflow/mlruns && chmod -R 777 /opt/airflow/mlruns

ENV MLFLOW_TRACKING_URI=file:/opt/airflow/mlruns

COPY pyproject.toml uv.lock ./

RUN uv pip install .

RUN pip install --no-cache-dir mlflow

COPY ends_id.txt ./



