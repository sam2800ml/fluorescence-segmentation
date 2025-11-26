FROM apache/airflow:2.10.5-python3.11

# Install uv
RUN pip install --no-cache-dir uv

# Create project directory
WORKDIR /opt/airflow/project

# Create MLflow directory with correct permissions
RUN mkdir -p /opt/airflow/mlruns && chmod -R 777 /opt/airflow/mlruns

ENV MLFLOW_TRACKING_URI=file:/opt/airflow/mlruns

# Copy dependency files
COPY pyproject.toml uv.lock ./

RUN uv pip install .

# Explicitly install MLflow (if not already in your pyproject.toml)
RUN pip install --no-cache-dir mlflow

# Copy the rest of your project files
COPY ends_id.txt ./



