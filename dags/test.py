"""
This is the creation of a data flow using airflow
"""
# Local imports
from datetime import timedelta, datetime

# Third-party imports
import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from pathlib import Path
from tasks.data_acquisition.load_config import load_config
from tasks.data_acquisition.dataset_builder import build_dataset
from tasks.segmentation.mask_generation import generate_masks
from tasks.train.loader import prepare_datasets
from tasks.train.comb_train import train_model_t
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 16),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}
REPO_PATH = "/Users/santiagoaristizabal/Desktop/segmentation/segmentation"

dockerops_kwargs = {
    "mount_tmp_dir": False,
    "mounts": [
        # This mirrors the host folder to the same path inside the container
        Mount(
            source=f"{REPO_PATH}/data",
            target="/opt/airflow/data/",
            type="bind",
        )
    ],
    "retries": 1,
    "docker_url": "tcp://docker-socket-proxy:2375",
    "network_mode": "bridge",
}

dag = DAG(
    'fluorescence-segmentation',
    default_args=default_args,
    description=(
        "This is a simple pipeline to be bale to load a file into a "
        "database using python"
        ),
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,  # Limits concurrent DAG runs
    concurrency=1
)

load_config_task = PythonOperator(
    task_id='load_config',
    python_callable=load_config,
    dag=dag,
)
build_dataset_task = PythonOperator(
    task_id='build_dataset',
    python_callable=build_dataset,
    dag=dag,
)
generate_mask_task = PythonOperator(
    task_id="mask_generation",
    python_callable=generate_masks,
    dag=dag
)
prepare_datasets_task = PythonOperator(
    task_id="prepare_datasets",
    python_callable=prepare_datasets,
    dag=dag
)
#train_model_task = PythonOperator(
#    task_id="train_model_t",
#    python_callable=train_model_t,
#    dag=dag
#)

train_model_task = DockerOperator(
    task_id="train_model_t",
    image="model-prediction:latest",
    command="python comb_train.py --csv_path {{ ti.xcom_pull(task_ids='prepare_datasets')['csv_path'] }}",
    environment={
        "MLFLOW_TRACKING_URI": "your_token_here",
        "MLFLOW_TRACKING_USERNAME": "sam2800ml",
        "MLFLOW_TRACKING_PASSWORD": "your_token_here",
    },
    **dockerops_kwargs
)
load_config_task >> build_dataset_task >> generate_mask_task >> prepare_datasets_task >> train_model_task 