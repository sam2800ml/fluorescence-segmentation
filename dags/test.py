"""
This is the creation of a data flow using airflow
"""
# Local imports
from datetime import timedelta, datetime

# Third-party imports
import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator

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
train_model_task = PythonOperator(
    task_id="train_model_t",
    python_callable=train_model_t,
    dag=dag
)
load_config_task >> build_dataset_task >> generate_mask_task >> prepare_datasets_task >> train_model_task 