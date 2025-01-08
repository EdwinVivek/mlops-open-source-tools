from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow import AirflowException
from datetime import datetime, timedelta
import subprocess
import logging

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)

def check_data_push():
    logging.info("enter method")
    # Command to activate the Conda environment and run the script
    env_name = "base"
    script_path = "/home/edwin/git/ML-IPython-notebooks/House price prediction - project/airflow/monitor_drift.py"
    command = f'eval \"$(conda shell.bash hook)\" && conda activate {env_name} && python "{script_path}"'
    result = subprocess.run(command, capture_output=True, shell=True)
    if(result.stderr ==  ' '):
        logging.info(result.stdout.decode())
    else:
        logging.info(result.stderr)
    if "Data pushed successfully" in result.stdout.decode():
        return "Push success"
    else:
        raise ValueError(result.stdout.decode())

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=5),
}

with DAG(
    'push_data',
    default_args=default_args,
    description='A DAG to push data to feature store',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    check_drift_task = PythonOperator(
        task_id='check_drift',
        python_callable=check_data_push
    )
