from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
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

def monitor_drift():
    logging.info("enter method")
    # Command to activate the Conda environment and run the script
    #env_name = "base"
    #script_path = "/home/edwin/git/ML-IPython-notebooks/House price prediction - project/airflow/monitor_drift.py"
    #command = f'eval \"$(conda shell.bash hook)\" && conda activate {env_name} && python "{script_path}"'
    #result = subprocess.run(command, capture_output=True, shell=True)

    python_path = "/home/edwin/anaconda3/bin/python"
    script_path = "/home/edwin/git/mlops-open-source-tools/airflow/monitor_drift.py"

    # Run the command using a list
    result = subprocess.run([python_path, script_path], capture_output=True, text=True)
    
    if(result.stdout.endswith("Data drift detected! Retraining required\n")):
        logging.info("trigger_retrain")
        logging.info(result.stdout)
        return "trigger_retrain"
    else:
        logging.info("no_retrain")
        logging.info(result.stderr)
        return "no_retrain"
        #raise ValueError(result.stdout.decode())
    
def retrain_model():
    python_path = "/home/edwin/anaconda3/bin/python"
    script_path = "/home/edwin/git/mlops-open-source-tools/airflow/train_model.py"

    # Run the command using a list
    result = subprocess.run([python_path, script_path], capture_output=True, text=True)

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
    'check_drift_and_retrain',
    default_args=default_args,
    description='DAG to check data drift and retrain if required',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    check_drift_task = BranchPythonOperator(
        task_id='check_data_drift',
        python_callable=monitor_drift
    )
    retrain_task = PythonOperator(
        task_id='trigger_retrain',
        python_callable=retrain_model
    )
    no_retrain_task = EmptyOperator(
        task_id='no_retrain'
    )


check_drift_task >> [retrain_task, no_retrain_task]
