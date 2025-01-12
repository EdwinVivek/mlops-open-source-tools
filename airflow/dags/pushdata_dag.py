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

def check_data_push():
    logging.info("enter method")
    # Command to activate the Conda environment and run the script
    #env_name = "base"
    #script_path = "/home/edwin/git/ML-IPython-notebooks/House price prediction - project/airflow/monitor_drift.py"
    #command = f'eval \"$(conda shell.bash hook)\" && conda activate {env_name} && python "{script_path}"'
    #result = subprocess.run(command, capture_output=True, shell=True)

    python_path = "/home/edwin/anaconda3/bin/python"
    script_path = "/home/edwin/git/mlops-open-source-tools/airflow/update_datastore.py"
    result = subprocess.run([python_path, script_path], capture_output=True, text=True)
    
    if(result.stdout.endswith("Data pushed successfully\n")):
        logging.info(result.stderr)
        return "update_dashboard"
    else:
        logging.info(result.stderr)
        return "no_update"
        #raise ValueError(result.stdout.decode())

def update_live_dashboard():
    python_path = "/home/edwin/anaconda3/bin/python"
    script_path = "/home/edwin/git/mlops-open-source-tools/airflow/live_dashboard.py"
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
    'push_features',
    default_args=default_args,
    description='A DAG to push data to feature store',
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    check_drift_task = BranchPythonOperator(
        task_id='push_feedbac_data',
        python_callable=check_data_push
    )
    update_dashboard_task = PythonOperator(
        task_id="update_dashboard",
        python_callable=update_live_dashboard
    )
    no_update_task = EmptyOperator(
        task_id='no_update'
    )

check_drift_task >> [update_dashboard_task, no_update_task]