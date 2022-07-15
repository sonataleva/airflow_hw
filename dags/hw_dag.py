import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator


from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
        dag=dag,
    )

    prediction = PythonOperator(
        task_id='prediction',
        python_callable=predict,
        dag=dag,
    )

    pipeline >> prediction

