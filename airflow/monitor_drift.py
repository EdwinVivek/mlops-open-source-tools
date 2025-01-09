import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from House_price_prediction import *
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import logging
from monitoring.evidently_monitoring import *

WORKSPACE = 'monitoring workspace'
PROJECT = 'monitoring project'

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)

class MonitorDrift():
    def __init__(self):       
        self.house = HousePricePrediction()      
        self.monitoring = Monitoring(DataDriftReport())
        self.start_date = datetime.now() - timedelta(days=20)

    def get_db_connection(self):
        connstr = 'postgresql+psycopg2://postgres:root@localhost:5432/feast_offline'
        engine = create_engine(connstr)
        return engine
            
    def get_reference_and_current_data(self):
        feature_list=[
            "house_features:area",
            "house_features:bedrooms",
            "house_features:mainroad"
        ]
        store = self.house.get_feature_store()
        features = self.house.get_historical_features()
        entity_df_ref = pd.DataFrame(features["house_id"])
        reference = self.house.get_online_features(store, entity_df_ref)
        engine = self.get_db_connection()
        entity_df_cur = pd.read_sql(str.format("select house_id from public.house_features_sql where event_timestamp >= '{0}'", self.start_date.strftime(r'%Y-%m-%d %H:%M:%S')), con=engine)
        current = self.house.get_online_features(store, entity_df_cur)
        return reference, current
     
    def monitor_drift(self, reference=None, current=None):
        if(reference is None or current is None):
            reference, current = self.get_reference_and_current_data()
        logging.info("reference:%s", reference)
        logging.info("reference:%s", current)
        ws = self.monitoring.create_workspace(WORKSPACE)
        project = self.monitoring.search_or_create_project(PROJECT, ws)
        #Data drift report
        print(self.monitoring.current_strategy)
        drift = self.monitoring.execute_strategy(reference, current, ws)
        #Data drfit test report
        self.monitoring.set_strategy = DataDriftTestReport()
        test_suite = self.monitoring.execute_strategy(reference, current, ws)
        # Check if drift is detected
        drift_detected = any(test["status"] == "FAIL" for test in test_suite.as_dict()["tests"])
        return drift_detected

if __name__ == "__main__":
    os.chdir("/home/edwin/git/mlops-open-source-tools/")
    drift_monitor = MonitorDrift()
    refr, curr = drift_monitor.get_reference_and_current_data()
    drift = drift_monitor.monitor_drift(refr, curr)
    if(drift):
        logging.info("Data drift detected! Retraining required.")
    else:
        logging.info("No drift detected!")
