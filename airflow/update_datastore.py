import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from House_price_prediction import *
from datetime import datetime, timedelta
from sqlalchemy import create_engine
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

class UpdateDataStore():
    def __init__(self):       
        self.house = HousePricePrediction()      
        #self.features = None
        self.start_date = datetime.now() - timedelta(days=20)
        self.end_date = datetime.now()

    def get_db_connection(self):
        connstr = 'postgresql+psycopg2://postgres:root@localhost:5432/feast_offline'
        engine = create_engine(connstr)
        return engine
              
    def push_feedback_to_db(self):
        try:
            engine = self.get_db_connection()
            logging.info(engine)
            house_feature = pd.read_sql("select house_id from public.house_features_sql", con=engine)
            #self.features = self.house.get_historical_features(entity_df=entity_df)
            last_id = house_feature.loc[house_feature["house_id"].idxmax()]["house_id"]
            logging.info(last_id)
            path = os.getcwd() + "/serving/feedback.csv"
            logging.info(path)
            #path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)), "serving/feedback.csv")
            data = pd.read_csv(path, parse_dates=["event_timestamp"])
            logging.info(data.head())
            df = data[data["event_timestamp"] > self.start_date ]
            df["house_id"] = range(last_id+1, last_id + len(df)+1)
            logging.info("new records: %s", df.shape)
            df_X = df.drop("prediction", axis=1)
            df_y = df[["event_timestamp","house_id"]]
            self.house.save_df_to_postgres(df_X, df_y, 'append')
            end_date = df.loc[df["event_timestamp"].idxmax()]["event_timestamp"]
            self.house.materialize(start_date=self.start_date , end_date = self.end_date)
            logging.info("Feedback data pushed to online feature store successfully!") 
        except Exception as e:   
            logging.error(e)
        
     
    
if __name__ == "__main__":
    os.chdir("/home/edwin/git/ML-IPython-notebooks/House price prediction - project/")
    uds = UpdateDataStore()
    uds.push_feedback_to_db()
    logging.info("Data pushed successfully")
    print("Data pushed successfully")
