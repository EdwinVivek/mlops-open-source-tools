import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import sqlalchemy as db
from House_price_prediction import *


class MonitorModel():
    def __init__(self):
        self.house = HousePricePrediction()
        self.features = None
    
    def push_feedback_to_db(self):
        start_date = datetime.now() - timedelta(days=2)

        self.features = self.house.get_historical_features()
        last_id = self.features.loc[self.features["house_id"].idxmax()]["house_id"]
        path = os.getcwd() + "//serving//feedback.csv"
        #path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)), "serving/feedback.csv")
        data = pd.read_csv(path, parse_dates=["event_timestamp"])
        df = data[data["event_timestamp"] > start_date]
        df["house_id"] = range(last_id+1, last_id + len(df)+1)

        print(df.head())
        df_X = df.drop("prediction", axis=1)
        df_y = df[["event_timestamp","house_id", "prediction"]]
        self.house.save_df_to_postgres(df_X, df_y, 'append')
        end_date = df.loc[df["event_timestamp"].idxmax()]["event_timestamp"]
        self.house.materialize(start_date=start_date, end_date = end_date)
        print("Feedback data pushed to online feature store successfully!")
        
    
    def get_reference_and_current_data(self, start_date, end_date):
       feature_list=[
            "house_features:area",
            "house_features:bedrooms",
            "house_features:mainroad"
        ]
       entity_df_ref = pd.DataFrame(self.features["house_id"]).to_dict(orient="records")
       reference =  self.house.get_online_features(entity_df_ref, feature_list)
       connstr = 'postgresql+psycopg://postgres:Syncfusion%40123@localhost:5432/feast_offline'
       engine = db.create_engine(connstr)
       entity_df_cur = pd.read_sql(str.format("select house_id from public.house_features_sql where event_timestamp >= '{0}'", start_date.strftime(r'%Y-%m-%d %H:%M:%S')), con=engine)
       current = self.house.get_online_features(entity_df_cur, feature_list)
       return (reference, current)

    def monitor_drift(self, reference, current):
        print(reference)
        print(current)


if __name__ == "__main__":
    #m = MonitorModel()
    print("main called")
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            print(f"Argument {i}: {arg}")
    else:
        print("No arguments passed.")
