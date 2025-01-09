import sys
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import sqlalchemy as db
import pickle
sys.path.append(os.getcwd())

from House_price_prediction import *
from model.house_model import HouseModel


class TrainModel():
    def __init__(self):
        self.house_main = HousePricePrediction()
        self.house_model = HouseModel()

    def get_current_features(self):
        connstr = 'postgresql+psycopg://postgres:Syncfusion%40123@localhost:5432/feast_offline'
        engine = db.create_engine(connstr)
        Y_hist = pd.read_sql(str.format("select house_id, price from public.house_target_sql"), con=engine)
        store = self.house_main.get_feature_store()
        X_hist = self.house_main.get_online_features(store, pd.DataFrame(Y_hist["house_id"]))
        X_hist["price"] = Y_hist["price"]
        return X_hist

    def predict_new_data(self):
        path = os.getcwd() + "//serving//feedback.csv"
        #path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)), "serving/feedback.csv")
        X_new = pd.read_csv(path)
        X_new.drop(["event_timestamp", "prediction"], axis=1, inplace=True)
        lr_model = self.house_model.load_model()
        X_new = X_new[lr_model.feature_names_in_]
        Y_new = self.house_model.predict(X_new)
        X_new["proxy_target"] = Y_new
        return X_new  

    def create_and_train_new_dataset_with_target(self, X_hist, X_new):  
        new_data_combined = X_new.copy()
        new_data_combined["price"] = new_data_combined["proxy_target"]      
        historical_data_combined = X_hist.copy()
        combined_data = pd.concat([historical_data_combined, new_data_combined], ignore_index=True)
        X_combined = combined_data.drop(columns=["price", "proxy_target", "house_id"])
        y_combined = combined_data["price"]
        print(combined_data)
        self.train_model(X_combined, y_combined)

    def train_model(self, x, y):
        self.params = {
            "fit_intercept": True,
            "positive": False
        }
        model = LinearRegression(**self.params)
        model.fit(x, y)
        with open("model\\house_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model re-trained and saved as model.pkl")



if __name__ == "__main__":
    t = TrainModel()
    X_hist = t.get_current_features()
    X_new = t.predict_new_data()
    t.create_and_train_new_dataset_with_target(X_hist, X_new)

