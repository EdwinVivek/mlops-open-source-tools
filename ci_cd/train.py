import sys
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
sys.path.append(os.getcwd())

from House_price_prediction import *
from model.house_model import HouseModel


class TrainModel():
    def __init__(self):
        self.house_main = HousePricePrediction()
        self.house_model = HouseModel()

    def get_current_features(self):
        features_path = "/feature_store/data/house_features.parquet"
        target_path = "/feature_store/data/house_target.parquet"
        X_hist = pd.read_parquet(os.getcwd() + features_path, columns=["area", "mainroad", "bedrooms"])
        Y_hist = pd.read_parquet(os.getcwd() + target_path, columns=["price"])
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
        X_combined = combined_data.drop(columns=["price", "proxy_target"])
        y_combined = combined_data["price"]
        print(combined_data)
        #self.train_model(X_combined, y_combined)
        self.house_model.train_model(X_combined, y_combined, test_size=0.1)
        print("Model re-trained and saved as model.pkl")
        logging.info("Model re-trained and saved as model.pkl")

    def train_model(self, x, y):
        self.params = {
            "fit_intercept": True,
            "positive": False
        }
        model = LinearRegression(**self.params)
        model.fit(x, y)
        with open("model/house_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model re-trained and saved as model.pkl")



if __name__ == "__main__":
    t = TrainModel()
    X_hist = t.get_current_features()
    X_new = t.predict_new_data()
    t.create_and_train_new_dataset_with_target(X_hist, X_new)

