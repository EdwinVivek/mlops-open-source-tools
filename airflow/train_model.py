import sys
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import sqlalchemy as db
import pickle
import logging
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from feature_store.exec_feature_store import ExecuteFeatureStore
from model.house_model import HouseModel
from serving.model_serving import BentoModel

logging.basicConfig(   
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True
)


class TrainModel():
    def __init__(self):
        self.f_store = ExecuteFeatureStore()
        self.house_model = HouseModel()

    def get_current_features(self):
        connstr = 'postgresql+psycopg2://postgres:root@localhost:5432/feast_offline'
        engine = db.create_engine(connstr)
        logging.info("engine init")
        Y_hist = pd.read_sql(str.format("select house_id, price from public.house_target_sql"), con=engine)
        store = self.f_store.get_feature_store()
        logging.info("store init")
        X_hist = self.f_store.get_online_features(store, pd.DataFrame(Y_hist["house_id"]))
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
        last_id = int(X_hist.loc[X_hist["house_id"].idxmax()]["house_id"])
        X_new["house_id"] = range(last_id+1, last_id + len(X_new)+1)
        new_data_combined = X_new.copy()
        new_data_combined["price"] = new_data_combined["proxy_target"]      
        historical_data_combined = X_hist.copy()
        combined_data = pd.concat([historical_data_combined, new_data_combined], ignore_index=True)
        X_combined = combined_data.drop(columns=["price", "proxy_target", "house_id"])
        y_combined = combined_data["price"]
        print(combined_data)
        logging.info("combine data called")
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
        logging.info("train before called")
        model.fit(x, y)
        logging.info("train data called")
        with open("model/house_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model re-trained and saved as model.pkl")
        logging.info("Model re-trained and saved as model.pkl")

    def register_model(self):
        self.house_model.mlflow_config()
        model_info = self.house_model.register()
        return model_info
    
    def serve_model(self, model_info):
        bento_model = BentoModel()
        model_name = bento_model.import_model("house_price_model", model_info.model_uri)
        logging.info(str.format('Imported model {0} to BentoML', model_name))

if __name__ == "__main__":
    os.chdir("/home/edwin/git/mlops-open-source-tools/")
    t = TrainModel()
    X_hist = t.get_current_features()
    X_new = t.predict_new_data()
    t.create_and_train_new_dataset_with_target(X_hist, X_new)
    m_info = t.register_model()
    t.serve_model(m_info)
    logging.info("Model trained and registered")

