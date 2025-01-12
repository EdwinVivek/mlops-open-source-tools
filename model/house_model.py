from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model

EXPERIMENT_NAME = "House price prediction"
EXPERIMENT_URI = "http://localhost:5000"

class HouseModel:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.params = None

    def train_model(self, features, target, test_size=0.25):
        self.params = {
            "fit_intercept": True,
            "positive": False
        }
        model = LinearRegression(**self.params)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, target, test_size=test_size)
        model.fit(self.x_train, self.y_train)

        # Save the trained model
        with open("model/house_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model trained and saved as model.pkl")

    # Load model 
    def load_model(self):
        #self.model = None
        # Load the saved model
        with open("model/house_regression_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
        
    def predict(self, data):
        model = self.load_model()
        # Test data (sample input for prediction)
        #test_data = [5.1, 3.5, 1.4, 0.2]  # Example features
        prediction = model.predict(data)
        #print(f"Prediction for {test_data}: {int(prediction[0])}")
        return prediction

    def metrics(self, y_pred):
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        mae = mean_absolute_error(self.y_test, y_pred)
        metric_dict = {
            "rmse": rmse,
            "mae": mae
        }
        return metric_dict
    
    def mlflow_config(self):
        mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        mlflow.set_tracking_uri(uri= EXPERIMENT_URI)

        try:
            exp= mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if (exp is not None):
                mlflow.set_experiment(experiment_id=exp.experiment_id)
        except:
            exp_id = mlflow.create_experiment(name =EXPERIMENT_NAME)
            mlflow.set_experiment(experiment_id=exp_id)
            
    def register(self):
        with mlflow.start_run(log_system_metrics=True) as run:
            mlflow.log_params(self.params)
            y_pred = self.predict(self.x_test)
            mlflow.log_metrics(self.metrics(y_pred))
            signature = infer_signature(np.array(self.x_train), np.array(self.predict(self.x_test)))
            model_info = log_model(
                sk_model=self.load_model(),
                artifact_path="house_model",
                signature=signature,
                input_example= self.x_train,
                registered_model_name="house_price_prediction"
            )
        return model_info