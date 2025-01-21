from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pickle
import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model

EXPERIMENT_NAME = "House price prediction"
EXPERIMENT_URI = "http://localhost:5000"

class HouseModel:
    def __init__(self):
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.grid_search = None

    # Custom scorer for MSE
    def mse_scorer(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def train_model(self, features, target, test_size=0.25):
        params = {
            "fit_intercept": [True, False],
            "positive": [True,False]
        }
        model = LinearRegression()

        # Set up GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=make_scorer(self.mse_scorer, greater_is_better=False),  # Negative MSE
            cv=5,   
            return_train_score=True
        )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, target, test_size=test_size)
        self.grid_search.fit(self.x_train, self.y_train)

        # Save the trained model
        with open("model/house_regression_model.pkl", "wb") as f:
            pickle.dump(self.grid_search.best_estimator_, f)
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
        r2 = r2_score(self.y_test, y_pred)
        metric_dict = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        return metric_dict
    
    def configure_mlflow(self):
        mlflow.set_tracking_uri(uri= EXPERIMENT_URI)

        try:
            exp= mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if (exp is not None):
                mlflow.set_experiment(experiment_id=exp.experiment_id)
        except:
            exp_id = mlflow.create_experiment(name =EXPERIMENT_NAME)
            mlflow.set_experiment(experiment_id=exp_id)
        finally:
            return mlflow.get_experiment_by_name("House price prediction")

    def log_gridsearch(self):
        for i, params in enumerate(self.grid_search.cv_results_["params"]):
            with mlflow.start_run(run_name="child_run" + i, nested=True):  # Use nested=True for sub-runs
                # Get metrics from cv_results_
                mean_test_score = self.grid_search.cv_results_['mean_test_score'][i]
                std_test_score = self.grid_search.cv_results_['std_test_score'][i]
                
                # Log parameters and cross-validation metrics
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_score", mean_test_score)
                mlflow.log_metric("std_cv_score", std_test_score)

                # Refit model on the best parameters and evaluate on test data
                model = self.grid_search.estimator.set_params(**params)
                model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_test)
                mlflow.log_metrics(self.metrics(y_pred))
                mlflow.sklearn.log_model(model, "model")

                print(f"Logged run with params: {params}, mean_cv_score: {mean_test_score:.4f}, std_test_score: {std_test_score:.4f}")
 
    def register(self):
        with mlflow.start_run(run_name="LinearReg_GridSearch_Best", log_system_metrics=True) as run:
            # Log the best parameters and metrics
            best_params = self.grid_search.best_params_
            best_score = self.grid_search.best_score_
            mlflow.log_params(best_params)
            mlflow.log_metric("best_mean_cv_score", best_score)
            y_pred = self.predict(self.x_test)
            mlflow.log_metrics(self.metrics(y_pred))

            #Define signature
            signature = infer_signature(np.array(self.x_train), np.array(self.predict(self.x_test)))

            # Log all runs for each parameter combination
            self.log_gridsearch()

            #Log and Register best model
            model_info = log_model(
                sk_model=self.grid_search.best_estimator_,
                artifact_path="house_model",
                signature=signature,
                input_example= self.x_train,
                registered_model_name="house_price_prediction"
            )
        return model_info