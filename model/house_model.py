from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

class HouseModel:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.params = None

    def train_model(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.features, self.target)
        self.params = {
            "fit_intercept": True,
            "positive": False
        }
        model = LinearRegression(**self.params)
        model.fit(self.x_train, self.y_train)

        # Save the trained model
        with open("model\\house_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Model trained and saved as model.pkl")

    # Load model 
    def load_model(self):
        #self.model = None
        # Load the saved model
        with open("model\\house_regression_model.pkl", "rb") as f:
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