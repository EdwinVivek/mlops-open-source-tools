import sys
import os
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
from deepchecks.tabular.suites import data_integrity

from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())


class ModelValidation():
    def __init__(self):
        pass

    def get_current_features(self):
       features_path = "/feature_store/data/house_features.parquet"
       target_path = "/feature_store/data/house_target.parquet"
       X_hist = pd.read_parquet(os.getcwd() + features_path, columns=["area", "mainroad", "bedrooms"])
       Y_hist = pd.read_parquet(os.getcwd() + target_path, columns=["price"])
       return X_hist,Y_hist
    
    def validate(self):
        features ,target = self.get_current_features()
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
        train_data = Dataset(x_train, label=y_train)
        test_data = Dataset(x_test, label=y_test)
        data_suite = data_integrity()
        result = data_suite.run(train_data, test_data)
        report_path = "deepchecks_report.html"
        result.save_as_html(report_path)
        if not result.passed():
            print("Model validation failed! Check deepchecks_report.html", save_as_pickle=False)
            exit(1) 
        else:
            print("Model validation passed!")
        #assert result.passed(), "Validation failed! Check deepchecks_report.html"


if __name__ == "__main__":
    validation = ModelValidation()
    validation.validate()
