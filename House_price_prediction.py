import os
from datetime import datetime,timedelta
import sys
from importlib import reload 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
import sqlalchemy as db

from data_ingestor import *
from basic_data_inspection import DataInspector, DataTypeInspection, SummaryDataInspection
from univariate_analysis import UnivariateContext, NumericalUnivariateAnalysis, CategoricalUnivariateAnalysis
from missing_value_handling import *
from data_encoding import DataEncoding

from feast import FeatureStore
from feature_store.feature_store import FeastFeatureStore
from feature_store.feature_repo.definitions import house, house_features, housing_source

import mlflow
from mlflow.models import infer_signature
from mlflow.sklearn import log_model, load_model
from model.house_model import HouseModel

from serving.model_serving import BentoModel
from bentoml import HTTPServer
import numpy as np

from monitoring.evidently_monitoring import *
from evidently.ui.dashboards import CounterAgg, PlotType
from evidently.renderers.html_widgets import WidgetSize


class HousePricePrediction():
    def __init__(self):
        self.df = None

    def load_and_inspect_data(self):
        path =os.path.abspath(os.path.join(os.getcwd(), os.pardir)+ "//data/Housing.zip")
        file_ext = os.path.splitext(path)[1]
        ingestor_type = DataIngestorFactory.get_data_ingestor(file_ext)
        self.df = ingestor_type.ingest(path)
        print(self.df.head())

        data_inspector = DataInspector(DataTypeInspection())
        data_inspector.execute_strategy(self.df)

        data_inspector.set_strategy(SummaryDataInspection())
        data_inspector.execute_strategy(self.df)

        univariate = UnivariateContext(NumericalUnivariateAnalysis)
        univariate.analyzestrategy(self.df, "price")

        univariate_cat = UnivariateContext(CategoricalUnivariateAnalysis)
        univariate_cat.analyzestrategy(self.df, "guestroom")

    def process_data(self) -> pd.DataFrame:
        #handle_missing_values
        missing_value_handling = MissingValueContext(DropMissingValueStrategy)
        newdf = missing_value_handling.execute(self.df)
        missing_value_handling.set_strategy(FillMissingValueStrategy)
        out_df = missing_value_handling.execute(newdf)
        return self.encode_data(out_df)

    def encode_data(self, df) -> pd.DataFrame:
        binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        cat_columns = ['furnishingstatus']
        numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        encode = DataEncoding()
        bin_df = encode.binary_encoding(df, binary_columns)
        cat_df = encode.categorical_encoding(bin_df, cat_columns)
        num_df = encode.numerical_encoding(cat_df, numerical_columns)
        return num_df


    def split_and_save_data(self, df):
        # Splitting the dataset into features (X) and target (y)
        X = df.drop(columns=['price'])
        y = pd.DataFrame(df['price'])

        timestamps = pd.date_range(
            end=pd.Timestamp.now(), 
            start=pd.Timestamp.now(), 
            periods=len(df), 
            freq=None).to_frame(name="event_timestamp", index=False)

        X["event_timestamp"] = timestamps.event_timestamp
        X["house_id"] = range(1, len(df) + 1)  # Assign unique IDs to each house

        y["event_timestamp"] = timestamps.event_timestamp
        y["house_id"] = range(1, len(df) + 1)  # Assign unique IDs to each house

        # Display the first few rows of the preprocessed features
        print(X.head())
        print(y.head())

        # Writing our DataFrames to parquet files
        X.to_parquet(path= os.path.join(os.getcwd() ,'feature_store/data/house_features.parquet'))
        y.to_parquet(path=os.path.join(os.getcwd() ,'feature_store/data/house_target.parquet'))
        self.save_df_to_postgres(X,y)


    def save_df_to_postgres(self, X, y, mode='replace'):
        #connstr = 'postgresql+psycopg://postgres:Syncfusion%40123@localhost:5432/feast_offline'
        connstr = 'postgresql+psycopg2://postgres:root@localhost:5432/feast_offline'
        engine = db.create_engine(connstr)
        X.to_sql('house_features_sql', engine, if_exists=mode, index=False)
        y.to_sql('house_target_sql', engine, if_exists=mode, index=False)
        print("Pushed data to offline store!")


    # <h2> Feast Feature store
    def get_feature_store(self):
        store = FeastFeatureStore(path=os.path.join(os.getcwd() + "//feature_store//feature_repo"))
        print(store.store)
        return store
    
    def execute_feauture_store(self, store=None): 
        if(store is None): 
            store = self.get_feature_store() 
        self.get_historical_features(store) 
        self.get_online_features(store) 

    def get_historical_features(self, store=None, entity_df=None): 
        if(store is None): 
            store = self.get_feature_store() 
            store.store.apply([house, house_features]) 

        if (entity_df is None): 
            entity_df = store.get_entity_dataframe(path=os.path.join(os.getcwd() + "//feature_store//data//house_target.parquet")) 

        features=[ 
            "house_features:area", 
            "house_features:bedrooms", 
            "house_features:mainroad" 
        ] 
        hist_df = store.get_historical_features(entity_df, features) 
        return hist_df 
 
    def get_online_features(self, store, entity_df=None): 
        features=[ 
            "house_features:area", 
            "house_features:bedrooms", 
            "house_features:mainroad" 
        ] 
        if (entity_df is None): 
            entity_df = store.get_entity_dataframe(path=os.path.join(os.getcwd() + "//feature_store//data//house_target.parquet")) 
        entity_rows = entity_df.to_dict(orient="records") 
        online_df = store.get_online_features(entity_rows, features) 
        return online_df

    def materialize(self, end_date = datetime.now(), start_date=None, increment=False, store=None):
        if(store is None):
            store = self.get_feature_store()
        if not increment:
            #Code for loading features to online store between two dates
            store.materialize(
                end_date=end_date,
                start_date=start_date)
        else:
            store.materialize(end_date=end_date, increment= increment)


    # <h2>MLFlow
    def experiment_tracking(self):
        features, target = self.execute_feauture_store()
        model = HouseModel()
        lreg_model = model.train_model(features, target)

        print(model.x_train.shape)
        print(model.x_test.shape)


        y_train_pred = model.predict(model.x_train)
        #train_metrics = model.metrics(y_train_pred)
        #train_metrics

        y_pred = model.predict(model.x_test)
        test_metrics = model.metrics(y_pred)
        test_metrics
        model_info = self.register_model(model)
        self.model_monitoring(model, y_train_pred, y_pred)
        self.model_serving(model, model_info)

    def register_model(self, model:HouseModel):
        mlflow.get_experiment_by_name("House price prediction")
        mlflow.set_tracking_uri(uri="http://localhost:5000")

        try:
            exp= mlflow.get_experiment_by_name("House price prediction")
            if (exp is not None):
                mlflow.set_experiment(experiment_id=exp.experiment_id)
        except:
            exp_id = mlflow.create_experiment(name ="House price prediction")
            mlflow.set_experiment(experiment_id=exp_id)

        
        with mlflow.start_run(log_system_metrics=True) as run:
            mlflow.log_params(model.params)
            y_pred = model.predict(model.x_test)
            mlflow.log_metrics(model.metrics(y_pred))
            signature = infer_signature(np.array(model.x_train), np.array(model.predict(model.x_test)))
            model_info = log_model(
                sk_model=model.load_model(),
                artifact_path="house_model",
                signature=signature,
                input_example= model.x_train,
                registered_model_name="house_price_prediction"
            )
        return model_info
        


    # <h2>Evidently
    def model_monitoring(self, model:HouseModel, y_train_pred, y_pred):
        monitoring = Monitoring(DataDriftReport())
        ws = monitoring.create_workspace("house price monitoring")
        project = monitoring.search_or_create_project("house price project", ws)
        

        reference = model.x_train.copy()
        reference["price"] = model.y_train.copy()

        current = model.x_test.copy()
        current["price"] = model.y_test.copy()
        
        #Data drift report
        print(monitoring.current_strategy)
        monitoring.execute_strategy(reference, current, ws)

        #Data quality report
        monitoring.set_strategy = DataQualityReport()
        monitoring.execute_strategy(reference, current, ws)

        #Regression report
        reference_with_pred = reference.copy()
        reference_with_pred["prediction"] = y_train_pred
        reference_with_pred

        current_with_pred = current.copy()
        current_with_pred["prediction"] = y_pred
        current_with_pred

        column_mapping = ColumnMapping()
        column_mapping.target = "price"
        column_mapping.prediction = "prediction"

        monitoring.set_strategy = RegressionReport()
        monitoring.execute_strategy(reference_with_pred, current_with_pred, ws, column_mapping)

        #Target Drift Report
        monitoring.set_strategy = TargetDriftReport()
        monitoring.execute_strategy(reference_with_pred, current_with_pred, ws, column_mapping)

        #Test suite
        monitoring.set_strategy = DataDriftTestReport()
        monitoring.execute_strategy(reference, current, ws)


        
        #Create dashboard panels
        monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "House price Monitoring dashboard",
            tags = [],  
            metric_id = None,
            field_path = "",
            legend = "",
            text = "",
            agg = CounterAgg.NONE,
            size = WidgetSize.FULL
        )

        monitoring.add_dashboard_panel(
            project, panel_type="Counter", 
            title = "Number of drifted columns",
            tags = [],  
            metric_id = "DatasetDriftMetric",
            field_path = "Drifted Columns",
            legend = "",
            text = "",
            agg = CounterAgg.LAST,
            size = WidgetSize.HALF
        )

        monitoring.add_dashboard_panel(
            project, panel_type="Plot", 
            title = "Share of drifted columns",
            tags = [],  
            metric_id = "DatasetDriftMetric",
            field_path = "share_of_drifted_columns",
            metric_args = {},
            legend = "share",
            plot_type = PlotType.LINE,
            size = WidgetSize.HALF,
                agg = CounterAgg.SUM
        )

        project.show_dashboard()
        #monitoring.delete_dashboard(project)



        # <h2>BentoML
    def model_serving(self, model, model_info):
        bento_model = BentoModel()
        model_name= bento_model.import_model("house_price_model", model_info.model_uri)
        model_name


        #load mlflow model
        b_model = bento_model.load_model(model_name)
        pred = bento_model.predict(b_model, model.x_test[:1])


        #get bento mlflow model
        b_runner = bento_model.get_model(model_name)
        b_runner.predict.run(model.x_test[:1])

        #start the server from python
        server = HTTPServer("house_service:latest", production=True, port=3000, host='127.0.0.1')
        server.start()
        client = server.get_client()

        with server.start() as client:
            result = client.predict(model.x_test[:1])
            print(result)


        with bentoml.SyncHTTPClient("http://localhost:3000") as client:
            result = client.predict(
                input_data=model.x_test[:1],
            )
            print(result)


if __name__ == "__main__":
    house = HousePricePrediction()
    house.load_and_inspect_data()
    df = house.process_data()
    house.split_and_save_data(df)
    house.execute_feauture_store()
    house.materialize_increment()
    house.experiment_tracking()
    
    



