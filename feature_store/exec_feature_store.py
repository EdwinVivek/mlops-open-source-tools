import os
import sqlalchemy as db
from feature_store.feature_store import FeastFeatureStore
from feature_store.feature_repo.definitions import house, house_features, housing_source
from feast.data_source import PushMode
import pandas as pd
from datetime import datetime


class ExecuteFeatureStore():
    def __init__(self):
        self.features=[ 
            "house_features:area", 
            "house_features:bedrooms", 
            "house_features:mainroad" 
        ] 

    def save_df_to_postgres(self, X, y, mode='replace'):
        #connstr = 'postgresql+psycopg://postgres:Syncfusion%40123@localhost:5432/feast_offline'
        connstr = 'postgresql+psycopg2://postgres:Syncfusion%40123@localhost:5432/feast_offline'
        engine = db.create_engine(connstr)
        X.to_sql('house_features_sql', engine, if_exists=mode, index=False)
        y.to_sql('house_target_sql', engine, if_exists=mode, index=False)
        print("Pushed data to offline store!")


    def get_feature_store(self):
        fstore = FeastFeatureStore(path=os.path.join(os.getcwd() + "//feature_store//feature_repo"))
        print(fstore.store)
        return fstore
    
    def get_historical_features(self, fstore=None, entity_df=None): 
        if(fstore is None): 
            fstore = self.get_feature_store() 
            fstore.store.apply([house, house_features]) 

        if (entity_df is None): 
            entity_df = fstore.get_entity_dataframe(path=os.path.join(os.getcwd() + "//feature_store//data//house_target.parquet")) 

        hist_df = fstore.get_historical_features(entity_df, self.features) 
        return hist_df 
 
    def get_online_features(self, fstore, entity_df=None): 
        if (entity_df is None): 
            entity_df = fstore.get_entity_dataframe(path=os.path.join(os.getcwd() + "//feature_store//data//house_target.parquet")) 
        entity_rows = entity_df.to_dict(orient="records") 
        online_df = fstore.get_online_features(entity_rows, self.features) 
        return online_df
    
    def push_feedback(self, name, data:pd.DataFrame):
        fstore = self.get_feature_store()
        fstore.store.push(push_source_name=name, df=data, to=PushMode.ONLINE_AND_OFFLINE)
        print("Data pushed to offline store")

    def materialize(self, end_date = datetime.now(), start_date=None, increment=False, fstore=None):
        if(fstore is None):
            fstore = self.get_feature_store()
        if not increment:
            #Code for loading features to online store between two dates
            fstore.materialize(
                end_date=end_date,
                start_date=start_date)
        else:
            fstore.materialize(end_date=end_date, increment= increment)