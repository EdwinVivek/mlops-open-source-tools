import os
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
import pandas as pd
from datetime import datetime, timedelta


class FeastFeatureStore:
    def __init__(self, path):
        self.store = FeatureStore(repo_path=path)
        self.retrievalJob = None

    def get_entity_dataframe(self, path) ->pd.DataFrame:
        # Reading our targets as an entity DataFrame
        entity_df = pd.read_parquet(path=path)
        return entity_df

    def get_historical_features(self, entity_df:pd.DataFrame, features) ->pd.DataFrame:
        self.retrievalJob = self.store.get_historical_features(
            entity_df = entity_df,
            features=features
        )
        return self.retrievalJob.to_df()
    
    def save_dataset(self, file_name, path):
        self.store.create_saved_dataset(
            from_=self.retrievalJob,
            name=file_name,
            storage=SavedDatasetFileStorage(path)
        )
        print(str.format("File {0} saved successfully", file_name))

    def materialize(self,end_date, start_date=None, increment=False):
        if not increment:
            #Code for loading features to online store between two dates
            self.store.materialize(
                end_date=end_date,
                start_date=start_date)
        else:
            self.store.materialize_incremental(end_date=end_date)

    def get_online_features(self, entity_rows, features) -> pd.DataFrame:
        retrievalJob = self.store.get_online_features(
            entity_rows=entity_rows,
            features=features
        )
        return retrievalJob.to_df()
        
    
        
        