import pandas as pd
from abc import ABC, abstractmethod

#Strategy desing pattern

class DataInspection(ABC):
    @abstractmethod
    def inspect(self, df:pd.DataFrame):
        pass

class DataTypeInspection(DataInspection):
     def inspect(self, df:pd.DataFrame):
         print("Data types and Non-null columns")
         print(df.info())


    
class SummaryDataInspection(DataInspection):
    def inspect(self, df:pd.DataFrame):
        print("Summary for numerical varaibles")
        print(df.describe())
        print("Summary for categorical varaibles")
        print(df.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy: DataTypeInspection):
        self._strategy = strategy

    def set_strategy(self, strategy: DataTypeInspection):\
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame):
        self._strategy.inspect(df)