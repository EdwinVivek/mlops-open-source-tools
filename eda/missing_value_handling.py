import logging
from abc import ABC, abstractmethod
import pandas as pd

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValueStrategy(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

class FillMissingValueStrategy(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0)

class MissingValueContext:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
         self.strategy = strategy
    
    def check_missing(self, df):
        return df.isnull().sum().sort_values(ascending=False)

    def execute(self, df):
        return self.strategy.handle(self, df)
        