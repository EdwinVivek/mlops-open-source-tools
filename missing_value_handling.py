import logging
from abc import ABC, abstractmethod
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValueStrategy(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Drop missing called")
        return df.dropna()

class FillMissingValueStrategy(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Fill missing called")
        return df.fillna(0)

class MissingValueContext:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
         self.strategy = strategy

    def execute(self, df):
        logging.info("Execute missing called")
        return self.strategy.handle(self, df)
        