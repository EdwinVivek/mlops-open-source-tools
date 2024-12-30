#strategy pattern
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#abstract class
class UnivariateStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature:str):
        pass



#concrete strategies
class NumericalUnivariateAnalysis(UnivariateStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], bins=10, kde= True)
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(10,6))
        sns.catplot(df[feature], kind="bar")
        plt.show()



#context
class UnivariateContext:
    def __init__(self, strategy):
        self.strategy = strategy

    def analyzestrategy(self, df, feature):
        self.strategy.analyze(self, df, feature)