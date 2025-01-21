#strategy pattern
from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#abstract class
class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature:str):
        pass


#concrete strategies
class NumericalUnivariateAnalysis(AnalysisStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], bins=10, kde= True)
        plt.show()


class CategoricalUnivariateAnalysis(AnalysisStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(10,6))
        sns.catplot(df[feature], kind="bar")
        plt.show()

class BivariateHeatmapAnalysis(AnalysisStrategy):
    def analyze(self, df, feature):
       sns.heatmap(df[feature].corr(),annot=True)


#context
class AnalysisContext:
    def __init__(self, strategy):
        self.strategy = strategy

    def analyze_strategy(self, df, feature):
        self.strategy.analyze(self, df, feature)


