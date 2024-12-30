import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataEncoding:
    def binary_encoding(self, data:pd.DataFrame, binary_columns) -> pd.DataFrame:
        # Encoding categorical variables
        for column in binary_columns:
            data[column] = data[column].apply(lambda x: 1 if x == 'yes' else 0)
        return data

    def categorical_encoding(self, data:pd.DataFrame, cat_columns) -> pd.DataFrame:
        # Encoding the furnishing status column using LabelEncoder
        furnishing_encoder = LabelEncoder()
        for column in cat_columns:
            data[column] = furnishing_encoder.fit_transform(data[column])
        return data

    def numerical_encoding(self, data:pd.DataFrame, num_columns) -> pd.DataFrame:
        # Standardizing numerical features
        scaler = StandardScaler()
        for column in num_columns:
            data[num_columns] = scaler.fit_transform(data[num_columns])
        return data



