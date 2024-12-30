from abc import ABC, abstractmethod
import os
import pandas as pd
import zipfile

#Factory design pattern

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path:str) -> pd.DataFrame:
        pass


class ZipFileIngestor(DataIngestor):
    def ingest(self, file_path:str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("Not a zip file")

        with zipfile.ZipFile(file_path, "r") as file:
            file.extractall("myfolder")

        extracted = os.listdir("myfolder")
        csv_file = [file for file in extracted if file.endswith(".csv")]

        if len(csv_file) == 0:
            raise FileNotFoundError("No CSV file found")
        if len(csv_file) > 1:
            raise ValueError("Multiple CSV files found. Specify one")
                             
        path = os.path.join("myfolder", csv_file[0])
        df = pd.read_csv(path)
        return df

class CsvFileIngestor(DataIngestor):
    def ingest(self, file_path) -> pd.DataFrame:
        raise NotImplementedError("Implement csv file reader")

#Factory 
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(extension_type) -> DataIngestor:
        file_types = {
            ".zip": ZipFileIngestor(),
            ".csv": CsvFileIngestor()
        }
        ingestor = file_types.get(extension_type, None)
        return ingestor