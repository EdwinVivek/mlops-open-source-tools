import bentoml
import numpy as np
import csv 
from datetime import datetime 
import pandas as pd 
from itertools import starmap 

@bentoml.service(
    resources={"cpu": "2"}, 
    traffic={"timeout": 10},
    logging={
    "access": {
        "enabled": True,
        "request_content_length": True,
        "request_content_type": True,
        "response_content_length": True,
        "response_content_type": True,
        "skip_paths": ["/metrics", "/healthz", "/livez", "/readyz"],
        "format": {
            "trace_id": "032x",
            "span_id": "016x"
        }
    }
})
class HouseService:
    bento_model = bentoml.models.get("house_price_model:latest")

    def __init__(self):
        self.model = self.bento_model.load_model()
        with open('feedback.csv', 'w', newline='') as file: 
            fieldnames = ["event_timestamp", "area", "bedrooms", "mainroad", "prediction"] 
            writer = csv.DictWriter(file, fieldnames = fieldnames) 
            writer.writeheader() 

    @bentoml.api
    def predict(self, input_data:np.ndarray) -> np.ndarray:
        pred = self.model.predict(input_data)
        print(pred)
        timestamps = pd.date_range( 
            end=pd.Timestamp.now(),  
            start=pd.Timestamp.now(),  
            periods=1,  
            freq=None).to_frame(name="event_timestamp", index=False) 
        with open('feedback.csv', 'a', newline='') as file: 
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC) 
            val = starmap(lambda x,y,z:[x,y,z], np.asarray(input_data).tolist()) 
            data = [] 
            for i in next(val):
                data.append(i)  
            writer.writerow([timestamps.event_timestamp[0], data[0], data[1], data[2], pred[0]])
        return np.asarray(pred)
