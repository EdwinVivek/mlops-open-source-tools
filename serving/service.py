import bentoml
import numpy as np

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

    @bentoml.api
    def predict(self, input_data:np.ndarray) -> np.ndarray:
        pred = self.model.predict(input_data)
        return np.asarray(pred)
