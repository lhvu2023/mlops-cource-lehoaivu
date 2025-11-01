from pydantic import BaseModel 
from fastapi import APIRouter

housing_router = APIRouter(prefix="/housing")

class PredictRequest(BaseModel):
    avg_area_income: list[float]
    avg_area_house_age: list[float]
    avg_area_number_of_rooms: list[float]
    avg_area_number_of_bedrooms: list[float]
    area_population: list[float]
    
class PredictResponse(BaseModel):
    predicted_price: float


@housing_router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    import mlflow.sklearn
    import pandas as pd

    model_name = "housing_predict_1"
    model_version = "1"

    mlflow.set_tracking_uri(uri="http://localhost:8080")

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    data = {
        "Avg. Area Income": request.avg_area_income,
        "Avg. Area House Age": request.avg_area_house_age,
        "Avg. Area Number of Rooms": request.avg_area_number_of_rooms,
        "Avg. Area Number of Bedrooms": request.avg_area_number_of_bedrooms,
        "Area Population": request.area_population,
    }
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    
    return PredictResponse(predicted_price=predictions[0])