from fastapi import FastAPI
from pydantic import BaseModel 
from enum import Enum
import uvicorn
from scripts.session_3.utils import utils_router
from scripts.session_3.predict import housing_router

class Method(str, Enum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    divide = "divide"

class CalculateRequest(BaseModel):
    method: Method
    a: float
    b: float
    
class CalculateResponse(BaseModel):
    result: float

app = FastAPI()

app.include_router(utils_router)
app.include_router(housing_router)

@app.get("/")
def root():
    return {"message": "Hello World!"}

@app.get("/health")
def health(dump_input: int = 0):
    if dump_input > 10:
        return { "message" : f"dump_input larger than 10: input value: {dump_input}" }
    else:
        return { "message" : f"dump_input less than 10: input value: {dump_input}" }

@app.post("/calculate", response_model=CalculateResponse)
def calculate(request: CalculateRequest) -> CalculateResponse:
    if request.method == Method.add:
        result = request.a + request.b
    elif request.method == Method.subtract:
        result = request.a - request.b
    elif request.method ==  Method.multiply:
        result = request.a * request.b
    elif request.method == Method.divide:
        if request.b == 0:
            raise ValueError("Division by zero is not allowed")
        else:
            result = request.a / request.b
    else:
        raise ValueError("Invalid method")
    
    return CalculateResponse(result=result)

class PredictRequest(BaseModel):
    avg_area_income: list[float]
    avg_area_house_age: list[float]
    avg_area_number_of_rooms: list[float]
    avg_area_number_of_bedrooms: list[float]
    area_population: list[float]
    
class PredictResponse(BaseModel):
    predicted_price: float

@app.post("/predict", response_model=PredictResponse)
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
          
          
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)