import mlflow.sklearn
import pandas as pd

model_name = "housing_predict_1"
model_version = "4"

mlflow.set_tracking_uri(uri="http://localhost:8080")

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

def create_sample_data():
    data = {
        "Avg. Area Income": [79545.458574, 79248.642455],
        "Avg. Area House Age": [5.682861, 5.569882],
        "Avg. Area Number of Rooms": [7.009188, 6.794465],
        "Avg. Area Number of Bedrooms": [4.09, 3.94],
        "Area Population": [23086.800000, 40173.000000],
    }
    return pd.DataFrame(data)

predictions = model.predict(create_sample_data())
print(f"Predicted price: ${predictions[0]:,.2f}")