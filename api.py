import uvicorn
import pandas as pd
import joblib
import requests

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException


app = FastAPI()
model_loaded = False

model = joblib.load("model.pkl")


def check_model():
    global model_loaded
    try:
        model = joblib.load("model.pkl")
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        print("Model file not found. Please make sure model.pkl exists.")
    except Exception as e:
        model_loaded = False
        print(f"Error loading model: {str(e)}")

class WineDescription(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class Result(BaseModel):
    result: float

@app.post("/predict", response_model=Result)
def predict(wine_data: WineDescription):
    wine_input = wine_data.dict()
    input_df = pd.DataFrame(wine_input, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.get("/healthcheck")
def healthcheck():
    global model_loaded
    try:
        check_model()
        if model_loaded:
            return {"status": "All good"}
        else:
            raise HTTPException(status_code=503, detail="Model is not found")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Service is unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Unexpected Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)