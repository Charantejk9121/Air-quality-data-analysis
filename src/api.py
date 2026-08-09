"""
api.py
------
FastAPI service for the Air Quality project.

Exposes the trained AQI-bucket classifier and dataset summary stats over
HTTP, so the model can be consumed by anything (the Streamlit dashboard,
a mobile app, curl, another service) rather than only being usable inside
one Python script.

Run:
    uvicorn src.api:app --reload

Docs (auto-generated):
    http://127.0.0.1:8000/docs
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from data_pipeline import load_and_clean
from model import FEATURES, predict_bucket_with_confidence, train

MODEL_PATH = "outputs/aqi_model.joblib"
DATA_PATH = "data/city_day.csv"

state = {"model_bundle": None, "df": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the cleaned dataset and the trained model (training
    # on the fly if no saved model exists yet, e.g. first deploy).
    state["df"] = load_and_clean(DATA_PATH)
    if not os.path.exists(MODEL_PATH):
        train(state["df"], model_out=MODEL_PATH)
    state["model_bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="India Air Quality API",
    description="Serves AQI category predictions and dataset summary stats "
                 "from CPCB city-day air quality data (2015-2020).",
    version="1.0.0",
    lifespan=lifespan,
)


class PollutantReading(BaseModel):
    """One day's pollutant readings used to predict the AQI category."""
    PM2_5: float = Field(..., alias="PM2.5", ge=0, description="Fine particulate matter, ug/m3")
    PM10: float = Field(..., ge=0)
    NO: float = Field(0, ge=0)
    NO2: float = Field(0, ge=0)
    NOx: float = Field(0, ge=0)
    NH3: float = Field(0, ge=0)
    CO: float = Field(0, ge=0)
    SO2: float = Field(0, ge=0)
    O3: float = Field(0, ge=0)
    Benzene: float = Field(0, ge=0)
    Toluene: float = Field(0, ge=0)
    Xylene: float = Field(0, ge=0)
    Month: int = Field(..., ge=1, le=12)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "PM2.5": 85.0, "PM10": 140.0, "NO": 12.0, "NO2": 30.0,
                "NOx": 35.0, "NH3": 15.0, "CO": 1.2, "SO2": 10.0, "O3": 25.0,
                "Benzene": 2.0, "Toluene": 5.0, "Xylene": 1.0, "Month": 12
            }
        }


class PredictionResponse(BaseModel):
    predicted_bucket: str
    confidence: float
    probabilities: dict


class CitySummary(BaseModel):
    city: str
    state: str | None
    avg_aqi: float
    avg_pm25: float
    most_common_bucket: str
    n_records: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model_bundle"] is not None}


@app.get("/cities", response_model=list[str])
def list_cities():
    """All cities available in the dataset."""
    return sorted(state["df"]["City"].unique().tolist())


@app.get("/cities/{city}/summary", response_model=CitySummary)
def city_summary(city: str):
    """Summary stats for one city. Case-sensitive, matches dataset naming."""
    df = state["df"]
    sub = df[df["City"] == city]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found. See /cities for valid names.")
    return CitySummary(
        city=city,
        state=sub["State"].iloc[0] if sub["State"].notna().any() else None,
        avg_aqi=round(sub["AQI"].mean(), 1),
        avg_pm25=round(sub["PM2.5"].mean(), 1),
        most_common_bucket=sub["AQI_Bucket"].mode().iloc[0],
        n_records=len(sub),
    )


@app.get("/model/info")
def model_info():
    """Feature importances and metadata for the currently loaded model."""
    clf = state["model_bundle"]["model"]
    le = state["model_bundle"]["label_encoder"]
    importances = dict(sorted(
        zip(FEATURES, clf.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True
    ))
    return {
        "features": FEATURES,
        "classes": le.classes_.tolist(),
        "feature_importances": importances,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: PollutantReading):
    """Predict the AQI category for a given day's pollutant readings."""
    readings_dict = reading.model_dump(by_alias=True)
    result = predict_bucket_with_confidence(state["model_bundle"], readings_dict)
    return PredictionResponse(**result)