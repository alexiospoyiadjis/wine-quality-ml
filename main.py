"""
main.py — FastAPI Model Serving
Loads the PyCaret pipeline saved by discovery.py and exposes a /predict endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# -------------------------------------------------------------------
# Load the saved PyCaret pipeline once at startup
# -------------------------------------------------------------------
model = load_model("best_pipeline")

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predicts wine quality score using a PyCaret ML pipeline.",
    version="1.0.0",
)


class WineFeatures(BaseModel):
    """Input schema matching the wine quality dataset features."""
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
    type: str  # "red" or "white"

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
                "type": "red",
            }
        }


class PredictionResponse(BaseModel):
    """Output schema."""
    prediction: str
    label: str


@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API — visit /docs for Swagger UI"}


@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures):
    """
    Accept wine features as JSON and return a quality prediction.
    """
    try:
        # Convert input to DataFrame (PyCaret expects a DataFrame)
        input_df = pd.DataFrame([wine.model_dump()])

        # Rename columns to match the training data (spaces, not underscores)
        # but keep 'type' as-is since it's a single word
        rename_map = {
            "fixed_acidity": "fixed acidity",
            "volatile_acidity": "volatile acidity",
            "citric_acid": "citric acid",
            "residual_sugar": "residual sugar",
            "free_sulfur_dioxide": "free sulfur dioxide",
            "total_sulfur_dioxide": "total sulfur dioxide",
        }
        input_df.rename(columns=rename_map, inplace=True)

        # Run prediction through the saved pipeline
        predictions = predict_model(model, data=input_df)

        # PyCaret adds 'prediction_label' column
        predicted_quality = str(predictions["prediction_label"].iloc[0])

        return PredictionResponse(
            prediction=predicted_quality,
            label=f"Predicted wine quality: {predicted_quality}/9",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
