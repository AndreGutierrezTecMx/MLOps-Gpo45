"""
FastAPI Application para servir el modelo
"""

from fastapi import FastAPI, HTTPException
import pandas as pd

from src.api.models import PredictionRequest, PredictionResponse
from src.modeling.model_predictor import ModelPredictor
from src.constants.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_METRIC
)

# Cargar el modelo al iniciar
predictor = ModelPredictor(
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_experiment=MLFLOW_EXPERIMENT,
    metric=MODEL_METRIC
)

# Inicializar FastAPI app
app = FastAPI(
    title="MLOps-Gpo45 Model API",
    description="API para predecir el número de shares de artículos",
    version="1.0.0"
)


@app.get("/")
async def root():
    """
    Endpoint raíz con información de la API.
    """
    return {
        "message": "MLOps-Gpo45 Model API",
        "version": "1.0.0"
    }


@app.get("/model-info")
async def model_info():
    """
    Endpoint para obtener información del modelo cargado.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if predictor.model_info is None:
        raise HTTPException(
            status_code=503, detail="Información del modelo no disponible")

    return predictor.model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint principal para realizar predicciones.

    Recibe un JSON con una lista de registros y retorna las predicciones.
    """
    try:
        if predictor is None or predictor.model is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible")

        # Convertir request a DataFrame
        df = pd.DataFrame(request.data)

        # Realizar predicción
        predictions = predictor.predict(df)

        return PredictionResponse(
            predictions=predictions.tolist()
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
