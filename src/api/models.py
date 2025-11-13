"""
Schemas Pydantic para validación de Request/Response de la API
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Schema para la petición de predicción.
    Acepta una lista de diccionarios con las features del artículo.
    """
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de diccionarios con las features del artículo",
        min_items=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "n_tokens_title": 10,
                        "n_tokens_content": 500,
                        "n_unique_tokens": 0.5,
                        "data_channel_is_entertainment": 1,
                        "weekday_is_monday": 0,
                        "url": "https://example.com/2013/01/01/article-title"
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema para la respuesta de predicción.
    """
    predictions: List[float] = Field(
        ...,
        description="Lista de predicciones (número de shares esperados)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [3456.78]
            }
        }
