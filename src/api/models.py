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
                        "n_non_stop_words": 0.8,
                        "n_non_stop_unique_tokens": 0.6,
                        "num_hrefs": 5,
                        "num_self_hrefs": 2,
                        "num_imgs": 3,
                        "num_videos": 0,
                        "average_token_length": 4.5,
                        "num_keywords": 5,
                        "data_channel_is_lifestyle": 0,
                        "data_channel_is_entertainment": 1,
                        "data_channel_is_bus": 0,
                        "data_channel_is_socmed": 0,
                        "data_channel_is_tech": 0,
                        "data_channel_is_world": 0,
                        "kw_min_min": 0,
                        "kw_max_min": 0,
                        "kw_avg_min": 0,
                        "kw_min_max": 0,
                        "kw_max_max": 0,
                        "kw_avg_max": 0,
                        "kw_min_avg": 0,
                        "kw_max_avg": 0,
                        "kw_avg_avg": 0,
                        "self_reference_min_shares": 0,
                        "self_reference_max_shares": 0,
                        "self_reference_avg_sharess": 0,
                        "weekday_is_monday": 0,
                        "weekday_is_tuesday": 0,
                        "weekday_is_wednesday": 0,
                        "weekday_is_thursday": 0,
                        "weekday_is_friday": 0,
                        "weekday_is_saturday": 0,
                        "weekday_is_sunday": 0,
                        "is_weekend": 0,
                        "LDA_00": 0.2,
                        "LDA_01": 0.3,
                        "LDA_02": 0.1,
                        "LDA_03": 0.2,
                        "LDA_04": 0.2,
                        "global_subjectivity": 0.5,
                        "global_sentiment_polarity": 0.1,
                        "global_rate_positive_words": 0.3,
                        "global_rate_negative_words": 0.1,
                        "rate_positive_words": 0.75,
                        "rate_negative_words": 0.25,
                        "avg_positive_polarity": 0.4,
                        "min_positive_polarity": 0.2,
                        "max_positive_polarity": 0.6,
                        "avg_negative_polarity": -0.3,
                        "min_negative_polarity": -0.5,
                        "max_negative_polarity": -0.1,
                        "title_subjectivity": 0.4,
                        "title_sentiment_polarity": 0.2,
                        "abs_title_subjectivity": 0.4,
                        "abs_title_sentiment_polarity": 0.2,
                        "timedelta": 731,
                        "article_year": 2013,
                        "article_month": 1,
                        "article_day": 1
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