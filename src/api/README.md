# API de Predicción - MLOps-Gpo45

API REST desarrollada con FastAPI para servir el modelo de predicción de shares de artículos.

## Descripción

Esta API permite realizar predicciones sobre el número de shares que recibirá un artículo en redes sociales. El modelo se carga automáticamente desde MLflow al iniciar la aplicación, seleccionando el mejor modelo según la métrica configurada (R2 por defecto).

## Configuración

La API utiliza las siguientes constantes por defecto (definidas en `src/constants/mlflow_config.py`):

- **MLFLOW_TRACKING_URI**: `http://127.0.0.1:5000`
- **MLFLOW_EXPERIMENT**: `Modeling_Experiment`
- **MODEL_METRIC**: `R2`

## Ejecución

Para ejecutar la API, usa uvicorn:

```bash
# Desde la raíz del proyecto
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

O desde el directorio `src/api/`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en:

- **URL**: `http://localhost:8000`
- **Documentación interactiva (Swagger)**: `http://localhost:8000/docs`

## Endpoints

### GET `/`

Endpoint raíz que retorna información básica de la API.

**Respuesta:**

```json
{
  "message": "MLOps-Gpo45 Model API",
  "version": "1.0.0"
}
```

### GET `/model-info`

Obtiene información del modelo cargado actualmente.

**Respuesta:**

```json
{
  "run_id": "abc123...",
  "run_name": "HistGradientBoosting (Poisson)",
  "metric": "R2",
  "metric_value": 0.85
}
```

**Códigos de estado:**

- `200`: Información obtenida exitosamente
- `503`: Modelo no disponible o no cargado

### POST `/predict`

Realiza predicciones sobre el número de shares esperados para uno o más artículos.

**Request Body:**

```json
{
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
      "article_year": 2013
    }
  ]
}
```

**Respuesta:**

```json
{
  "predictions": [3456.78]
}
```

**Códigos de estado:**

- `200`: Predicción realizada exitosamente
- `400`: Error de validación en los datos de entrada
- `500`: Error interno del servidor
- `503`: Modelo no disponible

## Ejemplos de Uso

### Usando cURL

```bash
# Obtener información del modelo
curl http://localhost:8000/model-info

# Realizar una predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "n_tokens_title": 10,
        "n_tokens_content": 500,
        "n_unique_tokens": 0.5,
        "data_channel_is_entertainment": 1,
        "weekday_is_monday": 0
      }
    ]
  }'
```

### Usando Python

```python
import requests

# URL base de la API
BASE_URL = "http://localhost:8000"

# Obtener información del modelo
response = requests.get(f"{BASE_URL}/model-info")
print(response.json())

# Realizar predicción
data = {
    "data": [
        {
            "n_tokens_title": 10,
            "n_tokens_content": 500,
            "n_unique_tokens": 0.5,
            "data_channel_is_entertainment": 1,
            "weekday_is_monday": 0,
            # ... resto de features
        }
    ]
}

response = requests.post(f"{BASE_URL}/predict", json=data)
predictions = response.json()
print(f"Predicción: {predictions['predictions'][0]} shares")
```

## Validación de Datos

La API utiliza Pydantic para validar automáticamente los datos de entrada:

- El campo `data` debe ser una lista de diccionarios
- La lista debe contener al menos un elemento
- Cada diccionario debe contener las features necesarias para el modelo

Si los datos no son válidos, la API retornará un error `400` con detalles sobre qué campos faltan o son incorrectos.
