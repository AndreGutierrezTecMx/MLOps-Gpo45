"""
Model Predictor
--------------
Carga el modelo desde MLflow y realiza predicciones.
"""

import mlflow
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """
    Clase para cargar modelos desde MLflow y realizar predicciones.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://127.0.0.1:5000",
        mlflow_experiment: str = "Modeling_Experiment",
        metric: str = "R2"
    ):
        """
        Inicializa el predictor con configuración de MLflow.

        Args:
            mlflow_tracking_uri: URI del servidor MLflow
            mlflow_experiment: Nombre del experimento
            metric: Métrica para seleccionar el mejor modelo
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment = mlflow_experiment
        self.metric = metric
        self.model = None
        self.model_info = None

        # Configurar MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        self._load_model()

    def _load_model(self) -> None:
        """
        Carga el mejor modelo desde MLflow según la métrica especificada.

        Raises:
            ValueError: Si el experimento no existe o no hay modelos disponibles
        """
        try:
            client = mlflow.tracking.MlflowClient(
                tracking_uri=self.mlflow_tracking_uri)
            experiment = client.get_experiment_by_name(self.mlflow_experiment)

            if experiment is None:
                raise ValueError(
                    f"Experimento '{self.mlflow_experiment}' no encontrado en MLflow")

            # Buscar el mejor run por la métrica especificada
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{self.metric} DESC"],
                max_results=1
            )

            if not runs:
                raise ValueError(
                    f"No se encontraron modelos en el experimento '{self.mlflow_experiment}'")

            best_run = runs[0]

            # Guardar información del modelo
            self.model_info = {
                'run_id': best_run.info.run_id,
                'run_name': best_run.info.run_name,
                'metric': self.metric,
                'metric_value': best_run.data.metrics.get(self.metric, 'N/A')
            }

            # Cargar el modelo
            model_uri = f"runs:/{best_run.info.run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)

            logger.info(
                f"Modelo cargado: {self.model_info['run_name']} "
                f"(run_id={self.model_info['run_id'][:8]}..., "
                f"{self.metric}={self.model_info['metric_value']})"
            )

        except Exception as e:
            logger.exception(f"Error cargando modelo desde MLflow: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones sobre un DataFrame.

        Args:
            data: DataFrame con las features necesarias para el modelo

        Returns:
            Arreglo de numpy con las predicciones

        Raises:
            RuntimeError: Si el modelo no está cargado
        """
        if self.model is None:
            raise RuntimeError(
                "Modelo no cargado. Llama a load_model() primero.")

        try:
            predictions = self.model.predict(data)
            logger.info(f"Predicción realizada para {len(data)} registro(s)")
            return np.array(predictions)
        except Exception as e:
            logger.exception(f"Error realizando predicción: {e}")
            raise
