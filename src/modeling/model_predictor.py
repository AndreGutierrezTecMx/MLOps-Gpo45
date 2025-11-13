"""
Model Predictor
--------------
Carga el modelo desde MLflow y realiza predicciones.
"""

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
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

        # Intentar cargar el modelo, pero no fallar si no se puede
        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo al inicializar: {e}")
            logger.info(
                "La API iniciará pero el modelo no estará disponible hasta que se cargue correctamente")

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

            # Si no es HTTP, cargar directamente desde el tracking URI actual
            if not self.mlflow_tracking_uri.startswith("http"):
                self.model = mlflow.sklearn.load_model(model_uri)
                return

            # Para HTTP, intentar cargar desde filesystem local primero para evitar problemas con el servidor MLflow
            project_root = Path(__file__).resolve().parents[2]
            local_mlruns = project_root / "mlruns"

            # Si no existe mlruns local, cargar desde servidor HTTP
            if not local_mlruns.exists():
                self.model = mlflow.sklearn.load_model(model_uri)
                return

            # Intentar cargar desde filesystem local
            try:
                local_tracking_uri = str(local_mlruns.resolve().as_uri())
                original_uri = mlflow.get_tracking_uri()
                mlflow.set_tracking_uri(local_tracking_uri)
                logger.info(
                    f"Intentando cargar modelo desde filesystem local: {local_tracking_uri}")
                self.model = mlflow.sklearn.load_model(model_uri)
                mlflow.set_tracking_uri(original_uri)  # Restaurar URI original
                logger.info("Modelo cargado exitosamente desde filesystem local")
                return
            except Exception as local_error:
                logger.warning(
                    f"Error cargando desde filesystem local: {local_error}")
                logger.info("Intentando cargar desde servidor HTTP...")
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            # Fallback: cargar desde servidor HTTP
            self.model = mlflow.sklearn.load_model(model_uri)

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
