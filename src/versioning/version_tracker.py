import hashlib
import pandas as pd
import json
import mlflow
from mlflow.data.pandas_dataset import from_pandas

import os
import subprocess
import sys
from datetime import datetime
from typing import Optional
from versioning.version_control import VersionControl
from typing import Dict, Any, Tuple
from utils.logger import get_logger
from mlflow.entities import RunInfo as run_info
import joblib

logger = get_logger(__name__)

class VersionTracker:
    """
    Clase para rastrear y versionar DataFrames con DVC automáticamente,
    y registrar sus hashes y metadatos en MLflow si hay un run activo.
    """

    def __init__(
        self,
        version_control: VersionControl,
        output_dir: str = "data/processed/",
        metadata_path: str = "registry/data_versions.json",
    ):
        """
        Inicializa el rastreador de versiones de datos.
        Args:
            version_control (VersionControl): Instancia de VersionControl para usar DVC.
            output_dir (str): Carpeta donde se guardarán las versiones de DataFrames.
            metadata_path (str): Ruta al archivo donde se registrarán los cambios.
        """
        self.vc = version_control
        self.output_dir = output_dir
        self.metadata_path = metadata_path

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w") as f:
                json.dump({}, f, indent=4)

    def _set_train_data(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        self.train_dataset = from_pandas(train_data, name="training_data")

    def _set_test_data(self, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_test = X_test
        test_data = self.X_test.copy()
        test_data["target"] = y_test
        self.test_dataset = from_pandas(test_data, targets="target", name="testing_data")
        
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Calcula un hash SHA1 único para el contenido del DataFrame."""
        df_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha1(df_bytes).hexdigest()[:10]

    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        """Guarda el DataFrame como CSV."""
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        return path

    def _run_dvc_command(self, *args):
        """Ejecuta comandos DVC con subprocess."""
        result = subprocess.run([sys.executable, "-m", "dvc", *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error ejecutando DVC: {result.stderr.strip()}")
        return result.stdout.strip()

    def _update_metadata(
        self,
        name: str,
        version_hash: str,
        file_path: str,
        description: str,
        commit_message: str,
    ):
        """Actualiza el registro local de versiones (JSON)."""
        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        if name not in data:
            data[name] = []

        data[name].append(
            {
                "version": version_hash,
                "file": file_path,
                "description": description,
                "commit_message": commit_message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        with open(self.metadata_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Metadata updated for dataset '{name}' with version '{version_hash}' in {self.metadata_path}.")

    def track_dvc_change(
        self,
        df: pd.DataFrame,
        description: str = "",
        name: str = "online_news_raw_cleaned",
        save_as: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Registra una versión del DataFrame usando DVC y, opcionalmente, en MLflow.
        """
        version_hash = self._hash_dataframe(df)
        filename = save_as or f"{name}.csv"
        file_path = self._save_dataframe(df, filename)
        # DVC
        self._run_dvc_command("add", file_path)
        commit_msg = commit_message or description
        self._run_dvc_command("commit", file_path)
        self._run_dvc_command("push", "-r", self.vc.dvc_remote_name)
        logger.info(f"✅ DataFrame versioned with DVC for change: {description}")

        # Actualizar registro local
        self._update_metadata(name, version_hash, file_path, description, commit_msg)

        return version_hash
    
    def track_mlflow_change(
            self,
            model_name: str,
            best_estimator,
            best_params: Dict[str, Any],
            metrics: Dict[str, float]
    ):
        """
        Registra información de un modelo en MLflow y VersionTracker.
        """
        with mlflow.start_run(run_name=model_name):
            # Input de los datasets utilizados
            mlflow.log_input(self.train_dataset, context="training")
            mlflow.log_input(self.test_dataset, context="testing")

            # Log de parámetros e hiperparámetros
            mlflow.log_params(best_params)

            # Log de métricas de evaluación
            mlflow.log_metrics(metrics)

            # Guardar el modelo completo
            mlflow.sklearn.log_model(best_estimator, name=model_name.replace(" ", "_"), 
                                     input_example=self.X_test, registered_model_name=model_name)

            run_id = mlflow.active_run().info.run_id

        logger.info(f"🧠 Modelo '{model_name}' registrado en MLflow y VersionTracker (run_id={run_id})")

    def get_best_tracked_model(self, metric: str = "R2") -> Optional[run_info]:
        """
        Obtiene el run_id del mejor modelo registrado en MLflow para un nombre dado.
        """
        client = mlflow.tracking.MlflowClient(tracking_uri=self.vc.mlflow_tracking_uri)
        experiment = client.get_experiment_by_name(self.vc.mlflow_experiment)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"]
        )
        if runs:
            best_run = runs[0]
            logger.info(f"Mejor modelo '{best_run.info.run_name}' encontrado: run_id={best_run.info.run_id}")
            return best_run.info
        logger.error(f"No se encontró ningún modelo con esas metricas.")
        return None

    def save_best_tracked_model(self, metric: str = "R2", save_path: str = "models/"):
        """
        Guarda el mejor modelo rastreado en MLflow localmente.
        """
        best_run = self.get_best_tracked_model(metric=metric)
        if best_run:
            model_uri = f"models:/{best_run.run_name}/latest"
            best_model = mlflow.sklearn.load_model(model_uri)
            if save_path == "models/":
                save_path = "models/{}{}.pkl".format(best_run.run_name.replace(" ", "_"), best_run.run_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(best_model, save_path)
            logger.info(f"Modelo '{best_run.run_name}' guardado en '{save_path}'.")
        else:
            logger.error(f"No se pudo guardar el modelo porque no se encontró ningún run_id.")