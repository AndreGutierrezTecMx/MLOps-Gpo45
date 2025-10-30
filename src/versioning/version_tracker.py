import hashlib
import pandas as pd
import json
import mlflow
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional
from versioning.version_control import VersionControl
from utils.logger import get_logger

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

    def track_change(
        self,
        df: pd.DataFrame,
        description: str = "",
        name: str = "online_news_raw_cleaned",
        save_as: Optional[str] = None,
        commit_message: Optional[str] = None,
        log_to_mlflow: bool = True,
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

        # MLflow
        if log_to_mlflow:
            with mlflow.start_run(run_name=f"dataset_{name}_{version_hash}"):
                mlflow.log_param("dataset_name", name)
                mlflow.log_param("version_hash", version_hash)
                mlflow.log_param("description", description)
                mlflow.log_artifact(file_path, artifact_path="datasets")

        # Actualizar registro local
        self._update_metadata(name, version_hash, file_path, description, commit_msg)

        return version_hash