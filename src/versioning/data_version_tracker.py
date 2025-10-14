import hashlib
import pandas as pd
import json
import logging
import mlflow
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.versioning.version_control import VersionControl

logger = logging.getLogger(__name__)

class DataVersionTracker:
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
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()

    def _load_metadata(self):
        if not self.metadata_path.exists():
            logger.info("🆕 Creando nuevo registro de versiones de datos...")
            return {}
        try:
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.exception(f"❌ Error al cargar metadata: {e}")
            return {}

    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=4)
        logger.debug("📝 Metadata de versiones de datos actualizada.")

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """
        Calcula un hash SHA256 del contenido del DataFrame.
        """
        hash_str = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha256(hash_str).hexdigest()[:12]

    def track_change(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = "",
        save_as: Optional[str] = None,
        commit_message: Optional[str] = None,
        log_to_mlflow: bool = True,
    ) -> str:
        """
        Registra una versión del DataFrame usando DVC y, opcionalmente, en MLflow.
        Args:
            df (pd.DataFrame): DataFrame modificado.
            name (str): Nombre lógico del dataset.
            description (str): Descripción del cambio.
            save_as (str): Nombre del archivo CSV a guardar (por defecto usa el hash).
            commit_message (str): Mensaje del commit DVC.
            log_to_mlflow (bool): Si True, registra info en MLflow.
        Returns:
            str: Hash de versión del DataFrame.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calcular hash del DataFrame
        version_hash = self._hash_dataframe(df)
        filename = save_as or f"data/processed/{name}_{version_hash}.csv"
        output_path = self.output_dir / filename

        # Guardar el DataFrame en CSV
        df.to_csv(output_path, index=False)
        logger.info(f"💾 DataFrame guardado en {output_path}")

        # Registrar dataset en DVC
        commit_msg = commit_message or f"{name} - {description or 'actualización de datos'}"
        dataset_version = self.vc.add_dataset(str(output_path), commit_msg)
        logger.info(f"📦 Dataset versionado en DVC con hash {version_hash}")

        # Registrar metadata local
        entry = {
            "name": name,
            "path": str(output_path),
            "hash": version_hash,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "dvc_version": dataset_version,
        }
        self._metadata[version_hash] = entry
        self._save_metadata()

        # Registrar en MLflow si hay un run activo
        if log_to_mlflow and mlflow.active_run():
            try:
                mlflow.set_tag("dataset_name", name)
                mlflow.set_tag("dataset_hash", version_hash)
                mlflow.log_param("dataset_path", str(output_path))
                mlflow.log_param("dataset_dvc_version", dataset_version)
                if description:
                    mlflow.set_tag("dataset_description", description)
                logger.info(f"🔗 Dataset vinculado al run de MLflow: {version_hash}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo registrar dataset en MLflow: {e}")

        return version_hash

    def get_version(self, version_hash: str) -> Optional[dict]:
        """Devuelve los metadatos de una versión específica."""
        return self._metadata.get(version_hash)

    def list_versions(self):
        """Lista todas las versiones registradas."""
        return list(self._metadata.keys())
