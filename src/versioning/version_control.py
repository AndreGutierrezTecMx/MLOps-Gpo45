import subprocess
from pathlib import Path
import mlflow
import psutil
import time
from utils.logger import get_logger
from constants.dvc_remote_type_enums import DvcRemoteType
from utils.secrets_manager import SecretsManager
import os
import sys
import random

logger = get_logger(__name__)

class VersionControl:
    """
    Controla el versionamiento de datasets (DVC) y modelos (MLflow),
    y configura automáticamente el remote de DVC según el tipo especificado.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str = "http://127.0.0.1:5000",
        mlflow_port: int = 5000,
        mlflow_experiment: str = "default",
        dvc_remote_type: DvcRemoteType = DvcRemoteType.LOCAL,
        dvc_remote_name: str = "myremote",
        dvc_remote_path: str = "../../dvc_remote"):
        """
        Inicializa el control de versiones con DVC y MLflow.

        Args:
            mlflow_tracking_uri (str): URI de MLflow.
            mlflow_experiment (str): Nombre del experimento MLflow.
            dvc_remote_type (str): Tipo de remoto DVC ('local', 'gdrive', 'azure', 's3', etc.).
            dvc_remote_path (str): Ruta o URL del almacenamiento remoto.
            dvc_remote_name (str): Nombre del remoto en DVC.
            secrets_manager (SecretsManager, optional): Para obtener credenciales seguras.
        """
        self.project_root = Path(__file__).resolve().parents[3]
        os.chdir(self.project_root)
        self.dvc_remote_type = dvc_remote_type
        self.dvc_remote_path = dvc_remote_path
        self.dvc_remote_name = dvc_remote_name
        self.secrets_manager = SecretsManager()
        self.mlflow_port = mlflow_port

        # Inicializar DVC en el proyecto si no está ya inicializado
        self._init_dvc()

        # Configurar DVC
        self._setup_dvc_remote()

        # Iniciar MLflow UI en segundo plano
        self._init_mlflow_ui()

        # Configurar MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        try:
            mlflow.set_experiment(mlflow_experiment)
            logger.info(f"✅ MLflow conectado a experimento: {mlflow_experiment}")
        except Exception as e:
            if "deleted experiment" in str(e).lower():
                new_name = f"{mlflow_experiment}_{random.randint(1000, 9999)}"
                logger.warning(f"⚠️ El experimento '{mlflow_experiment}' estaba eliminado. Creando nuevo: '{new_name}'")
                mlflow.set_experiment(new_name)
            else:
                logger.exception(f"❌ Error configurando experimento MLflow: {e}")
                raise
        logger.info(f"✅ MLflow conectado a: {mlflow_tracking_uri}")


    def _setup_dvc_remote(self):
        """
        Detecta el tipo de almacenamiento y configura el remote DVC.
        """
        try:
            # Verificar si el remoto ya existe
            remotes = subprocess.run(
                [sys.executable, "-m", "dvc", "remote", "list"],
                capture_output=True,
                text=True
            )
            if self.dvc_remote_name in remotes.stdout:
                logger.info(f"🔁 DVC remote '{self.dvc_remote_name}' ya configurado.")
                return

            # Crear remoto según tipo
            if self.dvc_remote_type == DvcRemoteType.LOCAL:
                path = Path(self.dvc_remote_path)
                path.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [sys.executable, "-m", "dvc", "remote", "add", "-d", self.dvc_remote_name, str(path), "-f"],
                    check=True
                )
                logger.info(f"📂 DVC remote local configurado en {path}")

            elif self.dvc_remote_type == DvcRemoteType.GDRIVE:
                drive_folder = self.secrets_manager.get_secret("drive_folder")
                subprocess.run(
                    [sys.executable, "-m", "dvc", "remote", "add", "-d", self.dvc_remote_name, f"gdrive://{drive_folder}", "-f"],
                    check=True
                )
                logger.info("🌐 DVC remote Google Drive configurado")

                # Leer credenciales desde SecretsManager si existe
                if self.secrets_manager:
                    client_id = self.secrets_manager.get_secret("client_secret")
                    client_secret = self.secrets_manager.get_secret("client_id")
                    if client_id and client_secret:
                        subprocess.run(
                            [sys.executable, "-m", "dvc", "remote", "modify", "--local", self.dvc_remote_name, "gdrive_client_id", client_id],
                            check=True
                        )
                        subprocess.run(
                            [sys.executable, "-m", "dvc", "remote", "modify", "--local", self.dvc_remote_name, "gdrive_client_secret", client_secret],
                            check=True,
                        )
                        logger.info("🔑 Credenciales de Google Drive configuradas desde SecretsManager")

            elif self.dvc_remote_type == DvcRemoteType.AZURE:
                container_url = self.secrets_manager.get_secret("container_url")
                subprocess.run(
                    [sys.executable, "-m", "dvc", "remote", "add", "-d", self.dvc_remote_name, f"azure://{container_url}", "-f"],
                    check=True,
                )
                logger.info("☁️ DVC remote Azure configurado")

                if self.secrets_manager:
                    account_name = self.secrets_manager.get_secret("account_name")
                    account_key = self.secrets_manager.get_secret("account_key")
                    if account_name and account_key:
                        subprocess.run(
                            [sys.executable, "-m", "dvc", "remote", "modify", "--local", self.dvc_remote_name, "account_name", account_name],
                            check=True,
                        )
                        subprocess.run(
                            [sys.executable, "-m", "dvc", "remote", "modify", "--local", self.dvc_remote_name, "account_key", account_key],
                            check=True,
                        )
                        logger.info("🔑 Credenciales de Azure configuradas desde SecretsManager")

            elif self.dvc_remote_type == DvcRemoteType.S3:
                bucket_url = self.secrets_manager.get_secret("bucket_url")
                subprocess.run(
                    [sys.executable, "-m", "dvc", "remote", "add", "-d", self.dvc_remote_name, f"s3://{bucket_url}", "-f"],
                    check=True,
                )
                logger.info("🪣 DVC remote AWS S3 configurado")

                if self.secrets_manager:
                    access_key = self.secrets_manager.get_secret("aws_id")
                    secret_key = self.secrets_manager.get_secret("aws_key")
                    if access_key and secret_key:
                        subprocess.run(
                            [sys.executable, "-m", "dvc", "remote", "modify", "--local", self.dvc_remote_name, "access_key_id", access_key],
                            check=True,
                        )
                        subprocess.run(
                            [sys.executable, "-m", "dvc", "remote", "modify", "--local", self.dvc_remote_name, "secret_access_key", secret_key],
                            check=True,
                        )
                        logger.info("🔑 Credenciales de AWS configuradas desde SecretsManager")

            else:
                logger.warning(f"⚠️ Tipo de remoto DVC '{self.dvc_remote_type}' no soportado.")

        except subprocess.CalledProcessError as e:
            logger.exception(f"❌ Error al configurar el remoto DVC: {e}")
            raise

    def _init_dvc(self):
        """
        Inicializa DVC si el repositorio aún no tiene la carpeta `.dvc/`.
        Si Git no está inicializado, también lo inicializa.
        """
        logger.debug(f"📂 Directorio raíz del proyecto: {self.project_root}")

        dvc_dir = self.project_root / ".dvc"

        # Si ya está inicializado, salir
        if dvc_dir.exists() and dvc_dir.is_dir():
            logger.info("📁 DVC ya está inicializado.")
            return

        # Asegurar que Git existe
        git_dir = self.project_root / ".git"
        if not git_dir.exists():
            logger.warning("⚠️ No se detectó repositorio Git. Inicializando Git...")
            try:
                subprocess.run(["git", "init"], check=True, capture_output=True, text=True)
                logger.info("🆕 Git inicializado correctamente.")
            except subprocess.CalledProcessError as e:
                logger.exception(f"❌ Error al inicializar Git: {e.stderr or e.stdout}")
                raise RuntimeError("No se pudo inicializar Git.") from e

        # Confirmar DVC instalado
        try:
            result = subprocess.run(
                [sys.executable, "-m", "dvc", "--version"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ DVC detectado correctamente: {result.stdout.strip()}")
        except Exception as e:
            logger.error("❌ No se pudo verificar DVC. ¿Está correctamente instalado?")
            raise

        # Intentar inicializar DVC
        logger.debug("⚙️ Ejecutando `dvc init`...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "dvc", "init", "-q"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("🆕 DVC inicializado correctamente.")
            elif "already initialized" in (result.stderr or result.stdout).lower():
                logger.warning("⚠️ DVC ya estaba inicializado. Continuando.")
            else:
                logger.error(f"❌ Error en `dvc init`: {result.stderr or result.stdout}")
                raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        except Exception as e:
            logger.exception(f"❌ Falló la inicialización de DVC: {e}")
            raise


    def _init_mlflow_ui(self):
        """
        Inicia el servidor MLflow UI en el puerto especificado.
        Si el puerto está ocupado por otro proceso de MLflow, lo termina primero.
        """
        port = self.mlflow_port
        in_use = False

        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []  # ← aseguramos iterable
                cmd_str = " ".join(cmdline) if isinstance(cmdline, (list, tuple)) else str(cmdline)

                if "mlflow" in cmd_str and str(port) in cmd_str:
                    in_use = True
                    logger.warning(f"⚠️ MLflow ya se está ejecutando (PID {proc.pid}). Terminando instancia previa...")
                    proc.kill()
                    time.sleep(1)

            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue

        if not in_use:
            logger.debug(f"🔍 Puerto {port} libre, iniciando MLflow UI...")

        # Iniciar MLflow UI
        try:
            subprocess.Popen(
                [sys.executable, "-m", "mlflow", "ui", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)
            logger.info(f"🚀 MLflow UI ejecutándose en http://127.0.0.1:{port}")
        except Exception as e:
            logger.exception(f"❌ Error al iniciar MLflow UI: {e}")



