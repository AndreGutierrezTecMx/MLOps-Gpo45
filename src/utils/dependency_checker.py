import subprocess
import sys
import importlib
import json
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class DependencyChecker:
    """Verifica dependencias a partir de un archivo JSON."""

    def __init__(self, config_path: str = "configs/dependencies.json"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo {self.config_path}")
        with open(self.config_path, "r") as f:
            self.dependencies = json.load(f)["dependencies"]

    def check_command(self, command: str):
        """Ejecuta un comando y muestra su salida."""
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"✅ {command}:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError:
            logger.exception(f"⚠️ Error ejecutando: {command}")

    def ensure_dependencies(self):
        """Verifica e instala dependencias necesarias."""
        logger.info("🔍 Verificando dependencias...\n")
        self.check_command("python3 --version")
        self.check_command("git --version")

        for dep in self.dependencies:
            module_name = dep["module"]
            pip_name = dep["pip_name"]

            try:
                importlib.import_module(module_name)
                logger.info(f"✅ {module_name} ya está instalado.")
            except ImportError:
                logger.info(f"📦 Instalando {pip_name}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
                logger.info(f"✅ {pip_name} instalado correctamente.")

                importlib.invalidate_caches()
                try:
                    importlib.import_module(module_name)
                    logger.info(f"✅ {module_name} cargado tras instalación.")
                except ImportError:
                    logger.warning(f"⚠️ {module_name} no se pudo cargar tras instalar. "
                                   f"Reinicia el intérprete si el error persiste.")
