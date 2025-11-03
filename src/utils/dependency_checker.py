import subprocess
import sys
import importlib
import json
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class DependencyChecker:
    """Verifica dependencias a partir de un archivo JSON."""

    @staticmethod
    def check_command(command: str):
        """Ejecuta un comando y muestra su salida."""
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"✅ {command}:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError:
            logger.exception(f"⚠️ Error ejecutando: {command}")

    @staticmethod
    def ensure_dependencies(config_path: str = "configs/dependencies.json"):
        """Verifica e instala dependencias necesarias a partir de un archivo JSON."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo {config_file}")

        with open(config_file, "r") as f:
            dependencies = json.load(f)["dependencies"]

        logger.info("🔍 Verificando dependencias...\n")
        DependencyChecker.check_command("python3 --version")
        DependencyChecker.check_command("git --version")

        for dep in dependencies:
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
