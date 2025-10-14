import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level=logging.INFO,
    log_to_file: bool = True,
    log_dir: str = "logs",
    log_filename: str = None
):
    """
    Configura el sistema de logging global para el proyecto MLOps.
    Llamar una sola vez al inicio del programa o pipeline.
    """
    # Evitar configurar múltiples veces
    if logging.getLogger().hasHandlers():
        return

    # Crear formato de salida
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configurar el manejador de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handlers = [console_handler]

    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_filename = log_filename or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_path = Path(log_dir) / log_filename

        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger con el nombre especificado.
    Usar dentro de cada módulo: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
