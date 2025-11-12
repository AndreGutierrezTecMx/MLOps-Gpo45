"""
Módulo de Monitoreo de Data Drift
==================================

Este módulo proporciona herramientas para simular, detectar y evaluar
el impacto del data drift en modelos de machine learning.

Componentes principales:
- DriftSimulator: Genera conjuntos de datos con drift simulado
- DriftDetector: Detecta cambios estadísticos en distribuciones
- DriftEvaluator: Evalúa el impacto del drift en el rendimiento del modelo
- DriftAlertSystem: Sistema de alertas basado en umbrales

Autor: MLOps-GPO45 Team
Fecha: Noviembre 2025
"""

from src.monitoring.drift_simulator import DriftSimulator
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.drift_evaluator import DriftEvaluator
from src.monitoring.drift_alert_system import DriftAlertSystem

__version__ = "1.0.0"
__all__ = [
    "DriftSimulator",
    "DriftDetector", 
    "DriftEvaluator",
    "DriftAlertSystem",
]