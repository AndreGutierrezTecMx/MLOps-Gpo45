"""
DriftEvaluator
==============

Evalúa el impacto del data drift en el rendimiento del modelo de ML.

Funcionalidades:
1. Carga modelo desde MLflow Registry
2. Evalúa métricas en datos con drift
3. Compara con baseline metrics
4. Calcula degradación de performance
5. Genera reporte de impacto

Uso:
    evaluator = DriftEvaluator(
        model_name="HistGradientBoosting (Poisson)",
        mlflow_tracking_uri="http://127.0.0.1:5000"
    )
    impact = evaluator.evaluate_drift_impact(
        X_drift, y_drift, baseline_metrics
    )
"""

import pandas as pd
import numpy as np
import mlflow
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftEvaluator:
    """
    Evalúa el impacto del drift en el rendimiento del modelo.
    
    Parámetros
    ----------
    model_name : str
        Nombre del modelo en MLflow
    mlflow_tracking_uri : str
        URI del servidor MLflow
    mlflow_experiment : str, opcional
        Nombre del experimento MLflow
    """
    
    def __init__(
        self,
        model_name: str,
        mlflow_tracking_uri: str = "http://127.0.0.1:5000",
        mlflow_experiment: str = "Modeling_Experiment"
    ):
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment = mlflow_experiment
        
        # Configurar MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        logger.info(
            f"📊 DriftEvaluator inicializado para modelo: {model_name}"
        )
    
    def load_best_model(self) -> Tuple[object, Dict]:
        """
        Carga el mejor modelo desde MLflow.
        
        Returns
        -------
        Tuple[object, Dict]
            - Modelo cargado (sklearn pipeline)
            - Métricas baseline del modelo
        """
        try:
            mlflow.set_experiment(self.mlflow_experiment)
            
            # Buscar el mejor run por R2
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment)
            
            if experiment is None:
                raise ValueError(f"Experimento '{self.mlflow_experiment}' no encontrado")
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.model_name = '{self.model_name}'",
                order_by=["metrics.R2 DESC"],
                max_results=1
            )
            
            if runs.empty:
                raise ValueError(f"No se encontraron runs para modelo '{self.model_name}'")
            
            best_run = runs.iloc[0]
            run_id = best_run['run_id']
            
            # Extraer métricas baseline
            baseline_metrics = {
                'MAE': best_run['metrics.MAE'],
                'RMSE': best_run['metrics.RMSE'],
                'R2': best_run['metrics.R2']
            }
            
            # Cargar modelo
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(
                f"✅ Modelo cargado - Run ID: {run_id[:8]}... | "
                f"Baseline: MAE={baseline_metrics['MAE']:.2f}, "
                f"R2={baseline_metrics['R2']:.4f}"
            )
            
            return model, baseline_metrics
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo desde MLflow: {e}")
            raise
    
    def evaluate_metrics(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evalúa métricas del modelo en un dataset.
        
        Parámetros
        ----------
        model : object
            Modelo entrenado (sklearn pipeline)
        X : pd.DataFrame
            Features
        y : pd.Series
            Target verdadero
        
        Returns
        -------
        Dict[str, float]
            Métricas: MAE, RMSE, R2
        """
        try:
            # Predicciones
            y_pred = model.predict(X)
            
            # Calcular métricas
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            logger.info(
                f"📈 Métricas calculadas: MAE={mae:.2f}, "
                f"RMSE={rmse:.2f}, R2={r2:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluando métricas: {e}")
            raise
    
    def calculate_degradation(
        self,
        baseline_metrics: Dict[str, float],
        drift_metrics: Dict[str, float]
    ) -> Dict:
        """
        Calcula degradación de performance comparando baseline vs drift.
        
        Parámetros
        ----------
        baseline_metrics : Dict
            Métricas en datos baseline/test originales
        drift_metrics : Dict
            Métricas en datos con drift
        
        Returns
        -------
        Dict
            Reporte de degradación con porcentajes y severidad
        """
        degradation = {}
        
        for metric in ['MAE', 'RMSE', 'R2']:
            base_val = baseline_metrics[metric]
            drift_val = drift_metrics[metric]
            
            # Para MAE y RMSE, menor es mejor (aumento es degradación)
            # Para R2, mayor es mejor (disminución es degradación)
            if metric in ['MAE', 'RMSE']:
                change_pct = ((drift_val - base_val) / base_val) * 100
                is_degraded = drift_val > base_val
            else:  # R2
                change_pct = ((drift_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
                is_degraded = drift_val < base_val
            
            degradation[metric] = {
                'baseline': base_val,
                'drift': drift_val,
                'change_%': change_pct,
                'is_degraded': is_degraded
            }
        
        # Calcular score global de degradación (0-100)
        # Ponderación: MAE (30%), RMSE (30%), R2 (40%)
        mae_degrade = abs(degradation['MAE']['change_%']) if degradation['MAE']['is_degraded'] else 0
        rmse_degrade = abs(degradation['RMSE']['change_%']) if degradation['RMSE']['is_degraded'] else 0
        r2_degrade = abs(degradation['R2']['change_%']) if degradation['R2']['is_degraded'] else 0
        
        degradation_score = (
            mae_degrade * 0.30 +
            rmse_degrade * 0.30 +
            r2_degrade * 0.40
        )
        
        # Clasificar severidad
        if degradation_score < 5:
            severity = 'no_degradation'
        elif degradation_score < 15:
            severity = 'low'
        elif degradation_score < 30:
            severity = 'medium'
        else:
            severity = 'high'
        
        degradation['summary'] = {
            'degradation_score': degradation_score,
            'severity': severity,
            'any_metric_degraded': any(
                degradation[m]['is_degraded'] for m in ['MAE', 'RMSE', 'R2']
            )
        }
        
        logger.info(
            f"📉 Degradación: score={degradation_score:.1f}/100, "
            f"severity={severity}"
        )
        
        return degradation
    
    def evaluate_drift_impact(
        self,
        X_drift: pd.DataFrame,
        y_drift: pd.Series,
        baseline_metrics: Optional[Dict] = None,
        model: Optional[object] = None
    ) -> Dict:
        """
        Evalúa el impacto completo del drift en el modelo.
        
        Parámetros
        ----------
        X_drift : pd.DataFrame
            Features con drift
        y_drift : pd.Series
            Target verdadero
        baseline_metrics : Dict, opcional
            Métricas baseline. Si None, se cargan de MLflow
        model : object, opcional
            Modelo. Si None, se carga de MLflow
        
        Returns
        -------
        Dict
            Reporte completo de impacto:
            - baseline_metrics
            - drift_metrics
            - degradation
            - recommendations
        """
        logger.info("🔍 Evaluando impacto del drift en el modelo...")
        
        # Cargar modelo y métricas si no se proporcionan
        if model is None or baseline_metrics is None:
            loaded_model, loaded_metrics = self.load_best_model()
            model = model or loaded_model
            baseline_metrics = baseline_metrics or loaded_metrics
        
        # Evaluar en datos con drift
        drift_metrics = self.evaluate_metrics(model, X_drift, y_drift)
        
        # Calcular degradación
        degradation = self.calculate_degradation(baseline_metrics, drift_metrics)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(degradation)
        
        impact_report = {
            'model_name': self.model_name,
            'baseline_metrics': baseline_metrics,
            'drift_metrics': drift_metrics,
            'degradation': degradation,
            'recommendations': recommendations
        }
        
        logger.info("✅ Evaluación de impacto completada")
        
        return impact_report
    
    def _generate_recommendations(self, degradation: Dict) -> List[str]:
        """
        Genera recomendaciones de acción basadas en la degradación.
        
        Parámetros
        ----------
        degradation : Dict
            Reporte de degradación
        
        Returns
        -------
        List[str]
            Lista de recomendaciones
        """
        recommendations = []
        severity = degradation['summary']['severity']
        score = degradation['summary']['degradation_score']
        
        if severity == 'no_degradation':
            recommendations.append("✅ No se detectó degradación significativa")
            recommendations.append("🔄 Continuar monitoreo regular")
        
        elif severity == 'low':
            recommendations.append("⚠️  Degradación leve detectada")
            recommendations.append("🔍 Incrementar frecuencia de monitoreo")
            recommendations.append("📊 Analizar features con mayor drift")
        
        elif severity == 'medium':
            recommendations.append("⚠️  Degradación moderada detectada")
            recommendations.append("🔄 Programar reentrenamiento en próxima ventana")
            recommendations.append("📊 Revisar pipeline de features")
            recommendations.append("🎯 Considerar feature engineering adicional")
        
        else:  # high
            recommendations.append("🚨 Degradación severa detectada")
            recommendations.append("🔴 ACCIÓN INMEDIATA: Reentrenar modelo urgentemente")
            recommendations.append("🔍 Investigar causa raíz del drift")
            recommendations.append("🛠️  Revisar pipeline completo de datos")
            recommendations.append("📈 Considerar cambio de arquitectura/algoritmo")
        
        # Recomendaciones específicas por métrica
        if degradation['R2']['is_degraded'] and abs(degradation['R2']['change_%']) > 20:
            recommendations.append(
                f"📉 R2 cayó {abs(degradation['R2']['change_%']):.1f}% - "
                "Revisar features predictivas principales"
            )
        
        if degradation['MAE']['is_degraded'] and abs(degradation['MAE']['change_%']) > 15:
            recommendations.append(
                f"📈 MAE aumentó {abs(degradation['MAE']['change_%']):.1f}% - "
                "Verificar outliers y escalado de datos"
            )
        
        return recommendations
    
    def generate_impact_summary(self, impact_report: Dict) -> str:
        """
        Genera resumen legible del reporte de impacto.
        
        Parámetros
        ----------
        impact_report : Dict
            Reporte de impacto completo
        
        Returns
        -------
        str
            Resumen formateado
        """
        degradation = impact_report['degradation']
        baseline = impact_report['baseline_metrics']
        drift = impact_report['drift_metrics']
        
        summary = f"""
{'='*60}
REPORTE DE IMPACTO DE DATA DRIFT
{'='*60}

Modelo: {impact_report['model_name']}
Severidad: {degradation['summary']['severity'].upper()}
Score de Degradación: {degradation['summary']['degradation_score']:.1f}/100

{'='*60}
MÉTRICAS
{'='*60}

{'Métrica':<10} {'Baseline':<12} {'Drift':<12} {'Cambio %':<12}
{'-'*60}
MAE        {baseline['MAE']:<12.2f} {drift['MAE']:<12.2f} {degradation['MAE']['change_%']:>+11.1f}%
RMSE       {baseline['RMSE']:<12.2f} {drift['RMSE']:<12.2f} {degradation['RMSE']['change_%']:>+11.1f}%
R2         {baseline['R2']:<12.4f} {drift['R2']:<12.4f} {degradation['R2']['change_%']:>+11.1f}%

{'='*60}
RECOMENDACIONES
{'='*60}
"""
        for i, rec in enumerate(impact_report['recommendations'], 1):
            summary += f"\n{i}. {rec}"
        
        summary += f"\n\n{'='*60}\n"
        
        return summary
