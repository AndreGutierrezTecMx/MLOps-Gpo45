"""
DriftAlertSystem
================

Sistema de alertas configurable para monitoreo de data drift y
degradación de modelos en producción.

Características:
1. Umbrales configurables por métrica
2. Niveles de severidad: INFO, WARNING, CRITICAL
3. Historial de alertas
4. Acciones recomendadas
5. Export/import de alertas

Uso:
    alert_system = DriftAlertSystem(thresholds={
        'psi': {'warning': 0.1, 'critical': 0.25},
        'degradation_score': {'warning': 15.0, 'critical': 30.0}
    })
    
    alerts = alert_system.check_drift_alerts(drift_results)
    alert_system.export_alert_history('alerts.json')
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Literal
from datetime import datetime
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftAlertSystem:
    """
    Sistema de alertas para monitoreo de drift y performance.
    
    Parámetros
    ----------
    thresholds : Dict, opcional
        Umbrales personalizados por métrica
    alert_history_path : str, opcional
        Ruta para guardar historial de alertas
    """
    
    # Umbrales por defecto (basados en estándares de la industria)
    DEFAULT_THRESHOLDS = {
        'psi': {
            'warning': 0.10,
            'critical': 0.25
        },
        'ks_statistic': {
            'warning': 0.15,
            'critical': 0.30
        },
        'js_divergence': {
            'warning': 0.15,
            'critical': 0.30
        },
        'degradation_score': {
            'warning': 15.0,
            'critical': 30.0
        },
        'r2_change': {
            'warning': -10.0,  # -10% de cambio
            'critical': -20.0  # -20% de cambio
        },
        'mae_change': {
            'warning': 15.0,   # +15% de aumento
            'critical': 30.0   # +30% de aumento
        }
    }
    
    def __init__(
        self,
        thresholds: Optional[Dict] = None,
        alert_history_path: Optional[str] = None
    ):
        # Combinar umbrales por defecto con personalizados
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
        
        self.alert_history_path = alert_history_path
        self.alert_history: List[Dict] = []
        
        # Cargar historial si existe
        if alert_history_path and Path(alert_history_path).exists():
            self._load_alert_history()
        
        logger.info(
            f"🚨 DriftAlertSystem inicializado con "
            f"{len(self.thresholds)} umbrales configurados"
        )
    
    def check_drift_alerts(
        self,
        drift_results: pd.DataFrame,
        timestamp: Optional[str] = None
    ) -> List[Dict]:
        """
        Verifica alertas basadas en resultados de detección de drift.
        
        Parámetros
        ----------
        drift_results : pd.DataFrame
            Resultados de DriftDetector.detect_all_drift()
        timestamp : str, opcional
            Timestamp de la verificación. Si None, usa datetime actual
        
        Returns
        -------
        List[Dict]
            Lista de alertas generadas
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        alerts = []
        
        logger.info(f"🔍 Verificando alertas para {len(drift_results)} features...")
        
        for _, row in drift_results.iterrows():
            column = row['column']
            
            # Verificar PSI
            if 'psi_value' in row and not pd.isna(row['psi_value']):
                psi_alerts = self._check_threshold(
                    value=row['psi_value'],
                    metric='psi',
                    feature=column,
                    timestamp=timestamp
                )
                alerts.extend(psi_alerts)
            
            # Verificar KS statistic
            if 'ks_statistic' in row and not pd.isna(row['ks_statistic']):
                ks_alerts = self._check_threshold(
                    value=row['ks_statistic'],
                    metric='ks_statistic',
                    feature=column,
                    timestamp=timestamp
                )
                alerts.extend(ks_alerts)
            
            # Verificar JS divergence
            if 'js_divergence' in row and not pd.isna(row['js_divergence']):
                js_alerts = self._check_threshold(
                    value=row['js_divergence'],
                    metric='js_divergence',
                    feature=column,
                    timestamp=timestamp
                )
                alerts.extend(js_alerts)
        
        # Agregar al historial
        self.alert_history.extend(alerts)
        
        logger.info(f"✅ Generadas {len(alerts)} alertas")
        
        return alerts
    
    def check_model_performance_alerts(
        self,
        impact_report: Dict,
        timestamp: Optional[str] = None
    ) -> List[Dict]:
        """
        Verifica alertas basadas en degradación de performance del modelo.
        
        Parámetros
        ----------
        impact_report : Dict
            Reporte de DriftEvaluator.evaluate_drift_impact()
        timestamp : str, opcional
            Timestamp de la verificación
        
        Returns
        -------
        List[Dict]
            Lista de alertas de performance
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        alerts = []
        degradation = impact_report['degradation']
        
        logger.info("🔍 Verificando alertas de performance del modelo...")
        
        # Verificar degradation score global
        degrade_score = degradation['summary']['degradation_score']
        score_alerts = self._check_threshold(
            value=degrade_score,
            metric='degradation_score',
            feature='modelo_completo',
            timestamp=timestamp,
            additional_info={
                'model_name': impact_report['model_name'],
                'severity': degradation['summary']['severity']
            }
        )
        alerts.extend(score_alerts)
        
        # Verificar cambios en R2
        if degradation['R2']['is_degraded']:
            r2_change = degradation['R2']['change_%']
            r2_alerts = self._check_threshold(
                value=r2_change,
                metric='r2_change',
                feature='R2',
                timestamp=timestamp,
                additional_info={
                    'baseline': degradation['R2']['baseline'],
                    'current': degradation['R2']['drift']
                },
                is_negative_metric=True  # Para R2, valores negativos son malos
            )
            alerts.extend(r2_alerts)
        
        # Verificar cambios en MAE
        if degradation['MAE']['is_degraded']:
            mae_change = degradation['MAE']['change_%']
            mae_alerts = self._check_threshold(
                value=mae_change,
                metric='mae_change',
                feature='MAE',
                timestamp=timestamp,
                additional_info={
                    'baseline': degradation['MAE']['baseline'],
                    'current': degradation['MAE']['drift']
                }
            )
            alerts.extend(mae_alerts)
        
        # Agregar al historial
        self.alert_history.extend(alerts)
        
        logger.info(f"✅ Generadas {len(alerts)} alertas de performance")
        
        return alerts
    
    def _check_threshold(
        self,
        value: float,
        metric: str,
        feature: str,
        timestamp: str,
        additional_info: Optional[Dict] = None,
        is_negative_metric: bool = False
    ) -> List[Dict]:
        """
        Verifica si un valor excede umbrales configurados.
        
        Parámetros
        ----------
        value : float
            Valor a verificar
        metric : str
            Nombre de la métrica
        feature : str
            Nombre del feature/modelo
        timestamp : str
            Timestamp
        additional_info : Dict, opcional
            Información adicional para la alerta
        is_negative_metric : bool
            Si True, valores negativos activan alerta (ej: R2)
        
        Returns
        -------
        List[Dict]
            Lista de alertas (puede ser vacía)
        """
        if metric not in self.thresholds:
            return []
        
        thresholds = self.thresholds[metric]
        alerts = []
        
        # Determinar severidad
        severity = None
        
        if is_negative_metric:
            # Para métricas donde negativo es malo (ej: cambio en R2)
            if value <= thresholds['critical']:
                severity = 'CRITICAL'
            elif value <= thresholds['warning']:
                severity = 'WARNING'
        else:
            # Para métricas donde positivo/alto es malo
            if value >= thresholds['critical']:
                severity = 'CRITICAL'
            elif value >= thresholds['warning']:
                severity = 'WARNING'
        
        # Crear alerta si hay severidad
        if severity:
            alert = {
                'timestamp': timestamp,
                'severity': severity,
                'metric': metric,
                'feature': feature,
                'value': value,
                'threshold_warning': thresholds['warning'],
                'threshold_critical': thresholds['critical'],
                'message': self._generate_alert_message(
                    severity, metric, feature, value
                )
            }
            
            if additional_info:
                alert['additional_info'] = additional_info
            
            alerts.append(alert)
            
            logger.warning(
                f"🚨 {severity} - {metric} en '{feature}': "
                f"{value:.4f} (umbral: {thresholds[severity.lower()]})"
            )
        
        return alerts
    
    def _generate_alert_message(
        self,
        severity: str,
        metric: str,
        feature: str,
        value: float
    ) -> str:
        """
        Genera mensaje descriptivo para la alerta.
        
        Returns
        -------
        str
            Mensaje de alerta
        """
        metric_names = {
            'psi': 'Population Stability Index',
            'ks_statistic': 'Kolmogorov-Smirnov statistic',
            'js_divergence': 'Jensen-Shannon divergence',
            'degradation_score': 'Model degradation score',
            'r2_change': 'R2 change',
            'mae_change': 'MAE change'
        }
        
        metric_display = metric_names.get(metric, metric)
        
        if severity == 'CRITICAL':
            return (
                f"🔴 CRÍTICO: {metric_display} de '{feature}' alcanzó {value:.4f}. "
                f"Se requiere acción inmediata."
            )
        else:  # WARNING
            return (
                f"⚠️  ADVERTENCIA: {metric_display} de '{feature}' alcanzó {value:.4f}. "
                f"Monitorear de cerca."
            )
    
    def get_active_alerts(
        self,
        severity: Optional[Literal['WARNING', 'CRITICAL']] = None,
        hours: int = 24
    ) -> List[Dict]:
        """
        Obtiene alertas activas de las últimas N horas.
        
        Parámetros
        ----------
        severity : str, opcional
            Filtrar por severidad específica
        hours : int
            Ventana de tiempo en horas
        
        Returns
        -------
        List[Dict]
            Alertas activas
        """
        if not self.alert_history:
            return []
        
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (hours * 3600)
        
        active_alerts = []
        
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            
            if alert_time >= cutoff_time:
                if severity is None or alert['severity'] == severity:
                    active_alerts.append(alert)
        
        logger.info(
            f"📋 {len(active_alerts)} alertas activas en últimas {hours}h"
        )
        
        return active_alerts
    
    def export_alert_history(self, filepath: str) -> None:
        """
        Exporta historial de alertas a JSON.
        
        Parámetros
        ----------
        filepath : str
            Ruta del archivo de salida
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.alert_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Historial exportado: {filepath} ({len(self.alert_history)} alertas)")
            
        except Exception as e:
            logger.error(f"❌ Error exportando historial: {e}")
            raise
    
    def _load_alert_history(self) -> None:
        """Carga historial de alertas desde archivo JSON."""
        try:
            with open(self.alert_history_path, 'r', encoding='utf-8') as f:
                self.alert_history = json.load(f)
            
            logger.info(
                f"📥 Historial cargado: {len(self.alert_history)} alertas"
            )
            
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar historial: {e}")
            self.alert_history = []
    
    def generate_alert_summary(
        self,
        alerts: List[Dict]
    ) -> str:
        """
        Genera resumen legible de alertas.
        
        Parámetros
        ----------
        alerts : List[Dict]
            Lista de alertas
        
        Returns
        -------
        str
            Resumen formateado
        """
        if not alerts:
            return "✅ No se generaron alertas"
        
        critical_count = sum(1 for a in alerts if a['severity'] == 'CRITICAL')
        warning_count = sum(1 for a in alerts if a['severity'] == 'WARNING')
        
        summary = f"""
{'='*70}
RESUMEN DE ALERTAS
{'='*70}

Total de Alertas: {len(alerts)}
  🔴 Críticas: {critical_count}
  ⚠️  Advertencias: {warning_count}

{'='*70}
DETALLE DE ALERTAS
{'='*70}
"""
        
        for i, alert in enumerate(alerts, 1):
            summary += f"\n{i}. {alert['message']}"
            summary += f"\n   Feature: {alert['feature']}"
            summary += f"\n   Valor: {alert['value']:.4f}"
            summary += f"\n   Timestamp: {alert['timestamp']}\n"
        
        summary += f"\n{'='*70}\n"
        
        return summary
