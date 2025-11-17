"""
Ejemplo de Uso: Sistema de Detección de Data Drift
====================================================

Este script demuestra cómo usar el sistema completo de detección
y evaluación de data drift.

Flujo completo:
1. Cargar datos baseline (cleaned)
2. Simular diferentes escenarios de drift
3. Detectar drift estadístico
4. Evaluar impacto en modelo
5. Generar alertas
6. Exportar resultados

Autor: MLOps-GPO45 Team
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Importar clases del sistema de drift
from src.monitoring.drift_simulator import DriftSimulator
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.drift_evaluator import DriftEvaluator
from src.monitoring.drift_alert_system import DriftAlertSystem
from src.utils.logger import get_logger, setup_logging

# Configurar logging
setup_logging()
logger = get_logger(__name__)


def main():
    """
    Ejecuta el flujo completo de detección y evaluación de drift.
    """
    logger.info("="*70)
    logger.info("INICIANDO SISTEMA DE DETECCIÓN DE DATA DRIFT")
    logger.info("="*70)
    
    # ========================================
    # 1. CARGAR DATOS BASELINE
    # ========================================
    logger.info("\n📂 Paso 1: Cargando datos baseline...")
    
    data_path = "data/processed/online_news_cleaned.csv"
    
    try:
        df_baseline = pd.read_csv(data_path)
        logger.info(f"✅ Datos cargados: {df_baseline.shape}")
    except FileNotFoundError:
        logger.error(f"❌ Archivo no encontrado: {data_path}")
        return
    
    # Separar features y target
    target_col = 'shares'
    exclude_cols = ['url', 'article_title', 'url_cleaned', 'mixed_type_col', target_col]
    
    X_baseline = df_baseline.drop(columns=[c for c in exclude_cols if c in df_baseline.columns])
    y_baseline = df_baseline[target_col] if target_col in df_baseline.columns else None
    
    logger.info(f"Features: {X_baseline.shape[1]}, Target: {target_col}")
    
    # ========================================
    # 2. SIMULAR ESCENARIOS DE DRIFT
    # ========================================
    logger.info("\n🎲 Paso 2: Simulando escenarios de drift...")
    
    simulator = DriftSimulator(df_baseline, random_state=42)
    
    # Escenario 1: Drift Leve
    logger.info("\n--- Escenario 1: Drift LEVE ---")
    df_mild_drift = simulator.simulate_combined_drift(
        mean_shift_cols=['n_tokens_content', 'num_hrefs'],
        variance_cols=['kw_avg_avg'],
        missing_cols=['num_videos'],
        seasonal_cols=['LDA_00'],
        category_cols=['data_channel_is_tech'],
        intensity='mild'
    )
    
    # Escenario 2: Drift Moderado
    logger.info("\n--- Escenario 2: Drift MODERADO ---")
    df_moderate_drift = simulator.simulate_combined_drift(
        mean_shift_cols=['n_tokens_content', 'num_hrefs', 'num_imgs'],
        variance_cols=['kw_avg_avg', 'global_subjectivity'],
        missing_cols=['num_videos', 'num_keywords'],
        seasonal_cols=['LDA_00', 'LDA_01'],
        category_cols=['data_channel_is_tech', 'is_weekend'],
        intensity='moderate'
    )
    
    # Escenario 3: Drift Severo
    logger.info("\n--- Escenario 3: Drift SEVERO ---")
    df_severe_drift = simulator.simulate_combined_drift(
        mean_shift_cols=['n_tokens_content', 'num_hrefs', 'num_imgs', 'average_token_length'],
        variance_cols=['kw_avg_avg', 'global_subjectivity', 'global_sentiment_polarity'],
        missing_cols=['num_videos', 'num_keywords', 'num_self_hrefs'],
        seasonal_cols=['LDA_00', 'LDA_01', 'LDA_02'],
        category_cols=['data_channel_is_tech', 'is_weekend', 'data_channel_is_lifestyle'],
        intensity='severe'
    )
    
    scenarios = {
        'mild': df_mild_drift,
        'moderate': df_moderate_drift,
        'severe': df_severe_drift
    }
    
    # ========================================
    # 3. DETECTAR DRIFT ESTADÍSTICO
    # ========================================
    logger.info("\n🔍 Paso 3: Detectando drift estadístico...")
    
    drift_results = {}
    
    for scenario_name, df_drift in scenarios.items():
        logger.info(f"\n--- Analizando escenario: {scenario_name.upper()} ---")
        
        detector = DriftDetector(df_baseline, df_drift)
        
        # Seleccionar columnas clave para analizar
        key_columns = [
            'n_tokens_content', 'num_hrefs', 'num_imgs', 'num_videos',
            'kw_avg_avg', 'LDA_00', 'global_subjectivity',
            'data_channel_is_tech', 'is_weekend'
        ]
        
        available_columns = [c for c in key_columns if c in detector.common_cols]
        
        results = detector.detect_all_drift(
            columns=available_columns,
            methods=['ks', 'psi', 'js']
        )
        
        drift_results[scenario_name] = results
        
        # Mostrar resumen
        high_drift = results[results['psi_severity'] == 'high']
        logger.info(f"Features con drift alto: {len(high_drift)}")
    
    # ========================================
    # 4. EVALUAR IMPACTO EN MODELO
    # ========================================
    logger.info("\n📊 Paso 4: Evaluando impacto en modelo...")
    
    evaluator = DriftEvaluator(
        model_name="HistGradientBoosting (Poisson)",
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment="Modeling_Experiment"
    )
    
    impact_reports = {}
    
    for scenario_name, df_drift in scenarios.items():
        logger.info(f"\n--- Evaluando escenario: {scenario_name.upper()} ---")
        
        X_drift = df_drift.drop(columns=[c for c in exclude_cols if c in df_drift.columns])
        y_drift = df_drift[target_col] if target_col in df_drift.columns else None
        
        try:
            impact = evaluator.evaluate_drift_impact(X_drift, y_drift)
            impact_reports[scenario_name] = impact
            
            # Mostrar resumen
            print(evaluator.generate_impact_summary(impact))
            
        except Exception as e:
            logger.error(f"❌ Error evaluando {scenario_name}: {e}")
    
    # ========================================
    # 5. GENERAR ALERTAS
    # ========================================
    logger.info("\n🚨 Paso 5: Generando alertas...")
    
    alert_system = DriftAlertSystem()
    
    all_alerts = {}
    
    for scenario_name in scenarios.keys():
        logger.info(f"\n--- Alertas para escenario: {scenario_name.upper()} ---")
        
        # Alertas de drift
        drift_alerts = alert_system.check_drift_alerts(
            drift_results[scenario_name]
        )
        
        # Alertas de performance
        if scenario_name in impact_reports:
            perf_alerts = alert_system.check_model_performance_alerts(
                impact_reports[scenario_name]
            )
        else:
            perf_alerts = []
        
        all_alerts[scenario_name] = {
            'drift': drift_alerts,
            'performance': perf_alerts
        }
        
        # Mostrar resumen
        combined_alerts = drift_alerts + perf_alerts
        print(alert_system.generate_alert_summary(combined_alerts))
    
    # ========================================
    # 6. EXPORTAR RESULTADOS
    # ========================================
    logger.info("\n💾 Paso 6: Exportando resultados...")
    
    output_dir = Path("outputs/drift_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exportar resultados de drift
    for scenario_name, results in drift_results.items():
        results.to_csv(
            output_dir / f"drift_results_{scenario_name}.csv",
            index=False
        )
        logger.info(f"✅ Exportado: drift_results_{scenario_name}.csv")
    
    # Exportar reportes de impacto
    import json
    for scenario_name, report in impact_reports.items():
        with open(output_dir / f"impact_report_{scenario_name}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✅ Exportado: impact_report_{scenario_name}.json")
    
    # Exportar historial de alertas
    alert_system.export_alert_history(
        str(output_dir / "alert_history.json")
    )
    
    # ========================================
    # RESUMEN FINAL
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("RESUMEN FINAL")
    logger.info("="*70)
    
    for scenario_name in scenarios.keys():
        logger.info(f"\n🎯 Escenario: {scenario_name.upper()}")
        
        # Resumen de drift
        results = drift_results[scenario_name]
        high_drift = len(results[results['psi_severity'] == 'high'])
        logger.info(f"  - Features con drift alto: {high_drift}/{len(results)}")
        
        # Resumen de impacto
        if scenario_name in impact_reports:
            impact = impact_reports[scenario_name]
            severity = impact['degradation']['summary']['severity']
            score = impact['degradation']['summary']['degradation_score']
            logger.info(f"  - Degradación: {severity} (score: {score:.1f}/100)")
        
        # Resumen de alertas
        if scenario_name in all_alerts:
            drift_count = len(all_alerts[scenario_name]['drift'])
            perf_count = len(all_alerts[scenario_name]['performance'])
            logger.info(f"  - Alertas: {drift_count} drift + {perf_count} performance")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESO COMPLETADO EXITOSAMENTE")
    logger.info("="*70)
    logger.info(f"\n📁 Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
