"""
DriftDetector
=============

Detecta cambios estadísticos en las distribuciones de features entre
un dataset baseline y un dataset de monitoreo.

Métodos estadísticos implementados:
1. Kolmogorov-Smirnov Test: Para features numéricas continuas
2. Chi-Squared Test: Para features categóricas/binarias
3. Population Stability Index (PSI): Métrica de estabilidad poblacional
4. Jensen-Shannon Divergence: Divergencia entre distribuciones

Cada método retorna:
- Estadístico/valor de drift
- P-value (cuando aplica)
- Clasificación de severidad

Uso:
    detector = DriftDetector(df_baseline, df_monitoring)
    results = detector.detect_all_drift()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Detecta drift estadístico comparando baseline vs monitoring data.
    
    Parámetros
    ----------
    df_baseline : pd.DataFrame
        Dataset de referencia (datos históricos/entrenamiento)
    df_monitoring : pd.DataFrame
        Dataset de monitoreo (datos nuevos/producción)
    """
    
    def __init__(self, df_baseline: pd.DataFrame, df_monitoring: pd.DataFrame):
        self.df_baseline = df_baseline.copy()
        self.df_monitoring = df_monitoring.copy()
        
        # Verificar columnas comunes
        self.common_cols = list(set(df_baseline.columns) & set(df_monitoring.columns))
        
        if not self.common_cols:
            raise ValueError("No hay columnas comunes entre baseline y monitoring datasets")
        
        logger.info(
            f"🔍 DriftDetector inicializado: "
            f"{len(df_baseline)} baseline, {len(df_monitoring)} monitoring, "
            f"{len(self.common_cols)} columnas comunes"
        )
    
    def kolmogorov_smirnov_test(
        self,
        column: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test de Kolmogorov-Smirnov para features numéricas continuas.
        
        Mide la máxima diferencia entre las funciones de distribución
        acumulativa (CDF) de dos muestras.
        
        Parámetros
        ----------
        column : str
            Nombre de la columna a analizar
        alpha : float
            Nivel de significancia (default: 0.05)
        
        Returns
        -------
        Dict
            - statistic: KS statistic (0-1, mayor = más drift)
            - p_value: P-value del test
            - drift_detected: bool (True si p < alpha)
            - severity: 'no_drift', 'low', 'medium', 'high'
        """
        if column not in self.common_cols:
            raise ValueError(f"Columna '{column}' no está en ambos datasets")
        
        # Eliminar NaNs para el test
        baseline_values = self.df_baseline[column].dropna()
        monitoring_values = self.df_monitoring[column].dropna()
        
        if len(baseline_values) == 0 or len(monitoring_values) == 0:
            logger.warning(f"⚠️  '{column}' no tiene suficientes valores no-NaN")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'severity': 'unknown',
                'method': 'KS-Test'
            }
        
        # Ejecutar test KS
        ks_stat, p_value = stats.ks_2samp(baseline_values, monitoring_values)
        drift_detected = p_value < alpha
        
        # Clasificar severidad basada en KS statistic
        if ks_stat < 0.1:
            severity = 'no_drift'
        elif ks_stat < 0.2:
            severity = 'low'
        elif ks_stat < 0.3:
            severity = 'medium'
        else:
            severity = 'high'
        
        logger.info(
            f"📊 KS-Test '{column}': statistic={ks_stat:.4f}, "
            f"p-value={p_value:.4f}, severity={severity}"
        )
        
        return {
            'statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'severity': severity,
            'method': 'KS-Test'
        }
    
    def chi_squared_test(
        self,
        column: str,
        alpha: float = 0.05,
        n_bins: int = 10
    ) -> Dict:
        """
        Test Chi-Cuadrado para features categóricas o numéricas discretas.
        
        Compara las frecuencias observadas vs esperadas en bins/categorías.
        
        Parámetros
        ----------
        column : str
            Nombre de la columna a analizar
        alpha : float
            Nivel de significancia
        n_bins : int
            Número de bins para discretizar features numéricas
        
        Returns
        -------
        Dict
            - statistic: Chi-squared statistic
            - p_value: P-value del test
            - drift_detected: bool
            - severity: clasificación de severidad
        """
        if column not in self.common_cols:
            raise ValueError(f"Columna '{column}' no está en ambos datasets")
        
        baseline_values = self.df_baseline[column].dropna()
        monitoring_values = self.df_monitoring[column].dropna()
        
        if len(baseline_values) == 0 or len(monitoring_values) == 0:
            logger.warning(f"⚠️  '{column}' no tiene suficientes valores")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'severity': 'unknown',
                'method': 'Chi-Squared'
            }
        
        # Determinar si es categórica o numérica
        is_numeric = pd.api.types.is_numeric_dtype(baseline_values)
        
        if is_numeric:
            # Discretizar en bins
            bins = np.linspace(
                min(baseline_values.min(), monitoring_values.min()),
                max(baseline_values.max(), monitoring_values.max()),
                n_bins + 1
            )
            baseline_binned = pd.cut(baseline_values, bins=bins, labels=False, duplicates='drop')
            monitoring_binned = pd.cut(monitoring_values, bins=bins, labels=False, duplicates='drop')
        else:
            baseline_binned = baseline_values
            monitoring_binned = monitoring_values
        
        # Contar frecuencias
        baseline_counts = pd.Series(baseline_binned).value_counts().sort_index()
        monitoring_counts = pd.Series(monitoring_binned).value_counts().sort_index()
        
        # Alinear índices
        all_categories = sorted(set(baseline_counts.index) | set(monitoring_counts.index))
        baseline_freq = np.array([baseline_counts.get(cat, 0) for cat in all_categories])
        monitoring_freq = np.array([monitoring_counts.get(cat, 0) for cat in all_categories])
        
        # Evitar división por cero
        if baseline_freq.sum() == 0 or monitoring_freq.sum() == 0:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'drift_detected': False,
                'severity': 'unknown',
                'method': 'Chi-Squared'
            }
        
        # Normalizar a proporciones
        baseline_prop = baseline_freq / baseline_freq.sum()
        monitoring_prop = monitoring_freq / monitoring_freq.sum()
        
        # Test Chi-cuadrado
        chi2_stat, p_value = stats.chisquare(
            f_obs=monitoring_freq + 1,  # +1 para evitar ceros
            f_exp=(baseline_prop * monitoring_freq.sum()) + 1
        )
        
        drift_detected = p_value < alpha
        
        # Clasificar severidad
        if chi2_stat < 5:
            severity = 'no_drift'
        elif chi2_stat < 15:
            severity = 'low'
        elif chi2_stat < 30:
            severity = 'medium'
        else:
            severity = 'high'
        
        logger.info(
            f"📊 Chi-Squared '{column}': statistic={chi2_stat:.4f}, "
            f"p-value={p_value:.4f}, severity={severity}"
        )
        
        return {
            'statistic': chi2_stat,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'severity': severity,
            'method': 'Chi-Squared'
        }
    
    def population_stability_index(
        self,
        column: str,
        n_bins: int = 10
    ) -> Dict:
        """
        Population Stability Index (PSI) - Métrica estándar en la industria.
        
        PSI mide el cambio en la distribución de una variable.
        Valores de referencia:
        - PSI < 0.1: Sin cambio significativo
        - 0.1 ≤ PSI < 0.25: Cambio moderado
        - PSI ≥ 0.25: Cambio significativo
        
        Parámetros
        ----------
        column : str
            Nombre de la columna
        n_bins : int
            Número de bins para discretizar
        
        Returns
        -------
        Dict
            - psi_value: Valor PSI calculado
            - severity: clasificación según umbrales estándar
        """
        if column not in self.common_cols:
            raise ValueError(f"Columna '{column}' no está en ambos datasets")
        
        baseline_values = self.df_baseline[column].dropna()
        monitoring_values = self.df_monitoring[column].dropna()
        
        if len(baseline_values) == 0 or len(monitoring_values) == 0:
            logger.warning(f"⚠️  '{column}' no tiene suficientes valores")
            return {
                'psi_value': np.nan,
                'severity': 'unknown',
                'method': 'PSI'
            }
        
        # Crear bins basados en baseline
        if pd.api.types.is_numeric_dtype(baseline_values):
            bins = np.percentile(baseline_values, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)  # Eliminar duplicados
            
            if len(bins) < 2:
                logger.warning(f"⚠️  '{column}' no tiene suficiente variabilidad para PSI")
                return {
                    'psi_value': np.nan,
                    'severity': 'unknown',
                    'method': 'PSI'
                }
            
            baseline_binned = pd.cut(baseline_values, bins=bins, labels=False, duplicates='drop', include_lowest=True)
            monitoring_binned = pd.cut(monitoring_values, bins=bins, labels=False, duplicates='drop', include_lowest=True)
        else:
            baseline_binned = baseline_values
            monitoring_binned = monitoring_values
        
        # Calcular distribuciones
        baseline_dist = pd.Series(baseline_binned).value_counts(normalize=True, dropna=True).sort_index()
        monitoring_dist = pd.Series(monitoring_binned).value_counts(normalize=True, dropna=True).sort_index()
        
        # Alinear índices
        all_bins = sorted(set(baseline_dist.index) | set(monitoring_dist.index))
        baseline_pct = np.array([baseline_dist.get(b, 0) for b in all_bins])
        monitoring_pct = np.array([monitoring_dist.get(b, 0) for b in all_bins])
        
        # Evitar log(0) añadiendo pequeño epsilon
        epsilon = 1e-10
        baseline_pct = np.maximum(baseline_pct, epsilon)
        monitoring_pct = np.maximum(monitoring_pct, epsilon)
        
        # Calcular PSI
        psi = np.sum((monitoring_pct - baseline_pct) * np.log(monitoring_pct / baseline_pct))
        
        # Clasificar severidad
        if psi < 0.1:
            severity = 'no_drift'
        elif psi < 0.25:
            severity = 'medium'
        else:
            severity = 'high'
        
        logger.info(f"📊 PSI '{column}': {psi:.4f}, severity={severity}")
        
        return {
            'psi_value': psi,
            'severity': severity,
            'method': 'PSI'
        }
    
    def jensen_shannon_divergence(
        self,
        column: str,
        n_bins: int = 20
    ) -> Dict:
        """
        Jensen-Shannon Divergence - Métrica simétrica de diferencia entre distribuciones.
        
        Rango: 0 (idénticas) a 1 (completamente diferentes)
        
        Parámetros
        ----------
        column : str
            Nombre de la columna
        n_bins : int
            Número de bins para discretizar
        
        Returns
        -------
        Dict
            - js_divergence: Valor de divergencia (0-1)
            - severity: clasificación
        """
        if column not in self.common_cols:
            raise ValueError(f"Columna '{column}' no está en ambos datasets")
        
        baseline_values = self.df_baseline[column].dropna()
        monitoring_values = self.df_monitoring[column].dropna()
        
        if len(baseline_values) == 0 or len(monitoring_values) == 0:
            return {
                'js_divergence': np.nan,
                'severity': 'unknown',
                'method': 'JS-Divergence'
            }
        
        # Crear histogramas con los mismos bins
        if pd.api.types.is_numeric_dtype(baseline_values):
            bins = np.linspace(
                min(baseline_values.min(), monitoring_values.min()),
                max(baseline_values.max(), monitoring_values.max()),
                n_bins + 1
            )
            
            baseline_hist, _ = np.histogram(baseline_values, bins=bins, density=True)
            monitoring_hist, _ = np.histogram(monitoring_values, bins=bins, density=True)
        else:
            # Para categóricas
            all_categories = sorted(set(baseline_values) | set(monitoring_values))
            baseline_counts = pd.Series(baseline_values).value_counts()
            monitoring_counts = pd.Series(monitoring_values).value_counts()
            
            baseline_hist = np.array([baseline_counts.get(c, 0) for c in all_categories])
            monitoring_hist = np.array([monitoring_counts.get(c, 0) for c in all_categories])
        
        # Normalizar
        baseline_hist = baseline_hist / (baseline_hist.sum() + 1e-10)
        monitoring_hist = monitoring_hist / (monitoring_hist.sum() + 1e-10)
        
        # Calcular JS divergence
        js_div = jensenshannon(baseline_hist, monitoring_hist)
        
        # Clasificar severidad
        if js_div < 0.1:
            severity = 'no_drift'
        elif js_div < 0.2:
            severity = 'low'
        elif js_div < 0.3:
            severity = 'medium'
        else:
            severity = 'high'
        
        logger.info(f"📊 JS-Divergence '{column}': {js_div:.4f}, severity={severity}")
        
        return {
            'js_divergence': js_div,
            'severity': severity,
            'method': 'JS-Divergence'
        }
    
    def detect_all_drift(
        self,
        columns: Optional[List[str]] = None,
        methods: List[str] = ['ks', 'psi', 'js']
    ) -> pd.DataFrame:
        """
        Detecta drift en múltiples columnas usando múltiples métodos.
        
        Parámetros
        ----------
        columns : List[str], opcional
            Columnas a analizar. Si None, analiza todas las comunes numéricas
        methods : List[str]
            Métodos a usar: 'ks', 'chi2', 'psi', 'js'
        
        Returns
        -------
        pd.DataFrame
            Resultados de detección por columna y método
        """
        if columns is None:
            columns = self.df_baseline.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c in self.common_cols]
        
        logger.info(f"🔍 Analizando {len(columns)} columnas con métodos: {methods}")
        
        results = []
        
        for col in columns:
            col_results = {'column': col}
            
            # KS Test
            if 'ks' in methods:
                ks_result = self.kolmogorov_smirnov_test(col)
                col_results['ks_statistic'] = ks_result['statistic']
                col_results['ks_p_value'] = ks_result['p_value']
                col_results['ks_severity'] = ks_result['severity']
            
            # Chi-Squared
            if 'chi2' in methods:
                chi2_result = self.chi_squared_test(col)
                col_results['chi2_statistic'] = chi2_result['statistic']
                col_results['chi2_p_value'] = chi2_result['p_value']
                col_results['chi2_severity'] = chi2_result['severity']
            
            # PSI
            if 'psi' in methods:
                psi_result = self.population_stability_index(col)
                col_results['psi_value'] = psi_result['psi_value']
                col_results['psi_severity'] = psi_result['severity']
            
            # JS Divergence
            if 'js' in methods:
                js_result = self.jensen_shannon_divergence(col)
                col_results['js_divergence'] = js_result['js_divergence']
                col_results['js_severity'] = js_result['severity']
            
            results.append(col_results)
        
        df_results = pd.DataFrame(results)
        logger.info(f"✅ Detección de drift completada para {len(columns)} columnas")
        
        return df_results
