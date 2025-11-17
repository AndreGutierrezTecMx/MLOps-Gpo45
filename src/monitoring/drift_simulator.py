"""
DriftSimulator
==============

Genera conjuntos de datos con diferentes tipos de drift simulado para
evaluar la robustez y detección de cambios en modelos de ML.

Tipos de drift implementados:
1. Mean Shift: Desplazamiento en la media de features numéricas
2. Variance Change: Cambio en la varianza de features
3. Missing Values: Introducción de valores faltantes
4. Seasonal Drift: Cambios estacionales/temporales
5. Category Drift: Cambios en distribuciones categóricas

Uso:
    simulator = DriftSimulator(df_baseline, random_state=42)
    df_drift = simulator.simulate_mean_shift(
        columns=['n_tokens_content', 'num_hrefs'],
        shift_percentage=0.15
    )
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftSimulator:
    """
    Simula diferentes tipos de data drift en un DataFrame baseline.
    
    Parámetros
    ----------
    df_baseline : pd.DataFrame
        Dataset baseline (limpio) usado como referencia
    random_state : int
        Semilla para reproducibilidad
    """
    
    def __init__(self, df_baseline: pd.DataFrame, random_state: int = 42):
        self.df_baseline = df_baseline.copy()
        self.random_state = random_state
        np.random.seed(random_state)
        logger.info(f"🎲 DriftSimulator inicializado con {len(df_baseline)} filas baseline")
    
    def simulate_mean_shift(
        self,
        columns: List[str],
        shift_percentage: float = 0.15,
        direction: Literal['increase', 'decrease', 'both'] = 'both'
    ) -> pd.DataFrame:
        """
        Simula desplazamiento en la media de columnas numéricas.
        
        Parámetros
        ----------
        columns : List[str]
            Columnas numéricas donde aplicar el shift
        shift_percentage : float
            Porcentaje de cambio en la media (0.15 = 15%)
        direction : str
            Dirección del shift: 'increase', 'decrease', 'both'
        
        Returns
        -------
        pd.DataFrame
            Dataset con mean shift aplicado
        """
        df_drift = self.df_baseline.copy()
        
        for col in columns:
            if col not in df_drift.columns:
                logger.warning(f"⚠️  Columna '{col}' no encontrada, omitiendo...")
                continue
                
            if not pd.api.types.is_numeric_dtype(df_drift[col]):
                logger.warning(f"⚠️  Columna '{col}' no es numérica, omitiendo...")
                continue
            
            original_mean = df_drift[col].mean()
            
            if direction == 'increase':
                multiplier = 1 + shift_percentage
            elif direction == 'decrease':
                multiplier = 1 - shift_percentage
            else:  # both - aleatoriamente aumenta o disminuye
                multiplier = 1 + (shift_percentage if np.random.rand() > 0.5 else -shift_percentage)
            
            df_drift[col] = df_drift[col] * multiplier
            new_mean = df_drift[col].mean()
            
            logger.info(
                f"📊 '{col}': media {original_mean:.2f} → {new_mean:.2f} "
                f"({((new_mean/original_mean - 1) * 100):+.1f}%)"
            )
        
        logger.info(f"✅ Mean shift aplicado en {len(columns)} columnas")
        return df_drift
    
    def simulate_variance_change(
        self,
        columns: List[str],
        variance_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Simula cambio en la varianza de columnas numéricas.
        
        Parámetros
        ----------
        columns : List[str]
            Columnas donde cambiar la varianza
        variance_multiplier : float
            Factor multiplicador de la desviación estándar
            (1.5 = aumenta varianza 50%)
        
        Returns
        -------
        pd.DataFrame
            Dataset con varianza modificada
        """
        df_drift = self.df_baseline.copy()
        
        for col in columns:
            if col not in df_drift.columns or not pd.api.types.is_numeric_dtype(df_drift[col]):
                logger.warning(f"⚠️  Columna '{col}' no válida, omitiendo...")
                continue
            
            original_std = df_drift[col].std()
            original_mean = df_drift[col].mean()
            
            # Aumentar/disminuir dispersión manteniendo la media
            df_drift[col] = original_mean + (df_drift[col] - original_mean) * variance_multiplier
            
            new_std = df_drift[col].std()
            logger.info(
                f"📈 '{col}': std {original_std:.2f} → {new_std:.2f} "
                f"({((new_std/original_std - 1) * 100):+.1f}%)"
            )
        
        logger.info(f"✅ Cambio de varianza aplicado en {len(columns)} columnas")
        return df_drift
    
    def simulate_missing_values(
        self,
        columns: List[str],
        missing_percentage: float = 0.10
    ) -> pd.DataFrame:
        """
        Introduce valores faltantes aleatoriamente en columnas.
        
        Parámetros
        ----------
        columns : List[str]
            Columnas donde introducir NaNs
        missing_percentage : float
            Porcentaje de valores a eliminar (0.10 = 10%)
        
        Returns
        -------
        pd.DataFrame
            Dataset con valores faltantes introducidos
        """
        df_drift = self.df_baseline.copy()
        n_rows = len(df_drift)
        
        for col in columns:
            if col not in df_drift.columns:
                logger.warning(f"⚠️  Columna '{col}' no encontrada, omitiendo...")
                continue
            
            # Seleccionar índices aleatorios para hacer NaN
            n_missing = int(n_rows * missing_percentage)
            missing_indices = np.random.choice(df_drift.index, size=n_missing, replace=False)
            
            original_missing = df_drift[col].isna().sum()
            df_drift.loc[missing_indices, col] = np.nan
            new_missing = df_drift[col].isna().sum()
            
            logger.info(
                f"🕳️  '{col}': {original_missing} → {new_missing} NaNs "
                f"({(new_missing/n_rows * 100):.1f}%)"
            )
        
        logger.info(f"✅ Valores faltantes introducidos en {len(columns)} columnas")
        return df_drift
    
    def simulate_seasonal_drift(
        self,
        columns: List[str],
        amplitude: float = 0.20,
        period: int = 7
    ) -> pd.DataFrame:
        """
        Simula patrón estacional/cíclico en columnas numéricas.
        
        Parámetros
        ----------
        columns : List[str]
            Columnas donde aplicar patrón estacional
        amplitude : float
            Amplitud de la variación estacional (0.20 = ±20%)
        period : int
            Período del ciclo (días/observaciones)
        
        Returns
        -------
        pd.DataFrame
            Dataset con drift estacional
        """
        df_drift = self.df_baseline.copy()
        n_rows = len(df_drift)
        
        # Generar patrón sinusoidal
        time_steps = np.arange(n_rows)
        seasonal_pattern = amplitude * np.sin(2 * np.pi * time_steps / period)
        
        for col in columns:
            if col not in df_drift.columns or not pd.api.types.is_numeric_dtype(df_drift[col]):
                logger.warning(f"⚠️  Columna '{col}' no válida, omitiendo...")
                continue
            
            original_mean = df_drift[col].mean()
            # Aplicar patrón multiplicativo
            df_drift[col] = df_drift[col] * (1 + seasonal_pattern)
            
            logger.info(f"🌊 '{col}': patrón estacional aplicado (periodo={period}, amplitud={amplitude})")
        
        logger.info(f"✅ Drift estacional aplicado en {len(columns)} columnas")
        return df_drift
    
    def simulate_category_drift(
        self,
        columns: List[str],
        shift_probability: float = 0.30
    ) -> pd.DataFrame:
        """
        Simula cambios en distribución de columnas categóricas/binarias.
        
        Parámetros
        ----------
        columns : List[str]
            Columnas categóricas/binarias donde aplicar drift
        shift_probability : float
            Probabilidad de cambiar cada valor (0.30 = 30%)
        
        Returns
        -------
        pd.DataFrame
            Dataset con drift en categorías
        """
        df_drift = self.df_baseline.copy()
        
        for col in columns:
            if col not in df_drift.columns:
                logger.warning(f"⚠️  Columna '{col}' no encontrada, omitiendo...")
                continue
            
            unique_vals = df_drift[col].unique()
            n_unique = len(unique_vals)
            
            if n_unique <= 1:
                logger.warning(f"⚠️  '{col}' tiene ≤1 valor único, omitiendo...")
                continue
            
            # Máscara de filas a modificar
            modify_mask = np.random.rand(len(df_drift)) < shift_probability
            n_modified = modify_mask.sum()
            
            # Para binarias (0/1), invertir valores
            if n_unique == 2 and set(unique_vals).issubset({0, 1}):
                df_drift.loc[modify_mask, col] = 1 - df_drift.loc[modify_mask, col]
                logger.info(f"🔄 '{col}': {n_modified} valores binarios invertidos")
            else:
                # Para categóricas, asignar valores aleatorios
                df_drift.loc[modify_mask, col] = np.random.choice(
                    unique_vals, size=n_modified
                )
                logger.info(f"🔄 '{col}': {n_modified} valores categóricos modificados")
        
        logger.info(f"✅ Category drift aplicado en {len(columns)} columnas")
        return df_drift
    
    def simulate_combined_drift(
        self,
        mean_shift_cols: Optional[List[str]] = None,
        variance_cols: Optional[List[str]] = None,
        missing_cols: Optional[List[str]] = None,
        seasonal_cols: Optional[List[str]] = None,
        category_cols: Optional[List[str]] = None,
        intensity: Literal['mild', 'moderate', 'severe'] = 'moderate'
    ) -> pd.DataFrame:
        """
        Combina múltiples tipos de drift en un solo dataset.
        
        Parámetros
        ----------
        mean_shift_cols : List[str], opcional
            Columnas para mean shift
        variance_cols : List[str], opcional
            Columnas para cambio de varianza
        missing_cols : List[str], opcional
            Columnas para valores faltantes
        seasonal_cols : List[str], opcional
            Columnas para drift estacional
        category_cols : List[str], opcional
            Columnas para category drift
        intensity : str
            Intensidad del drift: 'mild', 'moderate', 'severe'
        
        Returns
        -------
        pd.DataFrame
            Dataset con drift combinado
        """
        # Parámetros según intensidad
        params = {
            'mild': {
                'shift_pct': 0.08,
                'variance_mult': 1.2,
                'missing_pct': 0.05,
                'seasonal_amp': 0.10,
                'category_prob': 0.15
            },
            'moderate': {
                'shift_pct': 0.15,
                'variance_mult': 1.5,
                'missing_pct': 0.10,
                'seasonal_amp': 0.20,
                'category_prob': 0.30
            },
            'severe': {
                'shift_pct': 0.25,
                'variance_mult': 2.0,
                'missing_pct': 0.20,
                'seasonal_amp': 0.35,
                'category_prob': 0.50
            }
        }[intensity]
        
        logger.info(f"🎯 Generando drift combinado con intensidad: {intensity}")
        
        df_drift = self.df_baseline.copy()
        
        # Aplicar cada tipo de drift si se especificaron columnas
        if mean_shift_cols:
            df_drift = DriftSimulator(df_drift, self.random_state).simulate_mean_shift(
                columns=mean_shift_cols,
                shift_percentage=params['shift_pct']
            )
        
        if variance_cols:
            df_drift = DriftSimulator(df_drift, self.random_state).simulate_variance_change(
                columns=variance_cols,
                variance_multiplier=params['variance_mult']
            )
        
        if missing_cols:
            df_drift = DriftSimulator(df_drift, self.random_state).simulate_missing_values(
                columns=missing_cols,
                missing_percentage=params['missing_pct']
            )
        
        if seasonal_cols:
            df_drift = DriftSimulator(df_drift, self.random_state).simulate_seasonal_drift(
                columns=seasonal_cols,
                amplitude=params['seasonal_amp']
            )
        
        if category_cols:
            df_drift = DriftSimulator(df_drift, self.random_state).simulate_category_drift(
                columns=category_cols,
                shift_probability=params['category_prob']
            )
        
        logger.info(f"✅ Drift combinado ({intensity}) generado exitosamente")
        return df_drift
    
    def generate_drift_report(self, df_drift: pd.DataFrame) -> Dict:
        """
        Genera reporte comparativo entre baseline y drift dataset.
        
        Parámetros
        ----------
        df_drift : pd.DataFrame
            Dataset con drift aplicado
        
        Returns
        -------
        Dict
            Reporte con estadísticas comparativas
        """
        report = {}
        numeric_cols = self.df_baseline.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_drift.columns:
                base_mean = self.df_baseline[col].mean()
                drift_mean = df_drift[col].mean()
                base_std = self.df_baseline[col].std()
                drift_std = df_drift[col].std()
                
                report[col] = {
                    'original_mean': base_mean,
                    'drift_mean': drift_mean,
                    'mean_change_%': ((drift_mean - base_mean) / base_mean * 100) if base_mean != 0 else 0,
                    'original_std': base_std,
                    'drift_std': drift_std,
                    'std_change_%': ((drift_std - base_std) / base_std * 100) if base_std != 0 else 0,
                }
        
        return report
