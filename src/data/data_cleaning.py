import pandas as pd
import numpy as np

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from utils.logger import get_logger
from versioning.version_tracker import VersionTracker
import os

logger = get_logger(__name__)

class DataCleaning:
    """Una clase para limpiar y preprocesar dataframes desde datos crudos."""
    
    def __init__(self, dataframe: pd.DataFrame, tracker: VersionTracker):
        """
        Inicializa DataCleaning con un dataframe.
        
        Parámetros
        ----------
        dataframe : pd.DataFrame
            El dataframe crudo a ser limpiado
        """
        self.df_clean = dataframe.copy()
        self.cleaning_report = {}
        self.tracker = tracker
    
    def convert_data_types(self, exclude_columns: Optional[list] = None):
        """
        Convierte columnas a tipos numéricos apropiados donde sea posible.
        
        Parámetros
        ----------
        exclude_columns : list, opcional
            Lista de nombres de columnas a excluir de la conversión
        """
        if exclude_columns is None:
            exclude_columns = ['url']  # Exclusión por defecto
            
        logger.info("Convirtiendo columnas a tipos numéricos...")
        
        conversion_count = 0
        converted_columns = []
        
        for col in self.df_clean.columns:
            if col not in exclude_columns:
                original_dtype = self.df_clean[col].dtype
                # Intentar convertir a numérico
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
                
                if self.df_clean[col].dtype != original_dtype:
                    logger.info(f"Convertida {col} de {original_dtype} a {self.df_clean[col].dtype}")
                    conversion_count += 1
                    converted_columns.append({
                        'columna': col,
                        'de': str(original_dtype),
                        'a': str(self.df_clean[col].dtype)
                    })
        self.cleaning_report['conversion_tipos'] = {
            'total_convertidas': conversion_count,
            'columnas_convertidas': converted_columns
        }
        self.tracker.track_change(self.df_clean, log_to_mlflow=False, description="Conversion de tipos de datos")
        logger.info(f"✅ Se convirtieron exitosamente {conversion_count} columnas a tipos numéricos")
        return self
    
    def handle_missing_values(self, strategy: str = 'drop', threshold: float = 0.5):
        """
        Maneja valores faltantes basándose en la estrategia especificada.
        
        Parámetros
        ----------
        strategy : str
            Estrategia para manejar valores faltantes ('drop', 'mean', 'median', 'mode', 'forward_fill')
        threshold : float
            Umbral para eliminar columnas con demasiados valores faltantes (0-1)
        """
        logger.info(f"Manejando valores faltantes con estrategia: {strategy}")
        
        initial_shape = self.df_clean.shape
        initial_nulls = self.df_clean.isnull().sum().sum()
        
        # Eliminar columnas con demasiados valores faltantes
        columns_to_drop = []
        if threshold < 1.0:
            null_percentages = self.df_clean.isnull().sum() / len(self.df_clean)
            columns_to_drop = null_percentages[null_percentages > threshold].index.tolist()
            
            if columns_to_drop:
                self.df_clean = self.df_clean.drop(columns=columns_to_drop)
                self.tracker.track_change(self.df_clean, log_to_mlflow=False,
                                           description=f"Eliminadas columnas con >{threshold*100}% nulos")
                logger.info(f"Eliminadas {len(columns_to_drop)} columnas con >{threshold*100}% valores faltantes")
        
        # Aplicar estrategia para valores faltantes restantes
        if strategy == 'drop':
            self.df_clean = self.df_clean.dropna()
            self.tracker.track_change(self.df_clean,
                                       log_to_mlflow=False, description="Eliminadas filas con valores nulos",)
        elif strategy == 'mean':
            numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
            self.df_clean[numeric_columns] = self.df_clean[numeric_columns].fillna(
                self.df_clean[numeric_columns].mean()
            )
            self.tracker.track_change(self.df_clean, log_to_mlflow=False, description=
                                        "Imputados nulos con media",)
        elif strategy == 'median':
            numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
            self.df_clean[numeric_columns] = self.df_clean[numeric_columns].fillna(
                self.df_clean[numeric_columns].median()
            )
            self.tracker.track_change(self.df_clean,
                                        log_to_mlflow=False, description="Imputados nulos con mediana",)
        elif strategy == 'mode':
            for col in self.df_clean.columns:
                if not self.df_clean[col].mode().empty:
                    self.df_clean[col].fillna(self.df_clean[col].mode()[0], inplace=True)
            self.tracker.track_change(self.df_clean,
                                       log_to_mlflow=False, description="Imputados nulos con moda",)
        elif strategy == 'forward_fill':
            self.df_clean = self.df_clean.fillna(method='ffill')
            self.tracker.track_change(self.df_clean,
                                        log_to_mlflow=False, description="Imputados nulos con forward fill",)
        
        final_nulls = self.df_clean.isnull().sum().sum()
        
        self.cleaning_report['manejo_valores_faltantes'] = {
            'estrategia': strategy,
            'forma_inicial': initial_shape,
            'forma_final': self.df_clean.shape,
            'nulos_iniciales': int(initial_nulls),
            'nulos_finales': int(final_nulls),
            'filas_eliminadas': initial_shape[0] - self.df_clean.shape[0],
            'columnas_eliminadas': len(columns_to_drop)
        }
        
        logger.info(f"✅ Eliminado de valores faltantes. Nulos restantes: {final_nulls}")
        return self
    
    def remove_duplicates(self, subset: Optional[list] = None, keep: str = 'first'):
        """
        Elimina filas duplicadas del dataframe.
        
        Parámetros
        ----------
        subset : list, opcional
            Etiquetas de columnas a considerar para identificar duplicados
        keep : str
            Cuáles duplicados mantener ('first', 'last', False)
        """ 
        logger.info("Eliminando filas duplicadas...")
        
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates(subset=subset, keep=keep)
        self.tracker.track_change(self.df_clean,
                                   log_to_mlflow=False, description="Eliminadas filas duplicadas",)
        duplicates_removed = initial_rows - len(self.df_clean)
        
        self.cleaning_report['duplicados'] = {
            'duplicados_eliminados': duplicates_removed,
            'conteo_filas_final': len(self.df_clean)
        }
        
        logger.info(f"✅ Se eliminaron {duplicates_removed} filas duplicadas")
        return self
    
    def handle_outliers(self, columns: Optional[list] = None, method: str = 'iqr', 
                       threshold: float = 1.5):
        """
        Maneja valores atípicos en columnas numéricas.
        
        Parámetros
        ----------
        columns : list, opcional
            Lista de columnas para verificar valores atípicos. Si es None, se usan todas las columnas numéricas
        method : str
            Método para detección de valores atípicos ('iqr', 'zscore')
        threshold : float
            Umbral para detección de valores atípicos (1.5 para IQR, 3 para z-score)
        """ 
        if columns is None:
            columns = self.df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Manejando valores atípicos usando método {method}...")
        
        outliers_info = {}
        total_outliers = 0
        
        for col in columns:
            if col in self.df_clean.columns and pd.api.types.is_numeric_dtype(self.df_clean[col]):
                if method == 'iqr':
                    Q1 = self.df_clean[col].quantile(0.25)
                    Q3 = self.df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers_mask = (self.df_clean[col] < lower_bound) | (self.df_clean[col] > upper_bound)
                    outliers_count = outliers_mask.sum()
                    
                    # Limitar valores atípicos en lugar de eliminarlos
                    self.df_clean.loc[self.df_clean[col] < lower_bound, col] = lower_bound
                    self.df_clean.loc[self.df_clean[col] > upper_bound, col] = upper_bound
                    self.tracker.track_change(self.df_clean,
                                               log_to_mlflow=False, description=f"Limitados valores atípicos en {col} usando IQR")
                    
                elif method == 'zscore':
                    z_scores = np.abs((self.df_clean[col] - self.df_clean[col].mean()) / self.df_clean[col].std())
                    outliers_mask = z_scores > threshold
                    outliers_count = outliers_mask.sum()
                    
                    # Eliminar filas con valores atípicos según z-score
                    self.df_clean = self.df_clean[~outliers_mask]
                    self.tracker.track_change(self.df_clean,
                                               log_to_mlflow=False, description=f"Eliminadas filas con valores atípicos en {col} usando z-score")
                if outliers_count > 0:
                    outliers_info[col] = int(outliers_count)
                    total_outliers += outliers_count
        
        self.cleaning_report['valores_atipicos'] = {
            'metodo': method,
            'umbral': threshold,
            'total_valores_atipicos_manejados': total_outliers,
            'columnas_afectadas': outliers_info
        }
        
        logger.info(f"✅ Se manejaron {total_outliers} valores atípicos en {len(outliers_info)} columnas")
        return self
    
    def save_cleaned_data(self, path: str):
        """
        Guarda el dataframe limpio en un archivo CSV.
        
        Parámetros
        ----------
        path : str
            Ruta del archivo donde guardar el dataframe limpio
        """
        self.df_clean.to_csv(path, index=False)
        logger.info(f"✅ Dataframe limpio guardado en {path}")
        return self