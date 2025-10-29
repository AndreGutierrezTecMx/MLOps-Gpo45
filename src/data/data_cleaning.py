"""
Módulo para operaciones de limpieza de datos en el proyecto MLOps-GPO45.
Sigue la estructura Cookiecutter Data Science / MLflow Project.
"""

import pandas as pd
import numpy as np

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaning:
    """Una clase para limpiar y preprocesar dataframes desde datos crudos."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa DataCleaning con un dataframe.
        
        Parámetros
        ----------
        dataframe : pd.DataFrame
            El dataframe crudo a ser limpiado
        """
        self.df = dataframe
        self.df_clean = None
        self.cleaning_report = {}
    
    def convert_data_types(self, exclude_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Convierte columnas a tipos numéricos apropiados donde sea posible.
        
        Parámetros
        ----------
        exclude_columns : list, opcional
            Lista de nombres de columnas a excluir de la conversión
        
        Retorna
        -------
        pd.DataFrame
            DataFrame con tipos de datos convertidos
        """
        if exclude_columns is None:
            exclude_columns = ['url']  # Exclusión por defecto
            
        logger.info("Convirtiendo columnas a tipos numéricos...")
        
        if self.df_clean is None:
            self.df_clean = self.df.copy()
        
        conversion_count = 0
        converted_columns = []
        
        for col in self.df_clean.columns:
            if col not in exclude_columns:
                original_dtype = self.df_clean[col].dtype
                # Intentar convertir a numérico
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce')
                
                if self.df_clean[col].dtype != original_dtype:
                    logger.debug(f"Convertida {col} de {original_dtype} a {self.df_clean[col].dtype}")
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
        
        logger.info(f"Se convirtieron exitosamente {conversion_count} columnas a tipos numéricos")
        return self.df_clean
    
    def handle_missing_values(self, strategy: str = 'drop', threshold: float = 0.5) -> pd.DataFrame:
        """
        Maneja valores faltantes basándose en la estrategia especificada.
        
        Parámetros
        ----------
        strategy : str
            Estrategia para manejar valores faltantes ('drop', 'mean', 'median', 'mode', 'forward_fill')
        threshold : float
            Umbral para eliminar columnas con demasiados valores faltantes (0-1)
        
        Retorna
        -------
        pd.DataFrame
            DataFrame con valores faltantes manejados
        """
        if self.df_clean is None:
            self.df_clean = self.df.copy()
            
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
                logger.info(f"Eliminadas {len(columns_to_drop)} columnas con >{threshold*100}% valores faltantes")
        
        # Aplicar estrategia para valores faltantes restantes
        if strategy == 'drop':
            self.df_clean = self.df_clean.dropna()
        elif strategy == 'mean':
            numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
            self.df_clean[numeric_columns] = self.df_clean[numeric_columns].fillna(
                self.df_clean[numeric_columns].mean()
            )
        elif strategy == 'median':
            numeric_columns = self.df_clean.select_dtypes(include=[np.number]).columns
            self.df_clean[numeric_columns] = self.df_clean[numeric_columns].fillna(
                self.df_clean[numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in self.df_clean.columns:
                if not self.df_clean[col].mode().empty:
                    self.df_clean[col].fillna(self.df_clean[col].mode()[0], inplace=True)
        elif strategy == 'forward_fill':
            self.df_clean = self.df_clean.fillna(method='ffill')
        
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
        
        logger.info(f"Valores faltantes manejados. Nulos restantes: {final_nulls}")
        return self.df_clean
    
    def remove_duplicates(self, subset: Optional[list] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Elimina filas duplicadas del dataframe.
        
        Parámetros
        ----------
        subset : list, opcional
            Etiquetas de columnas a considerar para identificar duplicados
        keep : str
            Cuáles duplicados mantener ('first', 'last', False)
        
        Retorna
        -------
        pd.DataFrame
            DataFrame con duplicados eliminados
        """
        if self.df_clean is None:
            self.df_clean = self.df.copy()
            
        logger.info("Eliminando filas duplicadas...")
        
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = initial_rows - len(self.df_clean)
        
        self.cleaning_report['duplicados'] = {
            'duplicados_eliminados': duplicates_removed,
            'conteo_filas_final': len(self.df_clean)
        }
        
        logger.info(f"Se eliminaron {duplicates_removed} filas duplicadas")
        return self.df_clean
    
    def handle_outliers(self, columns: Optional[list] = None, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
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
        
        Retorna
        -------
        pd.DataFrame
            DataFrame con valores atípicos manejados
        """
        if self.df_clean is None:
            self.df_clean = self.df.copy()
            
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
                    
                elif method == 'zscore':
                    z_scores = np.abs((self.df_clean[col] - self.df_clean[col].mean()) / self.df_clean[col].std())
                    outliers_mask = z_scores > threshold
                    outliers_count = outliers_mask.sum()
                    
                    # Eliminar filas con valores atípicos según z-score
                    self.df_clean = self.df_clean[~outliers_mask]
                
                if outliers_count > 0:
                    outliers_info[col] = int(outliers_count)
                    total_outliers += outliers_count
        
        self.cleaning_report['valores_atipicos'] = {
            'metodo': method,
            'umbral': threshold,
            'total_valores_atipicos_manejados': total_outliers,
            'columnas_afectadas': outliers_info
        }
        
        logger.info(f"Se manejaron {total_outliers} valores atípicos en {len(outliers_info)} columnas")
        return self.df_clean
    
    def normalize_column_names(self) -> pd.DataFrame:
        """
        Normaliza los nombres de columnas a minúsculas con guiones bajos.
        
        Retorna
        -------
        pd.DataFrame
            DataFrame con nombres de columnas normalizados
        """
        if self.df_clean is None:
            self.df_clean = self.df.copy()
            
        logger.info("Normalizando nombres de columnas...")
        
        original_columns = self.df_clean.columns.tolist()
        self.df_clean.columns = (self.df_clean.columns
                                 .str.lower()
                                 .str.replace(' ', '_')
                                 .str.replace('[^a-z0-9_]', '', regex=True))
        
        renamed_columns = {old: new for old, new in zip(original_columns, self.df_clean.columns) if old != new}
        
        self.cleaning_report['normalizacion_columnas'] = {
            'columnas_renombradas': len(renamed_columns),
            'mapeo_renombrado': renamed_columns
        }
        
        logger.info(f"Se normalizaron {len(renamed_columns)} nombres de columnas")
        return self.df_clean
    
    def get_cleaned_dataframe(self) -> pd.DataFrame:
        """
        Obtiene el dataframe limpio.
        
        Retorna
        -------
        pd.DataFrame
            El dataframe limpio
        """
        if self.df_clean is None:
            logger.warning("No se han realizado operaciones de limpieza aún. Retornando dataframe original.")
            return self.df
        return self.df_clean
    
    def get_cleaning_report(self) -> Dict:
        """
        Obtiene un reporte completo de todas las operaciones de limpieza realizadas.
        
        Retorna
        -------
        dict
            Diccionario con detalles de todas las operaciones de limpieza
        """
        return self.cleaning_report
    
    def save_cleaned_data(self, filepath: Union[str, Path], format: str = 'csv') -> None:
        """
        Guarda el dataframe limpio en un archivo.
        
        Parámetros
        ----------
        filepath : str o Path
            Ruta donde se debe guardar el dato limpio
        format : str
            Formato para guardar el archivo ('csv', 'parquet', 'pickle')
        """
        if self.df_clean is None:
            logger.warning("No hay datos limpios para guardar. Ejecute operaciones de limpieza primero.")
            return
        
        filepath = Path(filepath)
        
        if format == 'csv':
            self.df_clean.to_csv(filepath, index=False)
        elif format == 'parquet':
            self.df_clean.to_parquet(filepath, index=False)
        elif format == 'pickle':
            self.df_clean.to_pickle(filepath)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Datos limpios guardados en {filepath}")
    
    def run_standard_cleaning_pipeline(self, 
                                      missing_strategy: str = 'drop',
                                      handle_outliers: bool = True,
                                      remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Ejecuta un pipeline de limpieza estándar con operaciones comunes.
        
        Parámetros
        ----------
        missing_strategy : str
            Estrategia para manejar valores faltantes
        handle_outliers : bool
            Si se deben manejar valores atípicos
        remove_duplicates : bool
            Si se deben eliminar filas duplicadas
        
        Retorna
        -------
        pd.DataFrame
            Dataframe completamente limpio
        """
        logger.info("Ejecutando pipeline de limpieza estándar...")
        
        # 1. Analizar nulos
        # Metodo eliminado. Analizar valores nulos corresponde al data explorer
        
        # 2. Convertir tipos de datos
        self.convert_data_types()
        
        # 3. Eliminar duplicados si se solicita
        if remove_duplicates:
            self.remove_duplicates()
        
        # 4. Manejar valores faltantes
        self.handle_missing_values(strategy=missing_strategy)
        
        # 5. Manejar valores atípicos si se solicita
        if handle_outliers:
            self.handle_outliers()
        
        # 6. Normalizar nombres de columnas
        self.normalize_column_names()
        
        logger.info("Pipeline de limpieza estándar completado exitosamente")
        logger.info(f"Forma final: {self.df_clean.shape}")
        
        return self.df_clean


# Funciones de utilidad para tareas comunes de limpieza de datos
def load_and_clean_data(filepath: Union[str, Path], 
                        missing_strategy: str = 'drop',
                        save_cleaned: bool = False,
                        output_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Función de conveniencia para cargar y limpiar datos en un solo paso.
    
    Parámetros
    ----------
    filepath : str o Path
        Ruta al archivo de datos crudos
    missing_strategy : str
        Estrategia para manejar valores faltantes
    save_cleaned : bool
        Si se deben guardar los datos limpios
    output_path : str o Path, opcional
        Ruta para guardar datos limpios (requerida si save_cleaned=True)
    
    Retorna
    -------
    tuple
        (dataframe_limpio, reporte_limpieza)
    """
    logger.info(f"Cargando datos desde {filepath}")
    
    # Cargar datos
    filepath = Path(filepath)
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato de archivo no soportado: {filepath.suffix}")
    
    # Limpiar datos
    cleaner = DataCleaning(df)
    df_clean = cleaner.run_standard_cleaning_pipeline(missing_strategy=missing_strategy)
    
    # Guardar si se solicita
    if save_cleaned:
        if output_path is None:
            output_path = filepath.parent / f"{filepath.stem}_cleaned{filepath.suffix}"
        cleaner.save_cleaned_data(output_path)
    
    return df_clean, cleaner.get_cleaning_report()


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        df_clean, report = load_and_clean_data(
            input_file,
            missing_strategy='drop',
            save_cleaned=True,
            output_path=output_file
        )
        
        print("\n=== REPORTE DE LIMPIEZA DE DATOS ===")
        for operation, details in report.items():
            print(f"\n{operation.upper()}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
    else:
        print("Uso: python data_cleaning.py <archivo_entrada> [archivo_salida]")
