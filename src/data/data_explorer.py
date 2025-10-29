from typing import List
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class DataExplorer:
    """A class for exploring and analyzing datasets."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initializes the DataExplorer with a pandas DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame to be explored.
        """
        self.dataframe: pd.DataFrame = dataframe
        self.categorical_cols: List[str] = self.dataframe.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols: List[str] = self.dataframe.select_dtypes(include=[np.number]).columns.tolist()

    def print_head(self, n: int = 5) -> None:
        """Prints the first n rows of the DataFrame"""
        logger.info(f"Mostrando las primeras {n} filas del DataFrame:")
        print(self.dataframe.head(n))

    def print_info(self) -> None:
        """Prints information about the DataFrame."""
        info = self.dataframe.info()
        logger.info(f"Mostrando información del DataFrame:{info}")
        print(info)

    def print_descriptive_statistics(self) -> None:
        """Returns descriptive statistics of the DataFrame."""
        descriptive_stats: pd.DataFrame = self.dataframe.describe()
        logger.info(f"Mostrando estadísticas descriptivas del DataFrame:{descriptive_stats}")
        print(descriptive_stats)

    def missing_values_analysis(self) -> None:
        """Returns a DataFrame with the count of missing values for each column."""
        null_counts: pd.Series = self.dataframe.isnull().sum()
        null_percentages: pd.Series = (self.dataframe.isnull().sum() / len(self.dataframe)) * 100
        null_info: pd.DataFrame = pd.DataFrame({
            'Nulos': null_counts,
            'Porcentaje': null_percentages
        })
        null_info = null_info[null_info['Nulos'] > 0].sort_values('Nulos', ascending=False)
        logger.info(f"Análisis de valores faltantes: {null_info}")
        print(null_info)

    def duplicate_analysis(self) -> None:
        """Returns the number of duplicate rows in the DataFrame."""
        duplicates: int = self.dataframe.duplicated().sum()
        logger.info(f"Número de filas duplicadas en el DataFrame: {duplicates}")
        print(f"\nFilas duplicadas: {duplicates}")

    def column_analysis(self) -> None:
        """Returns the number of columns and their data types in the DataFrame."""
        num_columns: int = len(self.numeric_cols)
        cat_columns: int = len(self.categorical_cols)
        print("\nColumnas numéricas vs categóricas:")
        logger.info(f"Columnas numéricas: {num_columns}, Columnas categóricas: {cat_columns}")
        print(f"   Numéricas: {num_columns}")
        print(f"   Categóricas/Object: {len(self.categorical_cols)}")

        if self.categorical_cols:
            logger.info(f"Columnas categóricas: {self.categorical_cols}")
            print("\nColumnas categóricas:")
            for col in self.categorical_cols:
                print(f"   - {col}")

    def unique_values_analysis(self) -> None:
        """Returns the number of unique values for each column in the DataFrame."""
        if self.categorical_cols.empty:
            logger.info("No hay columnas categóricas para analizar.")
            print("\n   ✓ No hay columnas categóricas para analizar")
            return

        for col in self.categorical_cols[:10]:  # Primeras 10 para no saturar
            unique_count: int = self.dataframe[col].nunique()
            logger.info(f"Columna: {col}, Valores únicos: {unique_count}")
            print(f"\n{col}:")
            print(f"   Valores únicos: {unique_count}")
            if unique_count < 20:
                logger.info(f"Valores en {col}: {self.dataframe[col].unique()}")
                print(f"   Valores: {self.dataframe[col].unique()[:10]}")


    def full_report(self) -> None:
        """Generates a full report of the DataFrame analysis."""
        logger.info("🔎 Generando exploración completa del DataFrame:")
        self.print_head()
        self.print_info()
        self.print_descriptive_statistics()
        self.missing_values_analysis()
        self.duplicate_analysis()
        self.column_analysis()
        self.unique_values_analysis()
        logger.info("✅ Exploración completa del DataFrame finalizada.")