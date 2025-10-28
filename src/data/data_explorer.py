import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class DataExplorer:
    """A class for exploring and analyzing datasets."""

    def __init__(self, dataframe):
        """
        Initializes the DataExplorer with a pandas DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame to be explored.
        """
        self.dataframe = dataframe
        self.categorical_cols = self.dataframe.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns.tolist()

    def print_head(self, n:int=5):
        """Prints the first n rows of the DataFrame"""
        logger.info(f"Mostrando las primeras {n} filas del DataFrame:")
        print(self.dataframe.head(n))

    def print_info(self):
        """Prints information about the DataFrame."""
        logger.info("Mostrando información del DataFrame:")
        print(self.dataframe.info())

    def print_descriptive_statistics(self):
        """Returns descriptive statistics of the DataFrame."""
        logger.info("Mostrando estadísticas descriptivas del DataFrame:")
        print(self.dataframe.describe())

    def missing_values_analysis(self):
        """Returns a DataFrame with the count of missing values for each column."""
        null_counts = self.dataframe.isnull().sum()
        null_percentages = (self.dataframe.isnull().sum() / len(self.dataframe)) * 100 
        null_info = pd.DataFrame({
            'Nulos': null_counts,
            'Porcentaje': null_percentages
        })
        null_info = null_info[null_info['Nulos'] > 0].sort_values('Nulos', ascending=False)
        logger.info("Análisis de valores faltantes:")
        print(null_info)

    def duplicate_analysis(self):
        """Returns the number of duplicate rows in the DataFrame."""
        duplicates = self.dataframe.duplicated().sum()
        logger.info(f"Número de filas duplicadas en el DataFrame: {duplicates}")
        print(f"\nFilas duplicadas: {duplicates}")

    def column_analysis(self):
        """Returns the number of columns and their data types in the DataFrame."""
        logger.info("Análisis de columnas del DataFrame:")
        print("\nColumnas numéricas vs categóricas:")
        print(f"   Numéricas: {len(self.numeric_cols)}")
        print(f"   Categóricas/Object: {len(self.categorical_cols)}")
        if self.categorical_cols:
            print("\nColumnas categóricas:")
            for col in self.categorical_cols:
                print(f"   - {col}")

    def unique_values_analysis(self):
        """Returns the number of unique values for each column in the DataFrame."""
        logger.info("Análisis de valores únicos por columna:")
        if self.categorical_cols:
            for col in self.categorical_cols[:10]:  # Primeras 10 para no saturar
                unique_count = self.dataframe[col].nunique()
                print(f"\n{col}:")
                print(f"   Valores únicos: {unique_count}")
                if unique_count < 20:
                    print(f"   Valores: {self.dataframe[col].unique()[:10]}")
        else:
            print("\n   ✓ No hay columnas categóricas para analizar")

    def full_report(self):
        """Generates a full report of the DataFrame analysis."""
        logger.info("Generando exploración completa del DataFrame:")
        self.print_head()
        self.print_info()
        self.print_descriptive_statistics()
        self.missing_values_analysis()
        self.duplicate_analysis()
        self.column_analysis()
        self.unique_values_analysis()