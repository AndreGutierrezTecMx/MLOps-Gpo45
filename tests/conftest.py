"""
Pytest configuration and shared fixtures for MLOps-GPO45 tests.

This module provides reusable fixtures for testing data pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """
    Crea un DataFrame de muestra que simula la estructura del dataset real.
    
    Returns:
        pd.DataFrame: DataFrame con estructura similar al dataset de noticias.
    """
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'url': [f'http://mashable.com/2013/01/{i:02d}/article-{i}' for i in range(1, n_rows + 1)],
        'shares': np.random.randint(100, 10000, n_rows),
        'n_tokens_title': np.random.randint(5, 20, n_rows),
        'n_tokens_content': np.random.randint(100, 2000, n_rows),
        'num_imgs': np.random.randint(0, 20, n_rows),
        'num_videos': np.random.randint(0, 10, n_rows),
        'num_hrefs': np.random.randint(5, 50, n_rows),
        
        # Boolean columns - data channels
        'data_channel_is_lifestyle': np.random.choice([0, 1], n_rows),
        'data_channel_is_entertainment': np.random.choice([0, 1], n_rows),
        'data_channel_is_bus': np.random.choice([0, 1], n_rows),
        'data_channel_is_socmed': np.random.choice([0, 1], n_rows),
        'data_channel_is_tech': np.random.choice([0, 1], n_rows),
        'data_channel_is_world': np.random.choice([0, 1], n_rows),
        
        # Boolean columns - weekdays
        'weekday_is_monday': np.random.choice([0, 1], n_rows),
        'weekday_is_tuesday': np.random.choice([0, 1], n_rows),
        'weekday_is_wednesday': np.random.choice([0, 1], n_rows),
        'weekday_is_thursday': np.random.choice([0, 1], n_rows),
        'weekday_is_friday': np.random.choice([0, 1], n_rows),
        'weekday_is_saturday': np.random.choice([0, 1], n_rows),
        'weekday_is_sunday': np.random.choice([0, 1], n_rows),
        'is_weekend': np.random.choice([0, 1], n_rows),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_with_nulls():
    """
    Crea un DataFrame de muestra con valores faltantes para testing de limpieza.
    
    Returns:
        pd.DataFrame: DataFrame con valores nulos estratégicamente colocados.
    """
    df = pd.DataFrame({
        'url': ['http://example.com/article-1', 'http://example.com/article-2', 
                'http://example.com/article-3', 'http://example.com/article-4'],
        'shares': [1000, np.nan, 3000, 4000],
        'n_tokens_title': [10, 12, np.nan, 15],
        'n_tokens_content': [500, 600, 700, np.nan],
        'num_imgs': [5, np.nan, 8, 10],
    })
    return df


@pytest.fixture
def sample_dataframe_with_duplicates():
    """
    Crea un DataFrame con filas duplicadas para testing de eliminación.
    
    Returns:
        pd.DataFrame: DataFrame con duplicados.
    """
    df = pd.DataFrame({
        'url': ['http://example.com/article-1', 'http://example.com/article-1',
                'http://example.com/article-2', 'http://example.com/article-3'],
        'shares': [1000, 1000, 2000, 3000],
        'n_tokens_title': [10, 10, 12, 15],
    })
    return df


@pytest.fixture
def sample_dataframe_with_outliers():
    """
    Crea un DataFrame con valores atípicos para testing de manejo de outliers.
    
    Returns:
        pd.DataFrame: DataFrame con outliers claros.
    """
    np.random.seed(42)
    normal_data = np.random.normal(1000, 200, 95)
    outliers = [10000, 15000, 20000, 50000, 100000]
    
    df = pd.DataFrame({
        'shares': np.concatenate([normal_data, outliers]),
        'n_tokens_content': np.random.randint(100, 2000, 100),
    })
    return df


@pytest.fixture
def sample_clean_dataframe():
    """
    Crea un DataFrame limpio listo para preprocessing.
    
    Returns:
        pd.DataFrame: DataFrame limpio sin nulos, duplicados ni outliers.
    """
    np.random.seed(42)
    n_rows = 50
    
    data = {
        'url': [f'http://mashable.com/2013/01/{i:02d}/article-{i}' for i in range(1, n_rows + 1)],
        'shares': np.random.randint(500, 5000, n_rows),
        'n_tokens_title': np.random.randint(5, 20, n_rows),
        'n_tokens_content': np.random.randint(200, 1500, n_rows),
        'num_imgs': np.random.randint(0, 15, n_rows),
        'num_videos': np.random.randint(0, 5, n_rows),
        'num_hrefs': np.random.randint(5, 30, n_rows),
        
        # Boolean columns
        'data_channel_is_lifestyle': np.random.choice([0, 1], n_rows),
        'data_channel_is_tech': np.random.choice([0, 1], n_rows),
        'weekday_is_monday': np.random.choice([0, 1], n_rows),
        'is_weekend': np.random.choice([0, 1], n_rows),
    }
    
    return pd.DataFrame(data)


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_version_tracker():
    """
    Crea un mock del VersionTracker para evitar dependencias de DVC/MLflow en tests.
    
    Returns:
        Mock: Mock object que simula el VersionTracker.
    """
    tracker = Mock()
    tracker.track_dvc_change = Mock(return_value=None)
    tracker.log_to_mlflow = Mock(return_value=None)
    tracker.version = "test-version-1.0.0"
    return tracker


@pytest.fixture
def mock_logger():
    """
    Crea un mock del logger para evitar output durante tests.
    
    Returns:
        Mock: Mock object que simula el logger.
    """
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.exception = Mock()
    return logger


# ============================================================================
# FILE FIXTURES
# ============================================================================

@pytest.fixture
def temp_csv_file(tmp_path, sample_dataframe):
    """
    Crea un archivo CSV temporal para testing de lectura de archivos.
    
    Args:
        tmp_path: Fixture de pytest que provee un directorio temporal.
        sample_dataframe: DataFrame de muestra.
    
    Returns:
        Path: Ruta al archivo CSV temporal.
    """
    file_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Crea un directorio temporal para guardar outputs de tests.
    
    Args:
        tmp_path: Fixture de pytest que provee un directorio temporal.
    
    Returns:
        Path: Ruta al directorio de output temporal.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """
    Provee configuración de testing común.
    
    Returns:
        dict: Diccionario con parámetros de configuración.
    """
    return {
        'random_state': 42,
        'test_size': 0.2,
        'target_col': 'shares',
        'outlier_threshold': 1.5,
    }


# ============================================================================
# PARAMETRIZED DATA FIXTURES
# ============================================================================

@pytest.fixture(params=['drop', 'mean', 'median', 'mode'])
def missing_value_strategy(request):
    """
    Parametriza las estrategias de manejo de valores faltantes.
    
    Returns:
        str: Estrategia de manejo de valores faltantes.
    """
    return request.param


@pytest.fixture(params=['iqr', 'zscore'])
def outlier_method(request):
    """
    Parametriza los métodos de detección de outliers.
    
    Returns:
        str: Método de detección de outliers.
    """
    return request.param


@pytest.fixture(params=['first', 'last', False])
def duplicate_keep_strategy(request):
    """
    Parametriza las estrategias de mantener duplicados.
    
    Returns:
        str or bool: Estrategia para mantener duplicados.
    """
    return request.param
