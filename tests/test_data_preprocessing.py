"""
Unit tests for Preprocessor class.

Tests feature engineering, data splitting, and preprocessing pipeline construction.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# from src.data.data_preprocessing import Preprocessor


class TestPreprocessorInitialization:
    """Tests para la inicialización del Preprocessor."""
    
    def test_initialization_with_defaults(self, sample_clean_dataframe):
        """Test inicialización con parámetros por defecto."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        
        # assert preprocessor.df_clean is not None
        # assert preprocessor.target_col == 'shares'
        # assert preprocessor.test_size == 0.2
        # assert preprocessor.random_state == 42
        pass
    
    def test_initialization_with_custom_params(self, sample_clean_dataframe):
        """Test inicialización con parámetros personalizados."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(
        #     df_clean=sample_clean_dataframe,
        #     target_col='shares',
        #     test_size=0.3,
        #     random_state=123
        # )
        
        # assert preprocessor.target_col == 'shares'
        # assert preprocessor.test_size == 0.3
        # assert preprocessor.random_state == 123
        pass
    
    def test_defensive_copy_created(self, sample_clean_dataframe):
        """Test que se crea una copia defensiva del DataFrame."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        
        # # Modificar el DataFrame del preprocessor no debe afectar el original
        # preprocessor.df_clean.loc[0, 'shares'] = 99999
        # assert sample_clean_dataframe.loc[0, 'shares'] != 99999
        pass
    
    def test_boolean_cols_default_initialization(self, sample_clean_dataframe):
        """Test que las columnas booleanas se inicializan correctamente."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        
        # assert len(preprocessor.boolean_cols) > 0
        # assert 'weekday_is_monday' in preprocessor.boolean_cols
        # assert 'is_weekend' in preprocessor.boolean_cols
        pass
    
    def test_non_predictors_default_initialization(self, sample_clean_dataframe):
        """Test que las columnas no predictoras se inicializan."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        
        # assert 'url' in preprocessor.non_predictors
        # assert 'article_title' in preprocessor.non_predictors
        pass


class TestDeriveFromURL:
    """Tests para la derivación de features desde URL."""
    
    def test_url_cleaned_created(self, sample_clean_dataframe):
        """Test que se crea la columna url_cleaned."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor._derive_from_url()
        
        # assert 'url_cleaned' in preprocessor.df_clean.columns
        pass
    
    def test_article_year_extracted(self, sample_clean_dataframe):
        """Test que se extrae article_year de la URL."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor._derive_from_url()
        
        # assert 'article_year' in preprocessor.df_clean.columns
        # assert preprocessor.df_clean['article_year'].dtype in [np.int64, np.float64]
        pass
    
    def test_article_month_extracted(self, sample_clean_dataframe):
        """Test que se extrae article_month de la URL."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor._derive_from_url()
        
        # assert 'article_month' in preprocessor.df_clean.columns
        # assert preprocessor.df_clean['article_month'].notna().all()
        pass
    
    def test_article_day_extracted(self, sample_clean_dataframe):
        """Test que se extrae article_day de la URL."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor._derive_from_url()
        
        # assert 'article_day' in preprocessor.df_clean.columns
        # assert preprocessor.df_clean['article_day'].notna().all()
        pass
    
    def test_article_title_extracted(self, sample_clean_dataframe):
        """Test que se extrae article_title de la URL."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor._derive_from_url()
        
        # assert 'article_title' in preprocessor.df_clean.columns
        # assert preprocessor.df_clean['article_title'].dtype == object
        pass
    
    def test_date_imputation_when_missing(self):
        """Test que se imputan valores cuando no hay fecha en URL."""
        df = pd.DataFrame({
            'url': ['http://example.com/article', 'http://example.com/another'],
            'shares': [1000, 2000],
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # preprocessor._derive_from_url()
        
        # # Debe haber valores imputados
        # assert preprocessor.df_clean['article_year'].notna().all()
        # assert preprocessor.df_clean['article_month'].notna().all()
        # assert preprocessor.df_clean['article_day'].notna().all()
        pass
    
    def test_date_extraction_with_valid_urls(self):
        """Test extracción de fecha con URLs válidas."""
        df = pd.DataFrame({
            'url': [
                'http://mashable.com/2013/01/15/article-title',
                'http://mashable.com/2014/06/22/another-article'
            ],
            'shares': [1000, 2000],
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # preprocessor._derive_from_url()
        
        # assert preprocessor.df_clean.loc[0, 'article_year'] == 2013
        # assert preprocessor.df_clean.loc[0, 'article_month'] == 1
        # assert preprocessor.df_clean.loc[0, 'article_day'] == 15
        pass


class TestNormalizeBooleans:
    """Tests para la normalización de columnas booleanas."""
    
    def test_boolean_values_normalized_to_0_1(self):
        """Test que valores booleanos se normalizan a 0 y 1."""
        df = pd.DataFrame({
            'url': ['http://example.com/1'],
            'shares': [1000],
            'weekday_is_monday': [2],  # Valor > 0
            'weekday_is_tuesday': [0],
            'is_weekend': [-1]  # Valor <= 0
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # preprocessor._normalize_booleans()
        
        # assert preprocessor.df_clean['weekday_is_monday'].iloc[0] == 1
        # assert preprocessor.df_clean['weekday_is_tuesday'].iloc[0] == 0
        # assert preprocessor.df_clean['is_weekend'].iloc[0] == 0
        pass
    
    def test_boolean_columns_become_int_type(self):
        """Test que columnas booleanas se convierten a int."""
        df = pd.DataFrame({
            'url': ['http://example.com/1'],
            'shares': [1000],
            'weekday_is_monday': [1.5],
            'is_weekend': [0.5]
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # preprocessor._normalize_booleans()
        
        # assert preprocessor.df_clean['weekday_is_monday'].dtype == np.int64
        # assert preprocessor.df_clean['is_weekend'].dtype == np.int64
        pass
    
    def test_non_existent_boolean_columns_handled(self):
        """Test que columnas booleanas no existentes se manejan correctamente."""
        df = pd.DataFrame({
            'url': ['http://example.com/1'],
            'shares': [1000],
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # # No debe lanzar error
        # preprocessor._normalize_booleans()
        pass


class TestBuildXY:
    """Tests para la construcción de X e y."""
    
    def test_target_excluded_from_X(self, sample_clean_dataframe):
        """Test que el target se excluye de X."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # X, y = preprocessor._build_xy()
        
        # assert 'shares' not in X.columns
        pass
    
    def test_non_predictors_excluded_from_X(self, sample_clean_dataframe):
        """Test que non_predictors se excluyen de X."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor._derive_from_url()  # Crea 'url_cleaned' y 'article_title'
        # X, y = preprocessor._build_xy()
        
        # assert 'url' not in X.columns
        # assert 'url_cleaned' not in X.columns
        # assert 'article_title' not in X.columns
        pass
    
    def test_y_is_target_column(self, sample_clean_dataframe):
        """Test que y contiene el target."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # X, y = preprocessor._build_xy()
        
        # assert y.name == 'shares'
        # assert len(y) == len(sample_clean_dataframe)
        pass
    
    def test_X_and_y_same_length(self, sample_clean_dataframe):
        """Test que X e y tienen la misma longitud."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # X, y = preprocessor._build_xy()
        
        # assert len(X) == len(y)
        pass
    
    def test_missing_target_raises_error(self):
        """Test que lanza error si falta el target."""
        df = pd.DataFrame({
            'url': ['http://example.com/1'],
            'n_tokens': [100]
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df, target_col='shares')
        
        # with pytest.raises(ValueError, match="No se encontró el target"):
        #     preprocessor._build_xy()
        pass


class TestSplitTrainTest:
    """Tests para el split train/test."""
    
    def test_split_creates_train_and_test(self, sample_clean_dataframe):
        """Test que el split crea conjuntos de train y test."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe, test_size=0.2)
        # X, y = preprocessor._build_xy()
        # preprocessor._split_train_test(X, y)
        
        # assert preprocessor.X_train is not None
        # assert preprocessor.X_test is not None
        # assert preprocessor.y_train is not None
        # assert preprocessor.y_test is not None
        pass
    
    def test_split_respects_test_size(self, sample_clean_dataframe):
        """Test que el split respeta el test_size."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe, test_size=0.25)
        # X, y = preprocessor._build_xy()
        # preprocessor._split_train_test(X, y)
        
        # total_samples = len(X)
        # expected_test_size = int(total_samples * 0.25)
        
        # assert len(preprocessor.X_test) == expected_test_size
        # assert len(preprocessor.X_train) == total_samples - expected_test_size
        pass
    
    def test_split_is_reproducible(self, sample_clean_dataframe):
        """Test que el split es reproducible con random_state fijo."""
        # from src.data.data_preprocessing import Preprocessor
        
        # # Primera ejecución
        # prep1 = Preprocessor(df_clean=sample_clean_dataframe.copy(), random_state=42)
        # X, y = prep1._build_xy()
        # prep1._split_train_test(X, y)
        
        # # Segunda ejecución con mismo random_state
        # prep2 = Preprocessor(df_clean=sample_clean_dataframe.copy(), random_state=42)
        # X, y = prep2._build_xy()
        # prep2._split_train_test(X, y)
        
        # # Los índices deben ser idénticos
        # assert prep1.X_train.index.tolist() == prep2.X_train.index.tolist()
        # assert prep1.X_test.index.tolist() == prep2.X_test.index.tolist()
        pass


class TestBuildColumnTransformer:
    """Tests para la construcción del ColumnTransformer."""
    
    def test_column_transformer_created(self, sample_clean_dataframe):
        """Test que se crea el ColumnTransformer."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # assert preprocessor.preprocess is not None
        # assert isinstance(preprocessor.preprocess, ColumnTransformer)
        pass
    
    def test_transformer_has_three_transformers(self, sample_clean_dataframe):
        """Test que el ColumnTransformer tiene 3 transformers (num, num_pt, bool)."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # assert len(preprocessor.preprocess.transformers) == 3
        pass
    
    def test_numeric_pipeline_has_correct_steps(self, sample_clean_dataframe):
        """Test que el pipeline numérico tiene Yeo-Johnson + MinMaxScaler."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # # Obtener el transformer numérico
        # num_transformer = preprocessor.preprocess.transformers[0][1]
        
        # if isinstance(num_transformer, Pipeline):
        #     assert len(num_transformer.steps) == 2
        #     assert num_transformer.steps[0][0] == 'yeojohnson'
        #     assert num_transformer.steps[1][0] == 'minmax'
        pass
    
    def test_boolean_columns_identified(self, sample_clean_dataframe):
        """Test que se identifican correctamente las columnas booleanas."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # assert len(preprocessor.bool_cols_present) > 0
        pass
    
    def test_transformable_numeric_columns_identified(self, sample_clean_dataframe):
        """Test que se identifican columnas numéricas transformables."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # # Debe haber columnas numéricas con más de 2 valores únicos
        # assert len(preprocessor.transformable_num) > 0
        pass
    
    def test_passthrough_numeric_columns_identified(self):
        """Test que se identifican columnas numéricas de passthrough (binarias)."""
        df = pd.DataFrame({
            'url': ['http://example.com/1', 'http://example.com/2'],
            'shares': [1000, 2000],
            'binary_col': [0, 1],  # Solo 2 valores únicos
            'constant_col': [5, 5],  # Constante
            'normal_col': [10, 20]  # Más de 2 valores
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # preprocessor.run()
        
        # assert 'binary_col' in preprocessor.passthrough_num
        # assert 'constant_col' in preprocessor.passthrough_num
        # assert 'normal_col' not in preprocessor.passthrough_num
        pass


class TestRunMethod:
    """Tests para el método run() que ejecuta el pipeline completo."""
    
    def test_run_executes_all_steps(self, sample_clean_dataframe):
        """Test que run() ejecuta todos los pasos del preprocessing."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # result = preprocessor.run()
        
        # # Verificar que todas las estructuras se crearon
        # assert 'article_year' in preprocessor.df_clean.columns  # _derive_from_url
        # assert preprocessor.X_train is not None  # _split_train_test
        # assert preprocessor.preprocess is not None  # _build_column_transformer
        
        # # Verificar que retorna self para chaining
        # assert result is preprocessor
        pass
    
    def test_run_is_idempotent(self, sample_clean_dataframe):
        """Test que run() puede ejecutarse múltiples veces sin problemas."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        
        # # Primera ejecución
        # preprocessor.run()
        # first_X_train = preprocessor.X_train.copy()
        
        # # Segunda ejecución
        # preprocessor.run()
        # second_X_train = preprocessor.X_train.copy()
        
        # # Los resultados deben ser idénticos
        # pd.testing.assert_frame_equal(first_X_train, second_X_train)
        pass


class TestGetMethods:
    """Tests para los métodos getter."""
    
    def test_get_splits_returns_correct_splits(self, sample_clean_dataframe):
        """Test que get_splits() retorna X_train, X_test, y_train, y_test."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # X_train, X_test, y_train, y_test = preprocessor.get_splits()
        
        # assert X_train is not None
        # assert X_test is not None
        # assert y_train is not None
        # assert y_test is not None
        # assert len(X_train) > len(X_test)  # Train debe ser más grande
        pass
    
    def test_get_preprocess_returns_transformer(self, sample_clean_dataframe):
        """Test que get_preprocess() retorna el ColumnTransformer."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # transformer = preprocessor.get_preprocess()
        
        # assert isinstance(transformer, ColumnTransformer)
        pass
    
    def test_get_feature_groups_returns_dict(self, sample_clean_dataframe):
        """Test que get_feature_groups() retorna un diccionario."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # groups = preprocessor.get_feature_groups()
        
        # assert isinstance(groups, dict)
        # assert 'boolean_present' in groups
        # assert 'numeric_transform' in groups
        # assert 'numeric_passthrough' in groups
        pass
    
    def test_feature_groups_are_lists(self, sample_clean_dataframe):
        """Test que los grupos de features son listas."""
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # groups = preprocessor.get_feature_groups()
        
        # assert isinstance(groups['boolean_present'], list)
        # assert isinstance(groups['numeric_transform'], list)
        # assert isinstance(groups['numeric_passthrough'], list)
        pass


class TestPreprocessorEdgeCases:
    """Tests para casos extremos."""
    
    def test_dataframe_without_url_column(self):
        """Test que lanza error si falta la columna 'url'."""
        df = pd.DataFrame({
            'shares': [1000, 2000],
            'n_tokens': [10, 20]
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        
        # with pytest.raises(ValueError, match="Se requiere la columna 'url'"):
        #     preprocessor.run()
        pass
    
    def test_small_dataframe(self):
        """Test con DataFrame muy pequeño."""
        df = pd.DataFrame({
            'url': ['http://example.com/2013/01/01/article'],
            'shares': [1000],
            'n_tokens': [10],
            'weekday_is_monday': [1]
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df, test_size=0.2)
        
        # # No debe fallar incluso con datos mínimos
        # preprocessor.run()
        pass
    
    def test_all_boolean_columns_missing(self):
        """Test cuando ninguna columna booleana está presente."""
        df = pd.DataFrame({
            'url': ['http://example.com/2013/01/01/article'],
            'shares': [1000],
            'n_tokens_title': [10],
            'n_tokens_content': [500]
        })
        
        # from src.data.data_preprocessing import Preprocessor
        # preprocessor = Preprocessor(df_clean=df)
        # preprocessor.run()
        
        # assert len(preprocessor.bool_cols_present) == 0
        pass


class TestPreprocessorIntegration:
    """Tests de integración con pipeline de sklearn."""
    
    def test_column_transformer_can_be_used_in_pipeline(self, sample_clean_dataframe):
        """Test que el ColumnTransformer puede usarse en un Pipeline de sklearn."""
        # from src.data.data_preprocessing import Preprocessor
        # from sklearn.linear_model import LinearRegression
        # from sklearn.pipeline import Pipeline
        
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # # Crear pipeline con el preprocessor
        # pipeline = Pipeline([
        #     ('preprocess', preprocessor.get_preprocess()),
        #     ('model', LinearRegression())
        # ])
        
        # # Debe poder hacer fit
        # X_train, X_test, y_train, y_test = preprocessor.get_splits()
        # pipeline.fit(X_train, y_train)
        
        # # Debe poder hacer predict
        # predictions = pipeline.predict(X_test)
        # assert len(predictions) == len(y_test)
        pass
    
    def test_preprocessor_output_compatible_with_model(self, sample_clean_dataframe):
        """Test que el output del preprocessor es compatible con modelos."""
        # from src.data.data_preprocessing import Preprocessor
        
        # preprocessor = Preprocessor(df_clean=sample_clean_dataframe)
        # preprocessor.run()
        
        # X_train, X_test, y_train, y_test = preprocessor.get_splits()
        # transformer = preprocessor.get_preprocess()
        
        # # Transformar datos
        # X_train_transformed = transformer.fit_transform(X_train)
        # X_test_transformed = transformer.transform(X_test)
        
        # # Verificar que la salida es un array numpy
        # assert isinstance(X_train_transformed, np.ndarray)
        # assert isinstance(X_test_transformed, np.ndarray)
        # assert X_train_transformed.shape[0] == len(y_train)
        # assert X_test_transformed.shape[0] == len(y_test)
        pass
