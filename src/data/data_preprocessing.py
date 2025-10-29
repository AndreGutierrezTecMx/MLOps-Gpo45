import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Prepara el dataset para modelado:
      - Features derivadas de 'url' (año, mes, día, título)
      - Normaliza booleanas como 0/1
      - Construye ColumnTransformer (Yeo-Johnson + MinMax en numéricas, passthrough en booleanas)
      - Separa Train/Test
    """

    def __init__(
        self,
        df_clean: pd.DataFrame,
        target_col: str = "shares",
        boolean_cols: List[str] = None,
        non_predictors: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        # Guarda referencias básicas
        self.df_clean = df_clean.copy()                  # copia de trabajo
        self.target_col = target_col                     # nombre del target
        self.test_size = test_size                       # tamaño test
        self.random_state = random_state                 # semilla

        # Listas por defecto si no vienen
        self.boolean_cols = boolean_cols or [
            'data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus',
            'data_channel_is_socmed','data_channel_is_tech','data_channel_is_world',
            'weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday',
            'weekday_is_thursday','weekday_is_friday','weekday_is_saturday',
            'weekday_is_sunday','is_weekend'
        ]
        self.non_predictors = non_predictors or ['url','article_title','url_cleaned','mixed_type_col']

        # Placeholders que se llenan al ejecutar
        self.preprocess: ColumnTransformer = None        # ColumnTransformer final
        self.bool_cols_present: List[str] = []           # booleanas presentes en X
        self.transformable_num: List[str] = []           # numéricas con YJ + MinMax
        self.passthrough_num: List[str] = []             # numéricas passthrough
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None

    # ---------- Público ----------

    def run(self) -> "Preprocessor":
        """Ejecuta todo el flujo de preprocesamiento y deja todo listo para modelado."""
        self._derive_from_url()          # crea article_year/month/day y article_title
        self._normalize_booleans()       # asegura 0/1 en booleanas
        X, y = self._build_xy()          # arma X e y removiendo no predictoras
        self._split_train_test(X, y)     # separa train/test
        self._build_column_transformer() # define ColumnTransformer
        return self

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Devuelve X_train, X_test, y_train, y_test."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_preprocess(self) -> ColumnTransformer:
        """Devuelve el ColumnTransformer final para reusar en pipelines."""
        return self.preprocess

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Devuelve grupos de columnas (útil para trazabilidad)."""
        return {
            "boolean_present": self.bool_cols_present,
            "numeric_transform": self.transformable_num,
            "numeric_passthrough": self.passthrough_num
        }

    # ---------- Privado ----------

    def _derive_from_url(self) -> None:
        """Crea columnas desde 'url' y limpia título."""
        df = self.df_clean
        df['url_cleaned'] = df['url'].astype(str).str.strip()                         # normaliza URL
        date_pattern = r'/(\d{4})/(\d{2})/(\d{2})/'                                   # YYYY/MM/DD
        date_match = df['url_cleaned'].str.extract(date_pattern)                      # extrae grupos

        df['article_year']  = pd.to_numeric(date_match[0], errors='coerce')           # año num
        df['article_month'] = pd.to_numeric(date_match[1], errors='coerce')           # mes num
        df['article_day']   = pd.to_numeric(date_match[2], errors='coerce')           # día num

        df['article_title'] = df['url_cleaned'].str.split('/').str[-2]                # penúltimo segmento
        df['article_title'] = df['article_title'].str.replace('-', ' ').str.title()   # limpia y capitaliza

        # imputación simple por moda (evita NaNs)
        for c in ['article_year','article_month','article_day']:
            if df[c].isna().any():
                df[c].fillna(df[c].mode().iloc[0], inplace=True)

        self.df_clean = df

    def _normalize_booleans(self) -> None:
        """Convierte columnas booleanas a 0/1 enteros."""
        for c in self.boolean_cols:
            if c in self.df_clean.columns:
                self.df_clean[c] = (self.df_clean[c] > 0).astype(int)

    def _build_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Genera X e y removiendo no predictoras."""
        drop_cols = [self.target_col] + [c for c in self.non_predictors if c in self.df_clean.columns]
        X = self.df_clean.drop(columns=drop_cols, errors="ignore")   # features
        y = self.df_clean[self.target_col].copy()                    # target
        return X, y

    def _split_train_test(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split reproducible 80/20 con shuffle."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

    def _build_column_transformer(self) -> None:
        """Crea ColumnTransformer: Yeo-Johnson + MinMax para numéricas transformables; passthrough para booleanas y numéricas no transformables."""
        X_train = self.X_train

        # booleanas presentes en X
        self.bool_cols_present = [c for c in self.boolean_cols if c in X_train.columns]

        # numéricas candidatas (excluye booleanas)
        num_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
        num_all = [c for c in num_all if c not in self.bool_cols_present]

        # separa transformables vs passthrough
        self.transformable_num, self.passthrough_num = [], []
        for c in num_all:
            vals = X_train[c].dropna()
            if vals.nunique() > 2 and vals.std() > 0:  # evita binarias/constantes
                self.transformable_num.append(c)
            else:
                self.passthrough_num.append(c)

        # pipeline numérico
        num_pipe = Pipeline(steps=[
            ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),  # reduce sesgo
            ("minmax", MinMaxScaler())                                                 # escala 0-1
        ])

        # ColumnTransformer final
        self.preprocess = ColumnTransformer(
            transformers=[
                ("num",   num_pipe,              self.transformable_num),
                ("num_pt","passthrough",         self.passthrough_num),
                ("bool",  "passthrough",         self.bool_cols_present),
            ],
            remainder="drop"
        )