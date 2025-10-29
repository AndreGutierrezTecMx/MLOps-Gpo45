"""
Preprocessor
-----------
Clase que implementa el flujo de preprocesamiento del dataset:

1. Deriva nuevas variables a partir de 'url' (article_year, article_month, article_day, article_title).
2. Normaliza las columnas booleanas a formato 0/1.
3. Separa las variables predictoras (X) y el target (y), aplicando train/test split.
4. Construye un ColumnTransformer con transformaciones:
   - PowerTransformer (Yeo-Johnson) y MinMaxScaler para columnas numéricas transformables.
   - Passthrough para columnas binarias o constantes.
   - Passthrough para columnas booleanas.

Uso:
    pre = Preprocessor(df_clean=df, target_col="shares").run()
    X_train, X_test, y_train, y_test = pre.get_splits()
    preprocess = pre.get_preprocess()
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    def __init__(
        self,
        df_clean: pd.DataFrame,
        target_col: str = "shares",
        boolean_cols: List[str] = None,
        non_predictors: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        # Se crea una copia del DataFrame para evitar modificar el original
        self.df_clean = df_clean.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        # Columnas booleanas conocidas del dataset (si no se proporcionan, se asignan por defecto)
        self.boolean_cols = boolean_cols or [
            "data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus",
            "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world",
            "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday",
            "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday",
            "weekday_is_sunday", "is_weekend"
        ]

        # Columnas que no deben incluirse en el modelado (no predictoras)
        self.non_predictors = non_predictors or ["url", "article_title", "url_cleaned", "mixed_type_col"]

        # Inicialización de variables de salida
        self.preprocess: ColumnTransformer = None
        self.bool_cols_present: List[str] = []
        self.transformable_num: List[str] = []
        self.passthrough_num: List[str] = []
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # ---------- MÉTODO PRINCIPAL ----------
    def run(self) -> "Preprocessor":
        """
        Ejecuta el flujo completo de preprocesamiento:
        1. Derivación de variables desde la URL.
        2. Normalización de booleanas.
        3. Creación de X, y.
        4. División train/test.
        5. Construcción del ColumnTransformer.
        """
        logger.info("⏳ [Preprocessor] Iniciando preprocesamiento…")
        self._derive_from_url()
        self._normalize_booleans()
        X, y = self._build_xy()
        self._split_train_test(X, y)
        self._build_column_transformer()
        logger.info("✅ [Preprocessor] Preprocesamiento completado.")
        return self

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Devuelve los conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test)."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_preprocess(self) -> ColumnTransformer:
        """Devuelve el ColumnTransformer generado."""
        return self.preprocess

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Devuelve los grupos de columnas detectadas durante el preprocesamiento:
            - boolean_present: columnas booleanas incluidas.
            - numeric_transform: columnas numéricas transformadas.
            - numeric_passthrough: columnas numéricas sin transformación.
        """
        return {
            "boolean_present": self.bool_cols_present,
            "numeric_transform": self.transformable_num,
            "numeric_passthrough": self.passthrough_num,
        }

    # ---------- MÉTODOS INTERNOS ----------
    @staticmethod
    def _apply_raw_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deriva columnas desde 'url' y genera variables temporales:
        - Limpieza del texto de URL.
        - Extracción de fecha (año, mes, día).
        - Creación de una variable de título a partir de la URL.
        Aplica imputación de valores faltantes mediante defaults o moda.
        """
        out = df.copy()
        if "url" not in out.columns:
            raise ValueError("Se requiere la columna 'url' para derivar article_year/month/day/title.")

        # Limpieza básica de la columna URL
        out["url_cleaned"] = out["url"].astype(str).str.strip()
        logger.info("✔️ url_cleaned creada")

        # Extracción de componentes de fecha
        date_match = out["url_cleaned"].str.extract(r"/(\d{4})/(\d{2})/(\d{2})/")
        out["article_year"] = pd.to_numeric(date_match[0], errors="coerce")
        out["article_month"] = pd.to_numeric(date_match[1], errors="coerce")
        out["article_day"] = pd.to_numeric(date_match[2], errors="coerce")

        # Imputación: reemplaza valores faltantes totales o parciales
        for c, default in [("article_year", 2013), ("article_month", 1), ("article_day", 1)]:
            n_nan = int(out[c].isna().sum())
            if out[c].isna().all():
                out[c] = default
                logger.info(f"ℹ️ '{c}': todo NaN → default {default}")
            elif n_nan > 0:
                moda = out[c].mode().iloc[0]
                out[c].fillna(moda, inplace=True)
                logger.info(f"ℹ️ '{c}': {n_nan} NaN → imputados con moda {moda}")

        # Derivación del título del artículo desde la URL
        out["article_title"] = out["url_cleaned"].str.split("/").str[-2]
        out["article_title"] = out["article_title"].str.replace("-", " ", regex=False).str.title()
        logger.info("✔️ article_title creado/normalizado")

        return out

    def _derive_from_url(self) -> None:
        """Identifica las nuevas columnas derivadas tras procesar la URL."""
        prev = set(self.df_clean.columns)
        self.df_clean = self._apply_raw_features(self.df_clean)
        new_cols = sorted(list(set(self.df_clean.columns) - prev))
        logger.info(f"🧩 Derivadas desde url: nuevas columnas {new_cols}")

    def _normalize_booleans(self) -> None:
        """
        Normaliza todas las columnas booleanas detectadas:
        convierte valores diferentes de cero en 1 y los demás en 0.
        """
        touched = []
        for c in self.boolean_cols:
            if c in self.df_clean.columns:
                before = self.df_clean[c].unique()
                self.df_clean[c] = (self.df_clean[c] > 0).astype(int)
                touched.append(c)
                logger.info(f"↔️ '{c}' → 0/1 (antes {before} | ahora {self.df_clean[c].unique()})")
        if not touched:
            logger.info("ℹ️ No se encontraron columnas booleanas a normalizar.")

    def _build_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa las variables predictoras (X) y la variable objetivo (y).
        Elimina del conjunto de features las columnas no predictoras definidas.
        """
        if self.target_col not in self.df_clean.columns:
            raise ValueError(f"No se encontró el target '{self.target_col}'.")
        drop_cols = [self.target_col] + [c for c in self.non_predictors if c in self.df_clean.columns]
        X = self.df_clean.drop(columns=drop_cols, errors="ignore")
        y = self.df_clean[self.target_col].copy()
        logger.info(f"✂️ Columnas fuera de X: {[c for c in drop_cols if c in self.df_clean.columns]}")
        logger.info(f"📐 Shapes → X: {X.shape} | y: {y.shape}")
        return X, y

    def _split_train_test(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Divide los datos en entrenamiento y prueba (por defecto 80/20)."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        logger.info(f"🔀 Split → train: {self.X_train.shape} | test: {self.X_test.shape}")

    def _build_column_transformer(self) -> None:
        """
        Construye el ColumnTransformer final para las transformaciones del modelo.
        - Aplica Yeo-Johnson + MinMaxScaler a columnas numéricas transformables.
        - Mantiene sin cambios las columnas numéricas binarias o constantes.
        - Mantiene sin cambios las columnas booleanas.
        """
        # Identificación de columnas booleanas presentes en X
        self.bool_cols_present = [c for c in self.boolean_cols if c in self.X_train.columns]

        # Detección de columnas numéricas (excluyendo las booleanas)
        num_all = [c for c in self.X_train.select_dtypes(include=[np.number]).columns
                   if c not in self.bool_cols_present]

        # Clasificación entre columnas transformables y passthrough
        self.transformable_num, self.passthrough_num = [], []
        for c in num_all:
            vals = self.X_train[c].dropna()
            if vals.nunique() > 2 and vals.std() > 0:
                self.transformable_num.append(c)
            else:
                self.passthrough_num.append(c)

        # Pipeline para transformar columnas numéricas
        num_pipe = Pipeline([
            ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("minmax", MinMaxScaler())
        ])

        # Definición del ColumnTransformer global
        self.preprocess = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.transformable_num),
                ("num_pt", "passthrough", self.passthrough_num),
                ("bool", "passthrough", self.bool_cols_present),
            ],
            remainder="drop",  # se eliminan columnas no especificadas
        )

        logger.info(
            f"🧱 ColumnTransformer → num_transform={self.transformable_num} | "
            f"num_passthrough={self.passthrough_num} | bool={self.bool_cols_present}"
        )
