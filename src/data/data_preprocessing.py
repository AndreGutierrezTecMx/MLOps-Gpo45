"""
Preprocessor
-----------
Clase que implementa el flujo de **preprocesamiento** del dataset y deja listo
el **pipeline de Scikit-Learn** (ColumnTransformer) para usar en modelado.

Automatiza:
1) Feature engineering a partir de 'url' (article_year, article_month, article_day, article_title).
2) Normalización de columnas booleanas a 0/1.
3) Separación en X/y y **train/test split** reproducible.
4) Construcción de un **ColumnTransformer**:
   - PowerTransformer (Yeo-Johnson) + MinMaxScaler para numéricas transformables.
   - Passthrough para numéricas binarias/constantes.
   - Passthrough para columnas booleanas.

Reproducibilidad:
- El split usa `random_state` fijo (por defecto 42).
- Las transformaciones no dependen del orden de las columnas.

Entradas esperadas:
- `df_clean`: DataFrame limpio (idealmente proveniente de una etapa de limpieza previa).
- Debe existir el target (por defecto: 'shares') y la columna 'url'.

Salidas:
- `get_splits()` → (X_train, X_test, y_train, y_test)
- `get_preprocess()` → ColumnTransformer listo para incrustar en un Pipeline
- `get_feature_groups()` → columnas usadas por cada sub-transformación (trazabilidad)

Uso mínimo:
    pre = Preprocessor(df_clean=df, target_col="shares").run()
    X_train, X_test, y_train, y_test = pre.get_splits()
    preprocess = pre.get_preprocess()

Notas:
- Si no hay fecha en la URL, se imputan valores por **default** o por **moda**.
- Las columnas especificadas en `non_predictors` se excluyen de X.
- Los logs registran cada operación para auditoría.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Encapsula el flujo de preprocesamiento para mantenerlo **claro, reusable y trazable**.

    Args:
        df_clean (pd.DataFrame): DataFrame limpio/origen con al menos 'url' y el target.
        target_col (str): Nombre de la columna objetivo (default: "shares").
        boolean_cols (List[str] | None): Lista de columnas booleanas conocidas (0/1).
        non_predictors (List[str] | None): Columnas a excluir explícitamente de X.
        test_size (float): Proporción para el conjunto de prueba (default: 0.2).
        random_state (int): Semilla para reproducibilidad (default: 42).

    Raises:
        ValueError: Si faltan columnas críticas (p.ej. 'url' o el target).
    """

    def __init__(
        self,
        df_clean: pd.DataFrame,
        target_col: str = "shares",
        boolean_cols: List[str] = None,
        non_predictors: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        # Copia defensiva para no mutar el DataFrame original
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

        # Placeholders de salida/estado interno
        self.preprocess: ColumnTransformer = None
        self.bool_cols_present: List[str] = []
        self.transformable_num: List[str] = []
        self.passthrough_num: List[str] = []
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # ---------- MÉTODO PRINCIPAL ----------
    def run(self) -> "Preprocessor":
        """
        Ejecuta el flujo completo de preprocesamiento:
          (a) Derivación de variables desde la URL.
          (b) Normalización de booleanas.
          (c) Construcción de X/y.
          (d) Train/Test split reproducible.
          (e) ColumnTransformer final.
        """
        logger.info("⏳ [Preprocessor] Iniciando preprocesamiento…")
        self._derive_from_url()
        self._normalize_booleans()
        X, y = self._build_xy()
        self._split_train_test(X, y)
        self._build_column_transformer()
        return self

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Devuelve los conjuntos de entrenamiento y prueba: (X_train, X_test, y_train, y_test)."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_preprocess(self) -> ColumnTransformer:
        """Devuelve el ColumnTransformer construido para integrarlo a un Pipeline de modelado."""
        return self.preprocess

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Devuelve los grupos de columnas detectadas (útil para inspección y trazabilidad):
            - boolean_present: columnas booleanas incluidas.
            - numeric_transform: numéricas con Yeo-Johnson + MinMaxScaler.
            - numeric_passthrough: numéricas que pasan sin transformación.
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
        Deriva columnas desde 'url' y genera variables temporales y de título.

        Operaciones:
            - Limpieza de 'url' → 'url_cleaned'
            - Extracción de fecha con regex → article_year/month/day
            - Imputación:
                * Si toda la columna es NaN, se fija un default robusto.
                * Si hay NaNs parciales, se imputan con la **moda**.
            - Derivación de 'article_title' desde el penúltimo segmento de la URL.

        Returns:
            pd.DataFrame: DataFrame con columnas derivadas añadidas.
        """
        out = df.copy()
        if "url" not in out.columns:
            raise ValueError("Se requiere la columna 'url' para derivar article_year/month/day/title.")

        # Limpieza básica de la columna URL
        out["url_cleaned"] = out["url"].astype(str).str.strip()
        logger.info("✔️ url_cleaned creada")

        # Extracción de componentes de fecha con patrón YYYY/MM/DD
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

        # Derivación del título del artículo desde la URL (penúltimo segmento)
        out["article_title"] = out["url_cleaned"].str.split("/").str[-2]
        out["article_title"] = out["article_title"].str.replace("-", " ", regex=False).str.title()
        logger.info("✔️ article_title creado/normalizado")

        return out

    def _derive_from_url(self) -> None:
        """Ejecuta el feature engineering a partir de 'url' y registra las nuevas columnas creadas."""
        prev = set(self.df_clean.columns)
        self.df_clean = self._apply_raw_features(self.df_clean)
        new_cols = sorted(list(set(self.df_clean.columns) - prev))
        logger.info(f"🧩 Derivadas desde url: nuevas columnas {new_cols}")

    def _normalize_booleans(self) -> None:
        """
        Normaliza todas las columnas booleanas detectadas:
        cualquier valor > 0 se mapea a 1; el resto a 0 (tipo entero).
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
        Construye X (features) e y (target) y excluye las columnas listadas en `non_predictors`.

        Raises:
            ValueError: Si `target_col` no existe en el DataFrame.
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
        """
        Aplica el **train/test split** reproducible (por defecto 80/20, random_state fijo).
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        logger.info(f"🔀 Split → train: {self.X_train.shape} | test: {self.X_test.shape}")

    def _build_column_transformer(self) -> None:
        """
        Construye el **ColumnTransformer** final:
          - `num`: Yeo-Johnson + MinMaxScaler para numéricas con variación real (más de 2 valores y std>0).
          - `num_pt`: passthrough para numéricas binarias/constantes (evita transformar ruido).
          - `bool`: passthrough para booleanas 0/1.
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

        # Pipeline para transformar columnas numéricas (robusto a negativos via Yeo-Johnson)
        num_pipe = Pipeline([
            ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("minmax", MinMaxScaler())
        ])

        # Definición del ColumnTransformer global (listo para integrarse en un Pipeline de modelado)
        self.preprocess = ColumnTransformer(
            transformers=[
                ("num",   num_pipe,              self.transformable_num),
                ("num_pt","passthrough",         self.passthrough_num),
                ("bool",  "passthrough",         self.bool_cols_present),
            ],
            remainder="drop",  # elimina cualquier columna no listada arriba
        )

        logger.info(
            f"🧱 ColumnTransformer → num_transform={self.transformable_num} | "
            f"num_passthrough={self.passthrough_num} | bool={self.bool_cols_present}"
        )

