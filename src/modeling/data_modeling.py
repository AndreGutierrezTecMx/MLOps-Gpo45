"""
ModelTrainer
------------
Entrena cuatro modelos con el MISMO preprocesamiento recibido (ColumnTransformer):
  - HistGradientBoosting (Poisson)         [sin TTR]
  - Ridge (con TransformedTargetRegressor log1p/expm1)
  - RandomForest (con TTR, búsqueda rápida)
  - XGBoost (con TTR, búsqueda rápida)

Devuelve métricas en escala real (RMSE, MAE, R2) y permite elegir el mejor por R².
Las búsquedas usan KFold reproducible; los logs dejan traza de mejores parámetros y métricas.

Notas:
- El preprocesamiento debe ser el mismo para todos los modelos (consistencia de evaluación).
- La pérdida Poisson requiere y >= 0 (conteos no negativos).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from xgboost import XGBRegressor
from scipy.stats import randint, uniform, loguniform
from utils.logger import get_logger
from versioning.version_tracker import VersionTracker

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        preprocess,                         # ColumnTransformer ya armado (mismo usado para todos los modelos)
        X_train, X_test, y_train, y_test,   # Conjuntos de entrenamiento y prueba previamente separados
        version_tracker: VersionTracker,                    # Para registrar versiones
        cv_splits: int = 5,                 # Número de folds para validación cruzada
        random_state: int = 42              # Semilla de aleatoriedad para reproducibilidad
    ):
        # Guarda referencias a datos y preprocesador para reutilizarlos en cada modelo
        self.preprocess = preprocess
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        # K-Fold reproducible (shuffle + random_state) para las búsquedas aleatorias
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        # Estructuras para registrar resultados y mejores estimadores (por nombre de modelo)
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_models: Dict[str, Any] = {}

        # Para registrar versiones
        self.version_tracker = version_tracker
        version_tracker._set_train_data(X_train, y_train)
        version_tracker._set_test_data(X_test, y_test)

        # Requisito de Poisson: el objetivo (conteos) debe ser no negativo
        if (self.y_train < 0).any() or (self.y_test < 0).any():
            raise ValueError("La pérdida Poisson requiere y >= 0.")

        # Aviso amigable si xgboost no está disponible
        if XGBRegressor is None:
            logger.warning("⚠️ xgboost no importado; se omitirá fit_xgboost_fast() si se invoca.")

    # ---------- utilidades ----------
    @staticmethod
    def eval_real(y_true, y_pred) -> Dict[str, float]:
        """
        Calcula métricas en escala real del target:
          - RMSE: raíz del error cuadrático medio
          - MAE: error absoluto medio
          - R2 : coeficiente de determinación
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def _make_pipe_ttr(self, estimator) -> Pipeline:
        """
        Construye un pipeline con:
          - preprocess (ColumnTransformer) compartido
          - TransformedTargetRegressor aplicando log1p/expm1 sobre y
        Este esquema se usa para modelos que no modelan conteos de forma nativa.
        """
        return Pipeline(steps=[
            ("prep", self.preprocess),
            ("model", TransformedTargetRegressor(
                regressor=estimator,
                func=np.log1p, inverse_func=np.expm1
            )),
        ])

    # ---------- modelos ----------
    def fit_hgb_poisson(self) -> Tuple[Any, Dict[str, float]]:
        """
        Entrena HistGradientBoostingRegressor con pérdida Poisson (sin TTR).
        Realiza una búsqueda aleatoria de hiperparámetros y evalúa en test.
        """
        logger.info("⏳ Entrenando HistGradientBoosting (Poisson)…")

        # Estimador base adecuado para datos de conteo (loss='poisson')
        hgb = HistGradientBoostingRegressor(
            loss="poisson",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
            max_bins=255,
            random_state=42,
        )

        # Pipeline: preprocesador + modelo (sin TTR)
        pipe = Pipeline([("prep", self.preprocess), ("model", hgb)])

        # Espacio de hiperparámetros (compacto pero expresivo para Poisson)
        param_space = {
            "model__learning_rate": loguniform(0.01, 0.12),
            "model__max_iter": randint(200, 600),
            "model__max_depth": randint(3, 10),
            "model__max_leaf_nodes": randint(24, 140),
            "model__min_samples_leaf": randint(15, 120),
            "model__l2_regularization": loguniform(1e-9, 1e-2),
        }

        # RandomizedSearchCV con validación cruzada; se optimiza MAE (negativo en sklearn)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_space,
            n_iter=40,
            scoring="neg_mean_absolute_error",
            cv=self.cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        # 1) Búsqueda y ajuste en entrenamiento
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ HGB-Poisson best_params={search.best_params_} | best_CV_MAE={-search.best_score_:.4f}")

        # 2) Extracción del mejor pipeline y predicción en test
        best = search.best_estimator_                # Pipeline(prep + modelo) ya ajustado con best_params
        preds = best.predict(self.X_test)            # Predicciones en el conjunto de prueba
        metrics = self.eval_real(self.y_test, preds) # RMSE/MAE/R2 sobre test
        logger.info(f"📊 HGB-Poisson test → {metrics}")

        # 3) Registro de artefactos en memoria (útil para MLflow fuera de esta clase)
        self.best_models["HistGradientBoosting (Poisson)"] = best
        self.results["HistGradientBoosting (Poisson)"] = metrics
        self.version_tracker.track_mlflow_change("HistGradientBoosting (Poisson)", best, search.best_params_, metrics)
        return best, metrics

    def fit_ridge(self) -> Tuple[Any, Dict[str, float]]:
        """
        Entrena una regresión Ridge envuelta en TransformedTargetRegressor (log1p/expm1).
        Búsqueda aleatoria sobre alpha y evaluación en test.
        """
        logger.info("⏳ Entrenando Ridge (con TTR log1p/expm1)…")

        # Estimador base Ridge (sin random_state)
        ridge = Ridge()
        pipe = self._make_pipe_ttr(ridge)  # prep + TTR(log1p/expm1) + Ridge

        # Búsqueda aleatoria enfocada en el parámetro de regularización
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions={"model__regressor__alpha": loguniform(1e-4, 1e3)},
            n_iter=30,
            scoring="r2",
            cv=self.cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        # 1) Búsqueda/ajuste en entrenamiento
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ Ridge best_params={search.best_params_} | best_CV_R2={search.best_score_:.4f}")

        # 2) Predicción y métricas en test
        best = search.best_estimator_                # Pipeline completo con mejores hiperparámetros
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)
        logger.info(f"📊 Ridge test → {metrics}")

        # 3) Persistencia en las estructuras de la clase
        self.best_models["Ridge (tuned)"] = best
        self.results["Ridge (tuned)"] = metrics
        self.version_tracker.track_mlflow_change("Ridge (tuned)", best, search.best_params_, metrics)
        return best, metrics

    def fit_random_forest_fast(self) -> Tuple[Any, Dict[str, float]]:
        """
        Entrena RandomForest con TTR (búsqueda rápida).
        Reduce tiempo usando cv=2 e hiperparámetros acotados.
        """
        logger.info("⏳ Entrenando RandomForest (rápido, con TTR)…")

        # Estimador base RandomForest (paralelizado)
        rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        pipe = self._make_pipe_ttr(rf)  # prep + TTR + RF

        # Espacio de hiperparámetros compacto para acelerar iteraciones
        param_space = {
            "model__regressor__n_estimators": randint(100, 300),
            "model__regressor__max_depth": randint(4, 12),
            "model__regressor__min_samples_split": randint(2, 8),
            "model__regressor__min_samples_leaf": randint(1, 5),
            "model__regressor__max_features": ["sqrt", 0.5, 0.7],
            "model__regressor__bootstrap": [True],
            "model__regressor__max_samples": [0.5, 0.7],
        }

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_space,
            n_iter=19,
            scoring="r2",
            cv=2,                   # CV reducido para acelerar (trade-off entre tiempo y estabilidad)
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        # 1) Búsqueda y ajuste
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ RF best_params={search.best_params_} | best_CV_R2={search.best_score_:.4f}")

        # 2) Evaluación en test con el mejor pipeline
        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)
        logger.info(f"📊 RF test → {metrics}")

        # 3) Registro en estructuras internas
        self.best_models["RandomForest (tuned fast)"] = best
        self.results["RandomForest (tuned fast)"] = metrics
        self.version_tracker.track_mlflow_change("RandomForest (tuned fast)", best, search.best_params_, metrics)
        return best, metrics

    def fit_xgboost_fast(self) -> Tuple[Any, Dict[str, float]]:
        """
        Entrena XGBoost con TTR (búsqueda rápida).
        Requiere que el paquete xgboost esté instalado.
        """
        if XGBRegressor is None:
            raise ImportError("xgboost no está instalado; instálalo para usar XGBRegressor.")
        logger.info("⏳ Entrenando XGBoost (rápido, con TTR)…")

        # Estimador base XGBoost (modo 'hist' para velocidad y n_jobs para paralelizar)
        xgb = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1)
        pipe = self._make_pipe_ttr(xgb)  # prep + TTR + XGB

        # Espacio de hiperparámetros (rápido) centrado en los knobs más influyentes
        param_space = {
            "model__regressor__n_estimators": randint(300, 800),
            "model__regressor__max_depth": randint(4, 9),
            "model__regressor__learning_rate": uniform(0.03, 0.04),
            "model__regressor__subsample": uniform(0.7, 0.2),
            "model__regressor__colsample_bytree": uniform(0.7, 0.2),
            "model__regressor__min_child_weight": randint(1, 5),
            "model__regressor__gamma": uniform(0.0, 0.2),
            "model__regressor__reg_lambda": uniform(0.8, 2.0),
            "model__regressor__reg_alpha": uniform(0.0, 0.3),
        }

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_space,
            n_iter=10,
            scoring="r2",
            cv=2,                    # CV reducido para rapidez
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        # 1) Ejecuta la búsqueda y ajusta el pipeline
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ XGB best_params={search.best_params_} | best_CV_R2={search.best_score_:.4f}")

        # 2) Recupera el mejor pipeline, predice en test y calcula métricas reales
        best = search.best_estimator_                # Pipeline completo listo para .predict()
        preds = best.predict(self.X_test)            # Predicciones en el conjunto de prueba
        metrics = self.eval_real(self.y_test, preds) # RMSE/MAE/R2 en escala original del target
        logger.info(f"📊 XGB test → {metrics}")

        # 3) Guarda el mejor estimador y sus métricas para consumir fuera (p.ej., MLflow)
        self.best_models["XGBoost (tuned fast)"] = best
        self.results["XGBoost (tuned fast)"] = metrics
        self.version_tracker.track_mlflow_change("XGBoost (tuned fast)", best, search.best_params_, metrics)
        return best, metrics