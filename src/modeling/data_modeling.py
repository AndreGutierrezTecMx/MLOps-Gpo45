"""
ModelTrainer
------------
Entrena los 4 modelos del notebook con el MISMO preprocesamiento:
  - HistGradientBoosting (Poisson)         [sin TTR]
  - Ridge (con TransformedTargetRegressor log1p/expm1)
  - RandomForest (con TTR, rápido)
  - XGBoost (con TTR, rápido)

Devuelve métricas en escala real (RMSE, MAE, R2) y permite elegir el mejor por R².
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

# Import limpio de XGBoost
try:
    from xgboost import XGBRegressor  # pip install xgboost
except Exception:
    XGBRegressor = None  # Si no está instalado, lo omitimos con warning en fit_xgboost_fast()

from scipy.stats import randint, uniform, loguniform
import joblib
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        preprocess,                         # ColumnTransformer ya armado
        X_train, X_test, y_train, y_test,   # splits listos
        cv_splits: int = 5,
        random_state: int = 42,
    ):
        self.preprocess = preprocess
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_models: Dict[str, Any] = {}

        # Poisson necesita conteos ≥ 0
        if (self.y_train < 0).any() or (self.y_test < 0).any():
            raise ValueError("La pérdida Poisson requiere y >= 0.")

        if XGBRegressor is None:
            logger.warning("⚠️ xgboost no importado; se omitirá fit_xgboost_fast() si se invoca.")

    # ---------- utilidades ----------
    @staticmethod
    def eval_real(y_true, y_pred) -> Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def _make_pipe_ttr(self, estimator) -> Pipeline:
        return Pipeline(steps=[
            ("prep", self.preprocess),
            ("model", TransformedTargetRegressor(
                regressor=estimator,
                func=np.log1p, inverse_func=np.expm1
            )),
        ])

    # ---------- modelos ----------
    def fit_hgb_poisson(self) -> Tuple[Any, Dict[str, float]]:
        logger.info("⏳ Entrenando HistGradientBoosting (Poisson)…")
        hgb = HistGradientBoostingRegressor(
            loss="poisson",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
            max_bins=255,
            random_state=42,
        )
        pipe = Pipeline([("prep", self.preprocess), ("model", hgb)])

        param_space = {
            "model__learning_rate": loguniform(0.01, 0.12),
            "model__max_iter": randint(200, 600),
            "model__max_depth": randint(3, 10),
            "model__max_leaf_nodes": randint(24, 140),
            "model__min_samples_leaf": randint(15, 120),
            "model__l2_regularization": loguniform(1e-9, 1e-2),
        }

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
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ HGB-Poisson best_params={search.best_params_} | best_CV_MAE={-search.best_score_:.4f}")

        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)
        logger.info(f"📊 HGB-Poisson test → {metrics}")

        self.best_models["HistGradientBoosting (Poisson)"] = best
        self.results["HistGradientBoosting (Poisson)"] = metrics
        return best, metrics

    def fit_ridge(self) -> Tuple[Any, Dict[str, float]]:
        logger.info("⏳ Entrenando Ridge (con TTR log1p/expm1)…")
        ridge = Ridge()  # Ridge no tiene random_state
        pipe = self._make_pipe_ttr(ridge)

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
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ Ridge best_params={search.best_params_} | best_CV_R2={search.best_score_:.4f}")

        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)
        logger.info(f"📊 Ridge test → {metrics}")

        self.best_models["Ridge (tuned)"] = best
        self.results["Ridge (tuned)"] = metrics
        return best, metrics

    def fit_random_forest_fast(self) -> Tuple[Any, Dict[str, float]]:
        logger.info("⏳ Entrenando RandomForest (rápido, con TTR)…")
        rf = RandomForestRegressor(n_jobs=-1, random_state=42)
        pipe = self._make_pipe_ttr(rf)

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
            cv=2,                   # rápido
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ RF best_params={search.best_params_} | best_CV_R2={search.best_score_:.4f}")

        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)
        logger.info(f"📊 RF test → {metrics}")

        self.best_models["RandomForest (tuned fast)"] = best
        self.results["RandomForest (tuned fast)"] = metrics
        return best, metrics

    def fit_xgboost_fast(self) -> Tuple[Any, Dict[str, float]]:
        if XGBRegressor is None:
            raise ImportError("xgboost no está instalado; instálalo para usar XGBRegressor.")
        logger.info("⏳ Entrenando XGBoost (rápido, con TTR)…")

        xgb = XGBRegressor(tree_method="hist", random_state=42, n_jobs=-1)
        pipe = self._make_pipe_ttr(xgb)

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
            cv=2,                    # rápido
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(self.X_train, self.y_train)
        logger.info(f"✔️ XGB best_params={search.best_params_} | best_CV_R2={search.best_score_:.4f}")

        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)
        logger.info(f"📊 XGB test → {metrics}")

        self.best_models["XGBoost (tuned fast)"] = best
        self.results["XGBoost (tuned fast)"] = metrics
        return best, metrics
