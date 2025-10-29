# modeling.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
# TODO: Asegúrate de que XGBoost esté en los requisitos del proyecto
from xgboost import XGBRegressor

from scipy.stats import randint, uniform, loguniform
import joblib

class ModelTrainer:
    """
    Entrena y afina modelos (Ridge, RF, XGB, HGB-Poisson) con el mismo preprocesamiento.
    Produce tabla comparativa y guarda el mejor modelo.
    """

    def __init__(
        self,
        preprocess,                         # ColumnTransformer del Preprocessor
        X_train, X_test, y_train, y_test,   # splits ya construidos
        cv_splits: int = 5,
        random_state: int = 42
    ):
        self.preprocess = preprocess
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_models: Dict[str, Any] = {}

    # ---------- Utilidades ----------

    @staticmethod
    def eval_real(y_true, y_pred) -> Dict[str, float]:
        """RMSE, MAE y R² en escala real."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def _make_pipe_ttr(self, estimator) -> Pipeline:
        """Pipeline con preprocess + TTR(log1p/expm1) para modelos que no modelan conteos directamente."""
        return Pipeline(steps=[
            ("prep", self.preprocess),
            ("model", TransformedTargetRegressor(
                regressor=estimator,
                func=np.log1p, inverse_func=np.expm1
            ))
        ])

    # ---------- Entrenamientos ----------

    def fit_hgb_poisson(self) -> Tuple[Any, Dict[str, float]]:
        """HistGradientBoosting con pérdida Poisson (sin TTR)."""
        hgb = HistGradientBoostingRegressor(
            loss="poisson",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            tol=1e-4,
            max_bins=255,
            random_state=42
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
            verbose=1
        )
        search.fit(self.X_train, self.y_train)
        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)

        self.best_models["HistGradientBoosting (Poisson)"] = best
        self.results["HistGradientBoosting (Poisson)"] = metrics
        return best, metrics

    def fit_ridge(self) -> Tuple[Any, Dict[str, float]]:
        """Ridge con TTR."""
        ridge = Ridge(random_state=42)
        pipe = self._make_pipe_ttr(ridge)

        param_space = {"model__regressor__alpha": loguniform(1e-4, 1e3)}

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_space,
            n_iter=30,
            scoring="r2",
            cv=self.cv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(self.X_train, self.y_train)
        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)

        self.best_models["Ridge (tuned)"] = best
        self.results["Ridge (tuned)"] = metrics
        return best, metrics

    def fit_random_forest_fast(self) -> Tuple[Any, Dict[str, float]]:
        """RandomForest con TTR (búsqueda rápida)."""
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
            cv=2,               # rápido
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(self.X_train, self.y_train)
        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)

        self.best_models["RandomForest (tuned fast)"] = best
        self.results["RandomForest (tuned fast)"] = metrics
        return best, metrics

    def fit_xgboost_fast(self) -> Tuple[Any, Dict[str, float]]:
        """XGBoost con TTR (búsqueda rápida)."""
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
            cv=2,               # rápido
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(self.X_train, self.y_train)
        best = search.best_estimator_
        preds = best.predict(self.X_test)
        metrics = self.eval_real(self.y_test, preds)

        self.best_models["XGBoost (tuned fast)"] = best
        self.results["XGBoost (tuned fast)"] = metrics
        return best, metrics

    # ---------- Salidas ----------

    def comparison_table(self) -> pd.DataFrame:
        """Devuelve tabla comparativa ordenada por R²."""
        if not self.results:
            raise RuntimeError("Primero entrena algún modelo.")
        df = pd.DataFrame([
            {"Modelo": k, **v} for k, v in self.results.items()
        ]).sort_values(by="R2", ascending=False).reset_index(drop=True)
        return df[["Modelo","RMSE","MAE","R2"]]

    def best_by_r2(self) -> Tuple[str, Any]:
        """Devuelve (nombre, modelo) con mayor R² en test."""
        if not self.results:
            raise RuntimeError("Primero entrena algún modelo.")
        best_name = max(self.results.items(), key=lambda x: x[1]["R2"])[0]
        return best_name, self.best_models[best_name]

    def save_model(self, name: str, path: str) -> None:
        """Guarda un modelo entrenado con joblib."""
        if name not in self.best_models:
            raise ValueError(f"No se encontró un modelo con nombre '{name}'. Modelos: {list(self.best_models)}")
        joblib.dump(self.best_models[name], path) #path