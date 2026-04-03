import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.config.settings import CV_FOLDS, MODELS_DIR, RANDOM_STATE


DELIVERY_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    "KNNRegressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
}


def train_and_validate_delivery_models(
    X_train,
    y_train,
    models=None,
    cv_folds=None,
    save_models=True,
    models_dir=None,
):
    """
    Entrena modelos de regresion y los valida con KFold cross-validation.
    Retorna modelos entrenados y resultados de validacion.
    """
    models = models or DELIVERY_MODELS
    cv_folds = cv_folds or CV_FOLDS
    models_dir = models_dir or os.path.join(MODELS_DIR, "delivery")

    os.makedirs(models_dir, exist_ok=True)

    cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    trained_models = {}
    cv_results = {}

    for name, model in models.items():
        print(f"  Entrenando {name}...")

        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=scoring_metrics,
            return_train_score=True,
        )

        test_rmse_scores = np.sqrt(-cv_scores["test_mse"])
        train_rmse_scores = np.sqrt(-cv_scores["train_mse"])

        cv_results[name] = {
            "cv_mae_mean": -cv_scores["test_mae"].mean(),
            "cv_mae_std": cv_scores["test_mae"].std(),
            "cv_rmse_mean": test_rmse_scores.mean(),
            "cv_rmse_std": test_rmse_scores.std(),
            "cv_r2_mean": cv_scores["test_r2"].mean(),
            "cv_r2_std": cv_scores["test_r2"].std(),
            "train_mae_mean": -cv_scores["train_mae"].mean(),
            "train_rmse_mean": train_rmse_scores.mean(),
            "train_r2_mean": cv_scores["train_r2"].mean(),
        }

        model.fit(X_train, y_train)
        trained_models[name] = model

        if save_models:
            joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))

    cv_results_df = pd.DataFrame(cv_results).T.sort_values("cv_rmse_mean")
    cv_results_df["cv_mae_std"] = cv_results_df["cv_mae_std"].abs()
    cv_results_df["cv_rmse_std"] = cv_results_df["cv_rmse_std"].abs()

    return trained_models, cv_results_df
