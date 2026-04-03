import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import RESULTS_DIR


def evaluate_delivery_models(trained_models, X_test, y_test, output_csv=None):
    """
    Evalua todos los modelos de delivery en el test set.
    Retorna metricas ordenadas por RMSE y predicciones por modelo.
    """
    output_csv = output_csv or os.path.join(RESULTS_DIR, "delivery_metrics.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_metrics = {}
    all_predictions = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        all_metrics[name] = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }
        all_predictions[name] = pd.DataFrame(
            {
                "actual_delivery_days": y_test,
                "predicted_delivery_days": y_pred,
                "residual": residuals,
            },
            index=X_test.index,
        )

    metrics_df = pd.DataFrame(all_metrics).T.sort_values("rmse")
    metrics_df.to_csv(output_csv)

    print("  Resultados de evaluacion en test set:")
    print(metrics_df.to_string())

    return metrics_df, all_predictions


def get_delivery_feature_importance(trained_models, feature_names):
    """
    Extrae feature importance para modelos que lo soportan.
    """
    importance_data = {}

    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            importance_data[name] = pd.Series(
                model.feature_importances_,
                index=feature_names,
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            importance_data[name] = pd.Series(
                np.abs(model.coef_),
                index=feature_names,
            ).sort_values(ascending=False)

    return importance_data


def identify_orders_with_delay_risk(model, X, order_metadata, threshold_days=0.0):
    """
    Identifica pedidos con riesgo de retraso comparando el plazo estimado por el modelo
    contra el plazo prometido al cliente.
    """
    risk_data = order_metadata.copy()
    risk_data["predicted_delivery_days"] = model.predict(X)
    risk_data["delay_risk_margin"] = (
        risk_data["predicted_delivery_days"] - risk_data["promised_delivery_days"]
    )
    risk_data["delay_risk"] = (risk_data["delay_risk_margin"] > threshold_days).astype(int)
    risk_data["actual_delay_margin"] = (
        risk_data["delivery_days"] - risk_data["promised_delivery_days"]
    )

    return risk_data.sort_values(
        ["delay_risk", "delay_risk_margin"],
        ascending=[False, False],
    )
