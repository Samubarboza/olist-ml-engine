import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from src.config.settings import RESULTS_DIR


def evaluate_model(model, X_test, y_test):
    """
    Evalua un modelo individual. Retorna metricas, matriz de confusion,
    classification report, predicciones y probabilidades.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # probabilidades para curvas ROC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    return metrics, cm, report, y_pred, y_proba


def evaluate_all_models(trained_models, X_test, y_test, output_csv=None):
    """
    Evalua todos los modelos entrenados en el test set.
    Guarda metricas en CSV y retorna resultados completos.
    """
    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "churn_metrics.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_metrics = {}
    all_confusion_matrices = {}
    all_reports = {}
    all_predictions = {}
    all_probabilities = {}

    for name, model in trained_models.items():
        metrics, cm, report, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics
        all_confusion_matrices[name] = cm
        all_reports[name] = report
        all_predictions[name] = y_pred
        all_probabilities[name] = y_proba

    # guardar metricas como CSV ordenado por F1
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df = metrics_df.sort_values(by="f1", ascending=False)
    metrics_df.to_csv(output_csv)

    print("  Resultados de evaluacion en test set:")
    print(metrics_df.to_string())

    return metrics_df, all_confusion_matrices, all_reports, all_predictions, all_probabilities


def get_feature_importance(trained_models, feature_names):
    """
    Extrae feature importance de los modelos que lo soportan.
    Para modelos basados en arboles usa feature_importances_.
    Para modelos lineales usa el valor absoluto de los coeficientes.
    """
    importance_data = {}

    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            importance_data[name] = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            importance_data[name] = pd.Series(
                np.abs(model.coef_[0]), index=feature_names
            ).sort_values(ascending=False)

    return importance_data
