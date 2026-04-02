import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from src.config.settings import FIGURES_DIR


def plot_metrics_comparison(metrics_df, output_dir=None):
    """Grafica comparacion de metricas entre modelos."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    df = metrics_df.reset_index().rename(columns={"index": "Model"})
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.barplot(data=df, x="Model", y=metric, ax=ax, color="steelblue")
        ax.set_title(f"{metric.capitalize()} por modelo")
        ax.set_ylabel(metric.capitalize())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylim(0, 1)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8)

    plt.suptitle("Metricas de evaluacion en Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    plt.close()


def plot_confusion_matrices(confusion_matrices, output_dir=None):
    """Grafica matrices de confusion para cada modelo."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(confusion_matrices)
    cols = 4
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=axes[idx]
        )
        axes[idx].set_title(name, fontsize=10)
        axes[idx].set_ylabel("Real")
        axes[idx].set_xlabel("Prediccion")

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Matrices de Confusion", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close()


def plot_roc_curves(trained_models, X_test, y_test, output_dir=None):
    """Grafica curvas ROC para todos los modelos."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))

    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC - Modelos de Churn")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_feature_importance(importance_data, top_n=15, output_dir=None):
    """Grafica feature importance para los modelos que lo soportan."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    for name, importance in importance_data.items():
        top_features = importance.head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_features.values, y=top_features.index, orient="h", color="steelblue")
        plt.title(f"Top {top_n} Features - {name}")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_{name}.png"), dpi=150)
        plt.close()


def plot_cv_results(cv_results_df, output_dir=None):
    """Grafica resultados de cross-validation con barras de error."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1"]
    models = cv_results_df.index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        means = cv_results_df[f"cv_{metric}_mean"]
        stds = cv_results_df[f"cv_{metric}_std"]

        axes[idx].barh(models, means, xerr=stds, capsize=5, color="steelblue", alpha=0.8)
        axes[idx].set_title(f"Cross-Validation {metric.capitalize()}")
        axes[idx].set_xlabel(metric.capitalize())
        axes[idx].set_xlim(0, 1)

    plt.suptitle("Resultados de Cross-Validation (5-Fold)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_results.png"), dpi=150)
    plt.close()
