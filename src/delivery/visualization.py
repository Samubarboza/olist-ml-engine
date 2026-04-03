import os

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.settings import FIGURES_DIR


def plot_delivery_metrics_comparison(metrics_df, output_dir=None):
    """Grafica comparacion de metricas de regresion entre modelos."""
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["mae", "rmse", "r2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.barplot(
            data=metrics_df.reset_index().rename(columns={"index": "Model"}),
            x="Model",
            y=metric,
            ax=ax,
            color="steelblue",
        )
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "delivery_metrics_comparison.png"), dpi=150)
    plt.close(fig)


def plot_delivery_cv_results(cv_results_df, output_dir=None):
    """Grafica resultados de cross-validation para regresion."""
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("cv_mae_mean", "cv_mae_std", "MAE"),
        ("cv_rmse_mean", "cv_rmse_std", "RMSE"),
        ("cv_r2_mean", "cv_r2_std", "R2"),
    ]

    for idx, (mean_column, std_column, title) in enumerate(metrics):
        axes[idx].barh(
            cv_results_df.index,
            cv_results_df[mean_column],
            xerr=cv_results_df[std_column],
            color="steelblue",
            alpha=0.85,
            capsize=4,
        )
        axes[idx].set_title(f"Cross-Validation {title}")
        axes[idx].set_xlabel(title)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "delivery_cv_results.png"), dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(predictions_df, model_name, output_dir=None):
    """Grafica valores predichos vs reales para el mejor modelo."""
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        predictions_df["actual_delivery_days"],
        predictions_df["predicted_delivery_days"],
        alpha=0.45,
        s=20,
        color="steelblue",
    )

    min_value = min(
        predictions_df["actual_delivery_days"].min(),
        predictions_df["predicted_delivery_days"].min(),
    )
    max_value = max(
        predictions_df["actual_delivery_days"].max(),
        predictions_df["predicted_delivery_days"].max(),
    )
    ax.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=2)
    ax.set_xlabel("Delivery real (dias)")
    ax.set_ylabel("Delivery predicho (dias)")
    ax.set_title(f"Predicho vs real - {model_name}")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "delivery_predicted_vs_actual.png"), dpi=150)
    plt.close(fig)


def plot_residuals(predictions_df, model_name, output_dir=None):
    """Grafica residuales del mejor modelo."""
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        predictions_df["predicted_delivery_days"],
        predictions_df["residual"],
        alpha=0.45,
        s=20,
        color="coral",
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Residuales vs prediccion - {model_name}")
    axes[0].set_xlabel("Delivery predicho (dias)")
    axes[0].set_ylabel("Residual")

    sns.histplot(predictions_df["residual"], bins=30, kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Distribucion de residuales")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "delivery_residuals.png"), dpi=150)
    plt.close(fig)


def plot_delivery_feature_importance(importance_data, top_n=15, output_dir=None):
    """Grafica feature importance para modelos de entrega que lo soportan."""
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    for name, importance in importance_data.items():
        top_features = importance.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=top_features.values,
            y=top_features.index,
            orient="h",
            ax=ax,
            color="steelblue",
        )
        ax.set_title(f"Top {top_n} features - {name}")
        ax.set_xlabel("Importancia")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"delivery_feature_importance_{name}.png"), dpi=150)
        plt.close(fig)
