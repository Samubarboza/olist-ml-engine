import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config.settings import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR
from src.features.feature_engineering import create_churn_features

CHURN_DATA_PATH = os.path.join(DATA_PROCESSED, "churn_data.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "churn_metrics.csv")
CV_RESULTS_PATH = os.path.join(RESULTS_DIR, "churn_cv_results.csv")


def results_exist():
    return all(os.path.exists(p) for p in [CHURN_DATA_PATH, METRICS_PATH, CV_RESULTS_PATH])


@st.cache_data
def load_results():
    churn_data = pd.read_csv(CHURN_DATA_PATH)
    metrics = pd.read_csv(METRICS_PATH, index_col=0)
    cv_results = pd.read_csv(CV_RESULTS_PATH, index_col=0)
    return churn_data, metrics, cv_results


@st.cache_data
def compute_features(churn_data):
    features, target, names = create_churn_features(churn_data)
    return features, target, names


def tab_datos(churn_data):
    st.header("Dataset de Churn")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total clientes", f"{len(churn_data):,}")
    col2.metric("Tasa de churn", f"{churn_data['churn'].mean():.1%}")
    col3.metric("Features", str(len(churn_data.columns) - 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    churn_counts = churn_data["churn"].value_counts()
    churn_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "coral"])
    axes[0].set_title("Distribucion de Churn")
    axes[0].set_xticklabels(["No Churn", "Churn"], rotation=0)
    axes[0].set_ylabel("Cantidad")

    churn_counts.plot(kind="pie", ax=axes[1], autopct="%.1f%%", colors=["steelblue", "coral"])
    axes[1].set_title("Proporcion")
    axes[1].set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Distribucion por clase")
    numeric_cols = churn_data.select_dtypes(include="number").columns.drop("churn")
    selected = st.selectbox("Feature:", numeric_cols)

    fig, ax = plt.subplots(figsize=(8, 4))
    churn_data[churn_data["churn"] == 0][selected].hist(bins=30, ax=ax, alpha=0.6, label="No Churn", color="steelblue")
    churn_data[churn_data["churn"] == 1][selected].hist(bins=30, ax=ax, alpha=0.6, label="Churn", color="coral")
    ax.set_title(selected)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Muestra de datos")
    st.dataframe(churn_data.head(20))


def tab_cv(cv_results):
    st.header("Cross-Validation (5-Fold)")

    cv_display = cv_results[["cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean", "cv_f1_mean"]].round(4)
    cv_display.columns = ["Accuracy", "Precision", "Recall", "F1"]
    st.dataframe(cv_display)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    models = cv_results.index.tolist()

    for idx, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        means = cv_results[f"cv_{metric}_mean"]
        stds = cv_results[f"cv_{metric}_std"]
        axes[idx].barh(models, means, xerr=stds, capsize=5, color="steelblue", alpha=0.8)
        axes[idx].set_title(f"CV {metric.capitalize()}")
        axes[idx].set_xlim(0, 1)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Control de Sobreajuste")
    overfit = pd.DataFrame({
        "Train F1": cv_results["train_f1_mean"],
        "CV F1": cv_results["cv_f1_mean"],
        "Gap": cv_results["train_f1_mean"] - cv_results["cv_f1_mean"]
    }).round(4)
    st.dataframe(overfit)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(models))
    width = 0.35
    ax.bar([i - width / 2 for i in x], overfit["Train F1"], width, label="Train", color="steelblue")
    ax.bar([i + width / 2 for i in x], overfit["CV F1"], width, label="CV", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Train vs CV - Sobreajuste")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def tab_evaluacion(metrics):
    st.header("Evaluacion en Test Set")

    best = metrics["f1"].idxmax()
    st.success(f"Mejor modelo: **{best}** (F1 = {metrics['f1'].max():.4f})")
    st.dataframe(metrics.round(4))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    df_plot = metrics.reset_index().rename(columns={"index": "Model"})

    for idx, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        sns.barplot(data=df_plot, x="Model", y=metric, ax=axes[idx], color="steelblue")
        axes[idx].set_title(metric.capitalize())
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].set_ylim(0, 1)
        for container in axes[idx].containers:
            axes[idx].bar_label(container, fmt="%.3f", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # imagenes generadas por el pipeline
    if os.path.exists(FIGURES_DIR):
        cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.subheader("Matrices de Confusion")
            st.image(cm_path)

        roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.subheader("Curvas ROC")
            st.image(roc_path)

        st.subheader("Feature Importance")
        fi_files = [f for f in os.listdir(FIGURES_DIR) if f.startswith("feature_importance_")]
        if fi_files:
            names = [f.replace("feature_importance_", "").replace(".png", "") for f in fi_files]
            selected = st.selectbox("Modelo:", names)
            st.image(os.path.join(FIGURES_DIR, f"feature_importance_{selected}.png"))


def tab_pca(churn_data):
    st.header("PCA - Reduccion de Dimensionalidad")

    features, target, _ = compute_features(churn_data)

    pca_full = PCA(random_state=42)
    pca_full.fit(features)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    n_95 = int(np.argmax(cumulative >= 0.95) + 1)
    n_90 = int(np.argmax(cumulative >= 0.90) + 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Features originales", features.shape[1])
    col2.metric("Componentes (90%)", n_90)
    col3.metric("Componentes (95%)", n_95)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_, color="steelblue", alpha=0.7)
    axes[0].set_title("Varianza por componente")
    axes[0].set_xlabel("Componente")

    axes[1].plot(range(1, len(cumulative) + 1), cumulative, "bo-", markersize=3)
    axes[1].axhline(y=0.95, color="r", linestyle="--", label="95%")
    axes[1].axhline(y=0.90, color="orange", linestyle="--", label="90%")
    axes[1].set_title("Varianza acumulada")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Proyeccion 2D")
    pca_2d = PCA(n_components=2, random_state=42)
    f2d = pca_2d.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(f2d[:, 0], f2d[:, 1], c=target, cmap="RdYlGn_r", alpha=0.3, s=5)
    plt.colorbar(scatter, label="Churn")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Churn vs No Churn")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def tab_kmeans(churn_data):
    st.header("K-Means - Segmentacion de Clientes")

    features, target, _ = compute_features(churn_data)

    k = st.slider("Numero de clusters (k):", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    score = silhouette_score(features, labels, sample_size=5000, random_state=42)

    st.metric("Silhouette Score", f"{score:.4f}")

    churn_clusters = churn_data.copy()
    churn_clusters["cluster"] = labels

    analysis = churn_clusters.groupby("cluster").agg(
        clientes=("churn", "count"),
        tasa_churn=("churn", "mean"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary_total", "mean")
    ).round(3)
    st.dataframe(analysis)

    pca_2d = PCA(n_components=2, random_state=42)
    f2d = pca_2d.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter1 = axes[0].scatter(f2d[:, 0], f2d[:, 1], c=labels, cmap="viridis", alpha=0.3, s=5)
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")
    axes[0].set_title("Clusters (PCA 2D)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    cluster_churn = churn_clusters.groupby("cluster")["churn"].mean()
    axes[1].bar(cluster_churn.index, cluster_churn.values, color="coral")
    axes[1].set_title("Tasa de Churn por Cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Tasa de Churn")
    for i, v in enumerate(cluster_churn.values):
        axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def main():
    st.set_page_config(page_title="Olist ML - Churn", layout="wide")
    st.title("Prediccion de Churn - Olist")

    if not results_exist():
        st.warning("No se encontraron resultados del pipeline.")
        if st.button("Ejecutar pipeline de churn"):
            with st.spinner("Ejecutando pipeline (esto puede tardar unos minutos)..."):
                from src.pipelines.churn_pipeline import run_churn_pipeline
                run_churn_pipeline()
            st.success("Pipeline completado.")
            st.rerun()
        return

    churn_data, metrics, cv_results = load_results()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Datos", "Cross-Validation", "Evaluacion", "PCA", "K-Means"]
    )

    with tab1:
        tab_datos(churn_data)
    with tab2:
        tab_cv(cv_results)
    with tab3:
        tab_evaluacion(metrics)
    with tab4:
        tab_pca(churn_data)
    with tab5:
        tab_kmeans(churn_data)


if __name__ == "__main__":
    main()
