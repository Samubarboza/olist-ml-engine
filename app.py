import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config.settings import DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR
from src.features.feature_engineering import create_churn_features

# ---------------------------------------------------------------------------
# Rutas de archivos - Churn (Bloque 1)
# ---------------------------------------------------------------------------
CHURN_DATA_PATH = os.path.join(DATA_PROCESSED, "churn_data.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "churn_metrics.csv")
CV_RESULTS_PATH = os.path.join(RESULTS_DIR, "churn_cv_results.csv")

# ---------------------------------------------------------------------------
# Rutas de archivos - Recomendacion (Bloque 2)
# ---------------------------------------------------------------------------
PRODUCT_CLUSTERS_PATH = os.path.join(DATA_PROCESSED, "product_clusters.csv")
SILHOUETTE_SCORES_PATH = os.path.join(RESULTS_DIR, "product_silhouette_scores.csv")
CLUSTER_SUMMARY_PATH = os.path.join(RESULTS_DIR, "product_cluster_summary.csv")
PCA_2D_PATH = os.path.join(DATA_PROCESSED, "product_pca_projection_2d.csv")
PCA_3D_PATH = os.path.join(DATA_PROCESSED, "product_pca_projection_3d.csv")
CUSTOMER_PREFERENCES_PATH = os.path.join(RESULTS_DIR, "customer_preference_summary.csv")
CLUSTER_PREFERENCES_PATH = os.path.join(RESULTS_DIR, "customer_cluster_preferences.csv")
CATEGORY_PREFERENCES_PATH = os.path.join(RESULTS_DIR, "customer_category_preferences.csv")
SAMPLE_RECOMMENDATIONS_PATH = os.path.join(RESULTS_DIR, "sample_recommendations.csv")

# ---------------------------------------------------------------------------
# Rutas de archivos - Delivery (Bloque 3)
# ---------------------------------------------------------------------------
DELIVERY_DATA_PATH = os.path.join(DATA_PROCESSED, "delivery_data.csv")
DELIVERY_METRICS_PATH = os.path.join(RESULTS_DIR, "delivery_metrics.csv")
DELIVERY_CV_PATH = os.path.join(RESULTS_DIR, "delivery_cv_results.csv")
DELIVERY_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "delivery_test_predictions.csv")
DELIVERY_RISK_PATH = os.path.join(RESULTS_DIR, "delivery_risk_orders.csv")


# ===================================================================
#  BLOQUE 1 - CHURN
# ===================================================================

def churn_results_exist():
    return all(os.path.exists(p) for p in [CHURN_DATA_PATH, METRICS_PATH, CV_RESULTS_PATH])


@st.cache_data
def load_churn_results():
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
    selected = st.selectbox("Feature:", numeric_cols, key="churn_feature_select")

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
            selected = st.selectbox("Modelo:", names, key="churn_fi_model")
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

    k = st.slider("Numero de clusters (k):", min_value=2, max_value=10, value=3, key="churn_kmeans_k")

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


def page_churn():
    st.title("Prediccion de Churn - Olist")

    if not churn_results_exist():
        st.warning("No se encontraron resultados del pipeline de churn.")
        if st.button("Ejecutar pipeline de churn"):
            with st.spinner("Ejecutando pipeline (esto puede tardar unos minutos)..."):
                from src.pipelines.churn_pipeline import run_churn_pipeline
                run_churn_pipeline()
            st.success("Pipeline completado.")
            st.rerun()
        return

    churn_data, metrics, cv_results = load_churn_results()

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


# ===================================================================
#  BLOQUE 2 - RECOMENDACION DE PRODUCTOS
# ===================================================================

def recommendation_results_exist():
    return all(os.path.exists(p) for p in [
        PRODUCT_CLUSTERS_PATH, SILHOUETTE_SCORES_PATH, CLUSTER_SUMMARY_PATH,
    ])


@st.cache_data
def load_recommendation_results():
    product_data = pd.read_csv(PRODUCT_CLUSTERS_PATH)
    silhouette_scores = pd.read_csv(SILHOUETTE_SCORES_PATH)
    cluster_summary = pd.read_csv(CLUSTER_SUMMARY_PATH, index_col=0)
    pca_2d = pd.read_csv(PCA_2D_PATH) if os.path.exists(PCA_2D_PATH) else None
    pca_3d = pd.read_csv(PCA_3D_PATH) if os.path.exists(PCA_3D_PATH) else None
    customer_prefs = pd.read_csv(CUSTOMER_PREFERENCES_PATH) if os.path.exists(CUSTOMER_PREFERENCES_PATH) else None
    sample_recs = pd.read_csv(SAMPLE_RECOMMENDATIONS_PATH) if os.path.exists(SAMPLE_RECOMMENDATIONS_PATH) else None
    return product_data, silhouette_scores, cluster_summary, pca_2d, pca_3d, customer_prefs, sample_recs


def rec_tab_clusters(product_data, silhouette_scores, cluster_summary):
    st.header("Clusters de Productos")

    n_clusters = product_data["cluster"].nunique()
    n_products = len(product_data)
    best_row = silhouette_scores.loc[silhouette_scores["silhouette_score"].idxmax()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total productos", f"{n_products:,}")
    col2.metric("Clusters optimos", int(best_row["k"]))
    col3.metric("Silhouette Score", f"{best_row['silhouette_score']:.4f}")

    st.subheader("Silhouette Score por k")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(silhouette_scores["k"], silhouette_scores["silhouette_score"], "o-", color="steelblue", linewidth=2)
    ax.axvline(best_row["k"], color="firebrick", linestyle="--", alpha=0.7, label=f"Mejor k={int(best_row['k'])}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(silhouette_scores["k"].astype(int))
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Resumen de Clusters")
    st.dataframe(cluster_summary.round(3))

    st.subheader("Distribucion de Productos por Cluster")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cluster_counts = product_data["cluster"].value_counts().sort_index()
    axes[0].bar(cluster_counts.index, cluster_counts.values, color="steelblue")
    axes[0].set_title("Productos por cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Cantidad")

    if "avg_review_score" in product_data.columns:
        avg_reviews = product_data.groupby("cluster")["avg_review_score"].mean().sort_index()
        axes[1].bar(avg_reviews.index, avg_reviews.values, color="coral")
        axes[1].set_title("Review promedio por cluster")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Review promedio")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def rec_tab_pca(product_data, pca_2d, pca_3d):
    st.header("Visualizacion PCA de Clusters")

    if pca_2d is not None and "PC1" in pca_2d.columns and "PC2" in pca_2d.columns:
        st.subheader("Proyeccion 2D")
        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(
            pca_2d["PC1"], pca_2d["PC2"],
            c=pca_2d["cluster"], cmap="viridis", alpha=0.55, s=14,
        )
        fig.colorbar(scatter, label="Cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Clusters de productos - PCA 2D")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No hay datos de proyeccion PCA 2D disponibles.")

    if pca_3d is not None and "PC1" in pca_3d.columns:
        st.subheader("Proyeccion 3D")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            pca_3d["PC1"], pca_3d["PC2"], pca_3d["PC3"],
            c=pca_3d["cluster"], cmap="viridis", alpha=0.45, s=12,
        )
        fig.colorbar(scatter, ax=ax, pad=0.12, label="Cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("Clusters de productos - PCA 3D")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Imagenes generadas por el pipeline
    for img_name, title in [
        ("product_pca_clusters_2d.png", "PCA 2D (pipeline)"),
        ("product_pca_clusters_3d.png", "PCA 3D (pipeline)"),
    ]:
        img_path = os.path.join(FIGURES_DIR, img_name)
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path)


def rec_tab_categories(product_data):
    st.header("Analisis por Categorias")

    if "category" not in product_data.columns:
        st.info("No hay datos de categorias disponibles.")
        return

    st.subheader("Productos por Cluster y Categoria (Top 10)")
    top_categories = product_data["category"].value_counts().head(10).index.tolist()
    filtered = product_data[product_data["category"].isin(top_categories)]
    cluster_category = pd.crosstab(filtered["cluster"], filtered["category"])

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_category.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Productos por cluster y categoria (top 10)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad")
    ax.legend(title="Categoria", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Heatmap: Distribucion de Categorias por Cluster (Top 15)")
    top_15 = product_data["category"].value_counts().head(15).index
    filtered_15 = product_data[product_data["category"].isin(top_15)]
    pivot = pd.crosstab(filtered_15["category"], filtered_15["cluster"])
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Distribucion de categorias por cluster (top 15)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Categoria")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def rec_tab_recommendations(customer_prefs, sample_recs):
    st.header("Recomendaciones de Productos")

    if customer_prefs is not None and not customer_prefs.empty:
        st.subheader("Resumen de Preferencias de Clientes")
        st.metric("Clientes analizados", f"{len(customer_prefs):,}")
        st.dataframe(customer_prefs.head(20))
    else:
        st.info("No hay datos de preferencias de clientes disponibles.")

    if sample_recs is not None and not sample_recs.empty:
        st.subheader("Recomendaciones de Ejemplo")
        st.metric("Recomendaciones generadas", f"{len(sample_recs):,}")

        if "customer_unique_id" in sample_recs.columns:
            customers = sample_recs["customer_unique_id"].unique()
            selected_customer = st.selectbox(
                "Seleccionar cliente:",
                customers,
                key="rec_customer_select",
            )
            customer_recs = sample_recs[sample_recs["customer_unique_id"] == selected_customer]
            st.dataframe(customer_recs)
        else:
            st.dataframe(sample_recs.head(20))
    else:
        st.info("No hay recomendaciones generadas. Ejecuta el pipeline primero.")


def page_recommendation():
    st.title("Recomendacion de Productos - Olist")

    if not recommendation_results_exist():
        st.warning("No se encontraron resultados del pipeline de recomendacion.")
        if st.button("Ejecutar pipeline de recomendacion"):
            with st.spinner("Ejecutando pipeline de recomendacion (esto puede tardar unos minutos)..."):
                from src.pipelines.recommendation_pipeline import run_recommendation_pipeline
                run_recommendation_pipeline()
            st.success("Pipeline completado.")
            st.rerun()
        return

    product_data, silhouette_scores, cluster_summary, pca_2d, pca_3d, customer_prefs, sample_recs = load_recommendation_results()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Clusters", "PCA", "Categorias", "Recomendaciones"]
    )

    with tab1:
        rec_tab_clusters(product_data, silhouette_scores, cluster_summary)
    with tab2:
        rec_tab_pca(product_data, pca_2d, pca_3d)
    with tab3:
        rec_tab_categories(product_data)
    with tab4:
        rec_tab_recommendations(customer_prefs, sample_recs)


# ===================================================================
#  BLOQUE 3 - ESTIMACION DE PLAZO DE ENTREGA
# ===================================================================

def delivery_results_exist():
    return all(os.path.exists(p) for p in [
        DELIVERY_DATA_PATH, DELIVERY_METRICS_PATH, DELIVERY_CV_PATH,
    ])


@st.cache_data
def load_delivery_results():
    delivery_data = pd.read_csv(DELIVERY_DATA_PATH)
    metrics = pd.read_csv(DELIVERY_METRICS_PATH, index_col=0)
    cv_results = pd.read_csv(DELIVERY_CV_PATH, index_col=0)
    predictions = pd.read_csv(DELIVERY_PREDICTIONS_PATH) if os.path.exists(DELIVERY_PREDICTIONS_PATH) else None
    risk_orders = pd.read_csv(DELIVERY_RISK_PATH) if os.path.exists(DELIVERY_RISK_PATH) else None
    return delivery_data, metrics, cv_results, predictions, risk_orders


def del_tab_datos(delivery_data):
    st.header("Dataset de Delivery")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pedidos", f"{len(delivery_data):,}")
    col2.metric("Delivery promedio", f"{delivery_data['delivery_days'].mean():.1f} dias")
    col3.metric("Tasa de retraso", f"{delivery_data['is_late'].mean():.1%}")
    col4.metric("Features", str(len(delivery_data.columns)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(delivery_data["delivery_days"], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].set_title("Distribucion de Delivery (dias)")
    axes[0].set_xlabel("Dias")
    axes[0].set_ylabel("Frecuencia")

    late_counts = delivery_data["is_late"].value_counts()
    late_counts.plot(kind="bar", ax=axes[1], color=["steelblue", "coral"])
    axes[1].set_title("A tiempo vs Retrasado")
    axes[1].set_xticklabels(["A tiempo", "Retrasado"], rotation=0)
    axes[1].set_ylabel("Cantidad")

    if "delay_days" in delivery_data.columns:
        delayed = delivery_data[delivery_data["delay_days"] > 0]["delay_days"]
        if not delayed.empty:
            axes[2].hist(delayed, bins=40, color="coral", alpha=0.7, edgecolor="white")
            axes[2].set_title("Distribucion de Retraso (dias)")
            axes[2].set_xlabel("Dias de retraso")
            axes[2].set_ylabel("Frecuencia")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Muestra de datos")
    st.dataframe(delivery_data.head(20))


def del_tab_cv(cv_results):
    st.header("Cross-Validation - Modelos de Regresion")

    cv_display_cols = [c for c in cv_results.columns if "mean" in c]
    if cv_display_cols:
        st.dataframe(cv_results[cv_display_cols].round(4))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_info = [
        ("cv_mae_mean", "cv_mae_std", "MAE"),
        ("cv_rmse_mean", "cv_rmse_std", "RMSE"),
        ("cv_r2_mean", "cv_r2_std", "R2"),
    ]

    for idx, (mean_col, std_col, title) in enumerate(metrics_info):
        if mean_col in cv_results.columns:
            axes[idx].barh(
                cv_results.index,
                cv_results[mean_col],
                xerr=cv_results.get(std_col, 0),
                color="steelblue", alpha=0.85, capsize=4,
            )
            axes[idx].set_title(f"CV {title}")
            axes[idx].set_xlabel(title)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def del_tab_evaluacion(metrics):
    st.header("Evaluacion en Test Set")

    best = metrics["rmse"].idxmin()
    st.success(f"Mejor modelo: **{best}** (RMSE = {metrics.loc[best, 'rmse']:.4f}, R2 = {metrics.loc[best, 'r2']:.4f})")

    st.dataframe(metrics.round(4))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    df_plot = metrics.reset_index().rename(columns={"index": "Model"})

    for idx, metric in enumerate(["mae", "rmse", "r2"]):
        if metric in df_plot.columns:
            sns.barplot(data=df_plot, x="Model", y=metric, ax=axes[idx], color="steelblue")
            axes[idx].set_title(metric.upper())
            axes[idx].tick_params(axis="x", rotation=45)
            for container in axes[idx].containers:
                axes[idx].bar_label(container, fmt="%.3f", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Imagenes generadas por el pipeline
    for img_name, title in [
        ("delivery_metrics_comparison.png", "Comparacion de Metricas"),
        ("delivery_cv_results.png", "Resultados CV"),
    ]:
        img_path = os.path.join(FIGURES_DIR, img_name)
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path)

    # Feature importance
    fi_files = []
    if os.path.exists(FIGURES_DIR):
        fi_files = [f for f in os.listdir(FIGURES_DIR) if f.startswith("delivery_feature_importance_")]
    if fi_files:
        st.subheader("Feature Importance")
        names = [f.replace("delivery_feature_importance_", "").replace(".png", "") for f in fi_files]
        selected = st.selectbox("Modelo:", names, key="delivery_fi_model")
        st.image(os.path.join(FIGURES_DIR, f"delivery_feature_importance_{selected}.png"))


def del_tab_predicciones(predictions):
    st.header("Predicciones: Actual vs Predicho")

    if predictions is None or predictions.empty:
        st.info("No hay predicciones disponibles. Ejecuta el pipeline primero.")
        return

    col1, col2 = st.columns(2)

    if "actual_delivery_days" in predictions.columns and "predicted_delivery_days" in predictions.columns:
        col1.metric("MAE en predicciones",
                     f"{(predictions['actual_delivery_days'] - predictions['predicted_delivery_days']).abs().mean():.2f} dias")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(
            predictions["actual_delivery_days"],
            predictions["predicted_delivery_days"],
            alpha=0.4, s=15, color="steelblue",
        )
        min_val = min(predictions["actual_delivery_days"].min(), predictions["predicted_delivery_days"].min())
        max_val = max(predictions["actual_delivery_days"].max(), predictions["predicted_delivery_days"].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        axes[0].set_xlabel("Delivery real (dias)")
        axes[0].set_ylabel("Delivery predicho (dias)")
        axes[0].set_title("Predicho vs Real")

        if "residual" in predictions.columns:
            sns.histplot(predictions["residual"], bins=30, kde=True, ax=axes[1], color="teal")
            axes[1].set_title("Distribucion de Residuales")
            axes[1].set_xlabel("Residual (dias)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Imagenes del pipeline
    for img_name, title in [
        ("delivery_predicted_vs_actual.png", "Predicho vs Real (pipeline)"),
        ("delivery_residuals.png", "Residuales (pipeline)"),
    ]:
        img_path = os.path.join(FIGURES_DIR, img_name)
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path)


def del_tab_riesgo(risk_orders):
    st.header("Pedidos con Riesgo de Retraso")

    if risk_orders is None or risk_orders.empty:
        st.info("No hay datos de riesgo disponibles. Ejecuta el pipeline primero.")
        return

    total = len(risk_orders)
    at_risk = risk_orders["delay_risk"].sum() if "delay_risk" in risk_orders.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total pedidos analizados", f"{total:,}")
    col2.metric("Con riesgo de retraso", f"{int(at_risk):,}")
    col3.metric("Tasa de riesgo", f"{at_risk / total:.1%}" if total > 0 else "0%")

    if "delay_risk" in risk_orders.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        risk_counts = risk_orders["delay_risk"].value_counts().sort_index()
        risk_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "coral"])
        axes[0].set_title("Sin riesgo vs Con riesgo")
        axes[0].set_xticklabels(["Sin riesgo", "Con riesgo"], rotation=0)
        axes[0].set_ylabel("Cantidad")

        if "delay_risk_margin" in risk_orders.columns:
            risky = risk_orders[risk_orders["delay_risk"] == 1]
            if not risky.empty:
                axes[1].hist(risky["delay_risk_margin"], bins=30, color="coral", alpha=0.7, edgecolor="white")
                axes[1].set_title("Margen de riesgo estimado (dias)")
                axes[1].set_xlabel("Dias por encima del prometido")
                axes[1].set_ylabel("Frecuencia")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Top pedidos con mayor riesgo")
    display_cols = [c for c in ["order_id", "delivery_days", "promised_delivery_days",
                                 "predicted_delivery_days", "delay_risk_margin", "delay_risk",
                                 "is_late"] if c in risk_orders.columns]
    risky_top = risk_orders[risk_orders.get("delay_risk", pd.Series(dtype=int)) == 1]
    if not risky_top.empty:
        st.dataframe(risky_top[display_cols].head(30))
    else:
        st.info("No se detectaron pedidos con riesgo de retraso.")


def page_delivery():
    st.title("Estimacion de Plazo de Entrega - Olist")

    if not delivery_results_exist():
        st.warning("No se encontraron resultados del pipeline de delivery.")
        if st.button("Ejecutar pipeline de delivery"):
            with st.spinner("Ejecutando pipeline de delivery (esto puede tardar unos minutos)..."):
                from src.pipelines.delivery_pipeline import run_delivery_pipeline
                run_delivery_pipeline()
            st.success("Pipeline completado.")
            st.rerun()
        return

    delivery_data, metrics, cv_results, predictions, risk_orders = load_delivery_results()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Datos", "Cross-Validation", "Evaluacion", "Predicciones", "Riesgo de Retraso"]
    )

    with tab1:
        del_tab_datos(delivery_data)
    with tab2:
        del_tab_cv(cv_results)
    with tab3:
        del_tab_evaluacion(metrics)
    with tab4:
        del_tab_predicciones(predictions)
    with tab5:
        del_tab_riesgo(risk_orders)


# ===================================================================
#  MAIN - NAVEGACION
# ===================================================================

def main():
    st.set_page_config(
        page_title="Olist ML Engine",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Olist ML Engine")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Seleccionar caso de uso:",
        [
            "Prediccion de Churn",
            "Recomendacion de Productos",
            "Estimacion de Delivery",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Bloques disponibles:**\n"
        "1. Churn Prediction\n"
        "2. Product Recommendation\n"
        "3. Delivery Estimation"
    )

    if page == "Prediccion de Churn":
        page_churn()
    elif page == "Recomendacion de Productos":
        page_recommendation()
    elif page == "Estimacion de Delivery":
        page_delivery()


if __name__ == "__main__":
    main()
