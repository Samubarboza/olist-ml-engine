import os

import pandas as pd

from src.config.settings import DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR
from src.data.preprocess import build_customer_product_history, build_product_dataset
from src.features.feature_engineering import create_recommendation_features
from src.recommendation.clustering import build_cluster_summary, build_pca_projection, find_optimal_k
from src.recommendation.engine import analyze_customer_preferences, generate_sample_recommendations
from src.recommendation.visualization import plot_product_category_heatmap, plot_product_cluster_distribution, plot_product_cluster_profile_heatmap, plot_product_pca_clusters_2d, plot_product_pca_clusters_3d, plot_product_silhouette_scores, plot_products_by_cluster


def _ensure_output_directories():
    """Crea los directorios de salida del pipeline si no existen."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save_clustering_artifacts(product_data, silhouette_scores, cluster_summary, pca_2d, pca_3d):
    """Guarda artefactos generados durante la etapa de clustering."""
    product_data.to_csv(os.path.join(DATA_PROCESSED, "product_clusters.csv"), index=False)

    scores_df = pd.DataFrame(sorted(silhouette_scores.items()), columns=["k", "silhouette_score"])
    scores_df.to_csv(os.path.join(RESULTS_DIR, "product_silhouette_scores.csv"), index=False)
    cluster_summary.to_csv(os.path.join(RESULTS_DIR, "product_cluster_summary.csv"))

    pca_2d.to_csv(os.path.join(DATA_PROCESSED, "product_pca_projection_2d.csv"), index=False)
    pca_3d.to_csv(os.path.join(DATA_PROCESSED, "product_pca_projection_3d.csv"), index=False)


def _save_preference_artifacts(history_enriched, cluster_preferences, category_preferences, customer_summary):
    """Guarda artefactos generados durante el analisis de preferencias."""
    history_enriched.to_csv(
        os.path.join(DATA_PROCESSED, "customer_product_history_clustered.csv"),
        index=False,
    )
    cluster_preferences.to_csv(
        os.path.join(RESULTS_DIR, "customer_cluster_preferences.csv"),
        index=False,
    )
    category_preferences.to_csv(
        os.path.join(RESULTS_DIR, "customer_category_preferences.csv"),
        index=False,
    )
    customer_summary.to_csv(
        os.path.join(RESULTS_DIR, "customer_preference_summary.csv"),
        index=False,
    )


def _generate_visualizations(product_data, silhouette_scores, cluster_summary, pca_2d, pca_model_2d, pca_3d, pca_model_3d):
    """Genera todas las visualizaciones del bloque 2."""
    plot_product_silhouette_scores(silhouette_scores)
    plot_product_cluster_distribution(product_data)
    plot_products_by_cluster(product_data)
    plot_product_pca_clusters_2d(pca_2d, pca_model_2d)
    plot_product_pca_clusters_3d(pca_3d, pca_model_3d)
    plot_product_category_heatmap(product_data)
    plot_product_cluster_profile_heatmap(cluster_summary)


def run_recommendation_pipeline():
    """
    Pipeline completo de recomendacion de productos.
    Construye clusters de productos, analiza preferencias de clientes,
    genera recomendaciones y exporta resultados + visualizaciones.
    """
    print("=" * 60)
    print("PIPELINE DE RECOMENDACION DE PRODUCTOS")
    print("=" * 60)

    _ensure_output_directories()

    # paso 1: construir dataset de productos
    print("\n[1/6] Construyendo dataset de productos...")
    product_data = build_product_dataset()
    print(f"  Productos disponibles: {len(product_data)}")

    # paso 2: crear features para clustering
    print("\n[2/6] Creando features para clustering...")
    clustering_features, feature_names = create_recommendation_features(product_data)
    print(f"  Features de clustering: {len(feature_names)}")

    # paso 3: buscar k optimo y clusterizar
    print("\n[3/6] Buscando k optimo y entrenando K-Means...")
    best_k, silhouette_scores, kmeans_model = find_optimal_k(clustering_features)
    product_data["cluster"] = kmeans_model.labels_

    cluster_summary = build_cluster_summary(product_data)
    print("\n  Resumen de clusters:")
    print(cluster_summary.to_string())

    pca_2d, pca_model_2d = build_pca_projection(clustering_features, kmeans_model.labels_, n_components=2)
    pca_2d["product_id"] = product_data["product_id"].values

    pca_3d, pca_model_3d = build_pca_projection(clustering_features, kmeans_model.labels_, n_components=3)
    pca_3d["product_id"] = product_data["product_id"].values

    _save_clustering_artifacts(product_data, silhouette_scores, cluster_summary, pca_2d, pca_3d)

    print(
        "  Varianza explicada PCA 2D:"
        f" {pca_model_2d.explained_variance_ratio_.sum():.2%}"
    )
    print(
        "  Varianza explicada PCA 3D:"
        f" {pca_model_3d.explained_variance_ratio_.sum():.2%}"
    )

    # paso 4: analizar preferencias de clientes
    print("\n[4/6] Analizando preferencias de clientes...")
    history = build_customer_product_history()
    history_enriched, cluster_preferences, category_preferences, customer_summary = (
        analyze_customer_preferences(history, product_data)
    )
    _save_preference_artifacts(
        history_enriched,
        cluster_preferences,
        category_preferences,
        customer_summary,
    )
    print(f"  Clientes analizados: {customer_summary.shape[0]}")

    # paso 5: generar recomendaciones
    print("\n[5/6] Generando recomendaciones basadas en clusters...")
    sample_recommendations = generate_sample_recommendations(
        history_enriched,
        product_data,
        cluster_preferences,
        category_preferences,
        top_customers=10,
        top_n=5,
    )

    if sample_recommendations.empty:
        print("  No se pudieron generar recomendaciones de ejemplo.")
    else:
        sample_recommendations.to_csv(
            os.path.join(RESULTS_DIR, "sample_recommendations.csv"),
            index=False,
        )
        print(
            "  Recomendaciones generadas:"
            f" {sample_recommendations.shape[0]} filas"
            f" para {sample_recommendations['customer_unique_id'].nunique()} clientes"
        )

    # paso 6: visualizaciones
    print("\n[6/6] Generando visualizaciones...")
    _generate_visualizations(
        product_data,
        silhouette_scores,
        cluster_summary,
        pca_2d,
        pca_model_2d,
        pca_3d,
        pca_model_3d,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print(f"  Mejor k encontrado: {best_k}")
    print(f"  Datos procesados en: {DATA_PROCESSED}")
    print(f"  Resultados en: {RESULTS_DIR}")
    print(f"  Graficas en: {FIGURES_DIR}")
    print("=" * 60)

    return product_data, cluster_preferences, sample_recommendations


if __name__ == "__main__":
    run_recommendation_pipeline()
