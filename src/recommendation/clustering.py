from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config.settings import RANDOM_STATE
from src.features.feature_engineering import apply_pca


def find_optimal_k(X, k_range=range(2, 11)):
    """
    Encuentra el numero optimo de clusters usando silhouette score.
    Retorna el mejor k, los scores por k y el modelo ajustado.
    """
    max_k = min(max(k_range), len(X) - 1)
    valid_k_values = [k for k in k_range if 2 <= k <= max_k]

    if not valid_k_values:
        raise ValueError("No hay suficientes productos para evaluar multiples clusters.")

    scores = {}
    sample_size = min(len(X), 10000)

    for k in valid_k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X)
        scores[k] = silhouette_score(X, labels, sample_size=sample_size, random_state=RANDOM_STATE)
        print(f"    k={k}: silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"  Mejor k: {best_k} (silhouette={scores[best_k]:.4f})")

    best_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    best_model.fit(X)

    return best_k, scores, best_model


def build_cluster_summary(product_data):
    """
    Resume cada cluster con volumen, categoria dominante y metricas promedio.
    """
    cluster_summary = product_data.groupby("cluster").agg(
        products=("product_id", "count"),
        distinct_categories=("category", "nunique"),
        top_category=("category", lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"),
        avg_price=("avg_price", "mean"),
        avg_review_score=("avg_review_score", "mean"),
        avg_total_orders=("total_orders", "mean"),
        avg_total_revenue=("total_revenue", "mean"),
    )

    return cluster_summary.round(3).sort_index()


def build_pca_projection(X, labels, n_components):
    """
    Proyecta las features de clustering en un espacio PCA de 2D o 3D.
    """
    projection, pca = apply_pca(X, n_components=n_components)

    if projection.shape[1] < n_components:
        raise ValueError("No hay suficientes dimensiones para construir la proyeccion PCA solicitada.")

    projection["cluster"] = labels

    return projection, pca
