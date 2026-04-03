import pandas as pd

from src.recommendation.utils import normalize_series



# Analiza las preferencias de los clientes a partir del historial de compras.
# Calcula afinidad por cluster y por categoria
def analyze_customer_preferences(history, product_data):
    product_reference = product_data[
        ["product_id", "cluster", "category", "avg_price", "avg_review_score", "total_orders"]
    ].rename(
        columns={
            "category": "product_category",
            "avg_price": "product_avg_price",
            "avg_review_score": "product_review_score",
            "total_orders": "product_total_orders",
        }
    )

    history_enriched = history.merge(product_reference, on="product_id", how="left")
    history_enriched["category"] = history_enriched["category"].fillna(history_enriched["product_category"])
    history_enriched["price"] = history_enriched["price"].fillna(history_enriched["product_avg_price"])
    history_enriched = history_enriched.drop(columns=["product_category"])

    cluster_preferences = (
        history_enriched.dropna(subset=["cluster"])
        .groupby(["customer_unique_id", "cluster"])
        .agg(
            purchase_count=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            total_spent=("price", "sum"),
            avg_paid_price=("price", "mean"),
            avg_review_score=("product_review_score", "mean"),
            avg_product_orders=("product_total_orders", "mean"),
        )
        .reset_index()
    )

    cluster_preferences["cluster"] = cluster_preferences["cluster"].astype(int)
    cluster_preferences["purchase_score"] = (
        cluster_preferences.groupby("customer_unique_id")["purchase_count"].transform(normalize_series)
    )
    cluster_preferences["spend_score"] = (
        cluster_preferences.groupby("customer_unique_id")["total_spent"].transform(normalize_series)
    )
    cluster_preferences["review_score"] = (
        cluster_preferences.groupby("customer_unique_id")["avg_review_score"].transform(normalize_series)
    )
    cluster_preferences["preference_score"] = (
        0.50 * cluster_preferences["purchase_score"]
        + 0.30 * cluster_preferences["spend_score"]
        + 0.20 * cluster_preferences["review_score"]
    )
    cluster_preferences = cluster_preferences.sort_values(
        ["customer_unique_id", "preference_score", "purchase_count"],
        ascending=[True, False, False],
    )

    category_preferences = (
        history_enriched.groupby(["customer_unique_id", "category"])
        .agg(
            purchase_count=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            total_spent=("price", "sum"),
            avg_paid_price=("price", "mean"),
        )
        .reset_index()
    )
    category_preferences["purchase_score"] = (
        category_preferences.groupby("customer_unique_id")["purchase_count"].transform(normalize_series)
    )
    category_preferences["spend_score"] = (
        category_preferences.groupby("customer_unique_id")["total_spent"].transform(normalize_series)
    )
    category_preferences["preference_score"] = (
        0.60 * category_preferences["purchase_score"]
        + 0.40 * category_preferences["spend_score"]
    )
    category_preferences = category_preferences.sort_values(
        ["customer_unique_id", "preference_score", "purchase_count"],
        ascending=[True, False, False],
    )

    preferred_cluster = (
        cluster_preferences.sort_values(["customer_unique_id", "preference_score"], ascending=[True, False])
        .drop_duplicates("customer_unique_id")
        [["customer_unique_id", "cluster", "preference_score"]]
        .rename(columns={"cluster": "preferred_cluster", "preference_score": "preferred_cluster_score"})
    )

    preferred_category = (
        category_preferences.sort_values(["customer_unique_id", "preference_score"], ascending=[True, False])
        .drop_duplicates("customer_unique_id")
        [["customer_unique_id", "category", "preference_score"]]
        .rename(columns={"category": "preferred_category", "preference_score": "preferred_category_score"})
    )

    customer_summary = (
        history_enriched.groupby("customer_unique_id")
        .agg(
            total_purchases=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            unique_categories=("category", "nunique"),
            avg_paid_price=("price", "mean"),
        )
        .reset_index()
    )
    customer_summary = customer_summary.merge(preferred_cluster, on="customer_unique_id", how="left")
    customer_summary = customer_summary.merge(preferred_category, on="customer_unique_id", how="left")

    return history_enriched, cluster_preferences, category_preferences, customer_summary


def recommend_for_customer(
    customer_id,
    history_enriched,
    product_data,
    cluster_preferences,
    category_preferences,
    top_n=5,
):
    """
    Genera recomendaciones de productos para un cliente usando clusters preferidos,
    categorias frecuentes y ranking por calidad / popularidad.
    """
    customer_history = history_enriched[history_enriched["customer_unique_id"] == customer_id]

    if customer_history.empty:
        return pd.DataFrame()

    bought_products = set(customer_history["product_id"].unique())

    preferred_clusters = (
        cluster_preferences[cluster_preferences["customer_unique_id"] == customer_id]["cluster"]
        .head(3)
        .tolist()
    )
    preferred_categories = (
        category_preferences[category_preferences["customer_unique_id"] == customer_id]["category"]
        .head(3)
        .tolist()
    )

    candidates = product_data[~product_data["product_id"].isin(bought_products)].copy()
    if candidates.empty:
        return pd.DataFrame()

    priority_mask = (
        candidates["cluster"].isin(preferred_clusters)
        | candidates["category"].isin(preferred_categories)
    )
    if priority_mask.any():
        candidates = candidates[priority_mask].copy()

    cluster_weights = {
        cluster: len(preferred_clusters) - index
        for index, cluster in enumerate(preferred_clusters)
    }
    category_weights = {
        category: len(preferred_categories) - index
        for index, category in enumerate(preferred_categories)
    }

    candidates["cluster_match"] = candidates["cluster"].map(cluster_weights).fillna(0).astype(float)
    candidates["category_match"] = candidates["category"].map(category_weights).fillna(0).astype(float)

    if candidates["cluster_match"].max() > 0:
        candidates["cluster_match"] = candidates["cluster_match"] / candidates["cluster_match"].max()
    if candidates["category_match"].max() > 0:
        candidates["category_match"] = candidates["category_match"] / candidates["category_match"].max()

    customer_avg_price = customer_history["price"].mean()
    if pd.isna(customer_avg_price) or customer_avg_price <= 0:
        customer_avg_price = product_data["avg_price"].median()

    candidates["review_score_norm"] = normalize_series(candidates["avg_review_score"])
    candidates["popularity_score_norm"] = normalize_series(candidates["total_orders"])
    candidates["price_match"] = 1 - (
        (candidates["avg_price"] - customer_avg_price).abs() / max(customer_avg_price, 1)
    )
    candidates["price_match"] = candidates["price_match"].clip(lower=0, upper=1)

    candidates["recommendation_score"] = (
        0.40 * candidates["cluster_match"]
        + 0.25 * candidates["category_match"]
        + 0.20 * candidates["review_score_norm"]
        + 0.10 * candidates["popularity_score_norm"]
        + 0.05 * candidates["price_match"]
    )

    recommendations = candidates.sort_values(
        ["recommendation_score", "avg_review_score", "total_orders"],
        ascending=[False, False, False],
    ).head(top_n)

    return recommendations[
        [
            "product_id",
            "category",
            "cluster",
            "avg_price",
            "avg_review_score",
            "total_orders",
            "recommendation_score",
        ]
    ]


def generate_sample_recommendations(
    history_enriched,
    product_data,
    cluster_preferences,
    category_preferences,
    top_customers=10,
    top_n=5,
):
    """
    Genera recomendaciones de ejemplo para los clientes con mayor historial de compras.
    """
    customer_ids = (
        history_enriched.groupby("customer_unique_id")["product_id"]
        .count()
        .nlargest(top_customers)
        .index.tolist()
    )

    all_recommendations = []
    for customer_id in customer_ids:
        recommendations = recommend_for_customer(
            customer_id,
            history_enriched,
            product_data,
            cluster_preferences,
            category_preferences,
            top_n=top_n,
        )

        if not recommendations.empty:
            recommendations.insert(0, "customer_unique_id", customer_id)
            all_recommendations.append(recommendations)

    if not all_recommendations:
        return pd.DataFrame()

    return pd.concat(all_recommendations, ignore_index=True)
