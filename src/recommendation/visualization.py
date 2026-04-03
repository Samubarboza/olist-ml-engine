import os

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.settings import FIGURES_DIR
from src.recommendation.utils import normalize_series


# Grafica los silhouette scores por valor de k
def plot_product_silhouette_scores(scores, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(scores.keys())
    values = list(scores.values())
    best_k = max(scores, key=scores.get)

    ax.plot(ks, values, marker="o", color="steelblue", linewidth=2)
    ax.axvline(best_k, color="firebrick", linestyle="--", alpha=0.7, label=f"Mejor k={best_k}")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Seleccion de k optimo")
    ax.set_xticks(ks)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "product_silhouette_scores.png"), dpi=150)
    plt.close(fig)


    # Grafica la distribucion de productos y reviews por cluster
def plot_product_cluster_distribution(product_data, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cluster_counts = product_data["cluster"].value_counts().sort_index()
    axes[0].bar(cluster_counts.index, cluster_counts.values, color="steelblue")
    axes[0].set_title("Distribucion de clusters")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Cantidad de productos")

    avg_reviews = product_data.groupby("cluster")["avg_review_score"].mean().sort_index()
    axes[1].bar(avg_reviews.index, avg_reviews.values, color="coral")
    axes[1].set_title("Review promedio por cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Review promedio")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "product_cluster_distribution.png"), dpi=150)
    plt.close(fig)


   # Grafica productos por cluster y categoria para las categorias mas frecuentes
def plot_products_by_cluster(product_data, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    top_categories = product_data["category"].value_counts().head(10).index.tolist()
    filtered = product_data[product_data["category"].isin(top_categories)]
    cluster_category_counts = pd.crosstab(filtered["cluster"], filtered["category"])

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_category_counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Productos por cluster y categoria (top 10 categorias)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad de productos")
    ax.legend(title="Categoria", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "products_by_cluster.png"), dpi=150)
    plt.close(fig)


    # Grafica clusters de productos proyectados en 2D con PCA
def plot_product_pca_clusters_2d(pca_projection, pca_model, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        pca_projection["PC1"],
        pca_projection["PC2"],
        c=pca_projection["cluster"],
        cmap="viridis",
        alpha=0.55,
        s=14,
    )
    fig.colorbar(scatter, label="Cluster")
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Clusters de productos en PCA 2D")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "product_pca_clusters_2d.png"), dpi=150)
    plt.close(fig)


    # Grafica clusters de productos proyectados en 3D con PCA
def plot_product_pca_clusters_3d(pca_projection, pca_model, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pca_projection["PC1"],
        pca_projection["PC2"],
        pca_projection["PC3"],
        c=pca_projection["cluster"],
        cmap="viridis",
        alpha=0.45,
        s=12,
    )
    fig.colorbar(scatter, ax=ax, pad=0.12, label="Cluster")
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca_model.explained_variance_ratio_[2]:.1%})")
    ax.set_title("Clusters de productos en PCA 3D")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "product_pca_clusters_3d.png"), dpi=150)
    plt.close(fig)


    # Grafica un heatmap de categorias por cluster
def plot_product_category_heatmap(product_data, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    top_categories = product_data["category"].value_counts().head(15).index
    filtered = product_data[product_data["category"].isin(top_categories)]

    pivot = pd.crosstab(filtered["category"], filtered["cluster"])
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Distribucion de categorias por cluster (top 15)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Categoria")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "product_category_heatmap.png"), dpi=150)
    plt.close(fig)


    # Grafica el perfil relativo de cada cluster usando metricas normalizadas
def plot_product_cluster_profile_heatmap(cluster_summary, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    heatmap_data = cluster_summary[
        ["products", "distinct_categories", "avg_price", "avg_review_score", "avg_total_orders", "avg_total_revenue"]
    ].copy()
    heatmap_data = heatmap_data.apply(normalize_series)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Perfil relativo de clusters")
    ax.set_xlabel("Metricas normalizadas")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "product_cluster_profile_heatmap.png"), dpi=150)
    plt.close(fig)
