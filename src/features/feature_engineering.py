import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from src.config.settings import RANDOM_STATE


def create_churn_features(df, target_column="churn"):
    """
    Transforma el dataset de churn en features listas para el modelo.
    Escala variables numericas con StandardScaler.
    Codifica variables categoricas con OneHotEncoder.
    Retorna features, target y nombres de features originales.
    """
    # separar target e identificador
    target = df[target_column].copy()
    features = df.drop(columns=[target_column, "customer_unique_id"])

    # identificar columnas numericas y categoricas
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(include="object").columns.tolist()

    # escalar variables numericas
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features[numeric_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_columns, index=features.index)

    # codificar variables categoricas con one-hot encoding
    encoded_df = pd.DataFrame(index=features.index)
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoded_values = encoder.fit_transform(features[categorical_columns])
        encoded_names = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_values, columns=encoded_names, index=features.index)

    # combinar numericas escaladas y categoricas codificadas
    final_features = pd.concat([scaled_df, encoded_df], axis=1)
    feature_names = final_features.columns.tolist()

    return final_features, target, feature_names


def apply_pca(X, n_components=10):
    """
    Aplica PCA para reduccion de dimensionalidad.
    Retorna el dataframe transformado y el modelo PCA ajustado.
    """
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(X)
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_values, columns=pca_columns, index=X.index)
    return pca_df, pca


def create_product_features(df):
    """
    Escala las features numericas del dataset de productos para clustering.
    Excluye product_id y category (son identificadores, no features).
    """
    exclude = ["product_id", "category", "cluster"]
    columns_to_drop = [c for c in exclude if c in df.columns]
    features = df.drop(columns=columns_to_drop)

    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features[numeric_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_columns, index=features.index)

    return scaled_df, numeric_columns


def create_recommendation_features(product_data):
    """
    Crea features de clustering para productos usando:
    - categoria del producto (one-hot encoding)
    - precio
    - reviews
    - popularidad / demanda
    """
    numeric_columns = [
        "avg_price",
        "avg_review_score",
        "review_count",
        "total_orders",
        "total_revenue",
        "total_items_sold",
        "avg_freight",
    ]
    numeric_columns = [column for column in numeric_columns if column in product_data.columns]

    numeric_input = product_data[["product_id", "category"] + numeric_columns].copy()
    scaled_numeric, _ = create_product_features(numeric_input)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_values = encoder.fit_transform(product_data[["category"]].fillna("unknown"))
    encoded_names = encoder.get_feature_names_out(["category"])
    encoded_df = pd.DataFrame(encoded_values, columns=encoded_names, index=product_data.index)

    clustering_features = pd.concat([scaled_numeric, encoded_df], axis=1)
    feature_names = clustering_features.columns.tolist()

    return clustering_features, feature_names
