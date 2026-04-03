import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_delivery_features(df, target_column="delivery_days"):
    """
    Transforma el dataset de delivery en features listas para regresion.
    Escala numericas y codifica categoricas con one-hot encoding.
    """
    target = df[target_column].copy()

    columns_to_drop = [
        "order_id",
        "delivery_days",
        "delay_days",
        "is_late",
        "customer_zip_code_prefix",
        "seller_zip_code_prefix",
        "customer_city",
        "seller_city",
    ]
    feature_columns = [column for column in df.columns if column not in columns_to_drop]
    features = df[feature_columns].copy()

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(include="object").columns.tolist()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features[numeric_columns])
    scaled_df = pd.DataFrame(scaled_values, columns=numeric_columns, index=features.index)

    encoded_df = pd.DataFrame(index=features.index)
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoded_values = encoder.fit_transform(features[categorical_columns])
        encoded_names = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_values, columns=encoded_names, index=features.index)

    final_features = pd.concat([scaled_df, encoded_df], axis=1)
    feature_names = final_features.columns.tolist()

    return final_features, target, feature_names
