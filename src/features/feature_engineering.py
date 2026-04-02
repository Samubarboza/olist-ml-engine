import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# crea features para churn predcition 
def create_features(df, target_column):
    # Separar target
    y = df[target_column]
    df = df.drop(columns=[target_column])
    
    # Variables numéricas, normalización
    num_cols = df.select_dtypes(include="number").columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Variables categóricas,  one-hot encoding
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse=False, drop="first")
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)
    
    # Features de comportamiento
    # Tiempo desde última compra en días
    if "last_purchase_days" in df.columns:
        df["recency"] = df["last_purchase_days"]
    
    # Número total de compras
    if "total_purchases" in df.columns:
        df["frequency"] = df["total_purchases"]
    
    # Promedio gasto por compra
    if "avg_purchase_value" in df.columns:
        df["monetary"] = df["avg_purchase_value"]
    
    # Normalizar behavioral features si existen
    for col in ["recency", "frequency", "monetary"]:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Retornar features y target
    X = df.copy()
    return X, y