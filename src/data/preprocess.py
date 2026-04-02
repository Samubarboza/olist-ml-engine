import pandas as pd

def clean_missing(df):
    # elimina columnas con más del 50% de valores nulos
    df = df.loc[:, df.isnull().mean() < 0.5]
    # reemplaza nulos por la media para numéricas
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df