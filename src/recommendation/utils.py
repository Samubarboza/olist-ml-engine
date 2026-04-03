import pandas as pd



# Normaliza una serie numerica al rango [0, 1].
# Si la serie es constante, retorna 1 para valores positivos o 0 en caso contrario
def normalize_series(series):
    series = series.fillna(0).astype(float)
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        fill_value = 1.0 if max_value > 0 else 0.0
        return pd.Series(fill_value, index=series.index, dtype=float)

    return (series - min_value) / (max_value - min_value)
