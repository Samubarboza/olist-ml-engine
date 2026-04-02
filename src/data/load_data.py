import pandas as pd
from src.config.settings import DATA_RAW

def load_csv(filename):
    path = f"{DATA_RAW}{filename}"
    return pd.read_csv(path)