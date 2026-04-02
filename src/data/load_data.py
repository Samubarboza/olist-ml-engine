import os
import pandas as pd
from src.config.settings import DATA_RAW


def load_csv(filename):
    """Carga un archivo CSV desde la carpeta de datos raw."""
    path = os.path.join(DATA_RAW, filename)
    return pd.read_csv(path)
