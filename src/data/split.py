from sklearn.model_selection import train_test_split
from src.config.settings import TEST_SIZE, RANDOM_STATE


def split_dataset(X, y, use_stratify=True):
    """
    Divide el dataset en conjuntos de entrenamiento y test.
    Usa stratify por defecto para mantener la proporcion de clases.
    """
    stratify_value = y if use_stratify else None
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_value
    )
