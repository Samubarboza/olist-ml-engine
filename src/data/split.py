from sklearn.model_selection import train_test_split
from src.config.settings import TEST_SIZE, RANDOM_STATE

def split_dataset(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)