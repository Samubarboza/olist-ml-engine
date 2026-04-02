import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "reports", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CHURN_THRESHOLD_DAYS = 180
