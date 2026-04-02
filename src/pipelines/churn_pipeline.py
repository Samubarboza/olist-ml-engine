from src.data.load_data import load_csv
from src.data.preprocess import clean_missing
from src.data.split import split_dataset
from src.features.feature_engineering import create_features
from src.models.train import train_models
from src.evaluation.evaluate import evaluate_all_models
from src.evaluation.visualize import plot_model_metrics
from src.config.settings import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, FIGURES_DIR
import os

def run_churn_pipeline():
    # cargar datos
    raw_data = load_csv("customers.csv")

    # limpiar datos
    clean_data = clean_missing(raw_data)

    # crear features
    features, target = create_features(clean_data, target_column="churn")

    # guardar dataset procesado
    os.makedirs("data/processed", exist_ok=True)
    processed_df = features.copy()
    processed_df["churn"] = target
    processed_df.to_csv("data/processed/churn_data.csv", index=False)

    # dividir dataset
    X_train, X_test, y_train, y_test = split_dataset(features, target)

    # entrenar modelos y calcular métricas
    trained_models, results = train_models(
        X_train, y_train,
        X_test=X_test,
        y_test=y_test,
        save_models=True,
        models_dir=MODELS_DIR
    )

    # guardar métricas
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_csv_path = os.path.join(RESULTS_DIR, "churn_metrics.csv")
    evaluate_all_models(X_test, y_test, models_folder=MODELS_DIR, output_csv=metrics_csv_path)

    # generar gráficas automáticamente
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_model_metrics(metrics_csv=metrics_csv_path)

    print("Churn pipeline finished successfully.")
    return trained_models, results