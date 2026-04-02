from src.data.load_data import load_csv
from src.data.preprocess import clean_missing
from src.data.split import split_dataset
from src.features.feature_engineering import create_features
from src.models.train import train_models
from src.evaluation.evaluate import evaluate_models
from src.config.settings import DATA_PROCESSED, MODELS_DIR

def run_churn_pipeline():
    # Cargar datos
    raw_data = load_csv("customers.csv")

    # Limpiar datos
    clean_data = clean_missing(raw_data)

    # Crear features
    features, target = create_features(clean_data, target_column="churn")

    # Dividir dataset
    X_train, X_test, y_train, y_test = split_dataset(features, target)

    # Entrenar modelos
    trained_models = train_models(X_train, y_train)

    # Evaluar modelos
    metrics = evaluate_models(trained_models, X_test, y_test)

    # Guardar métricas y modelos
    print("Evaluation Metrics:", metrics)
    for name, model in trained_models.items():
        model_path = f"{MODELS_DIR}{name}_churn.pkl"
        model.save(model_path)

    print("Churn pipeline finished successfully.")