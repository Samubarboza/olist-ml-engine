import os
from src.data.preprocess import build_churn_dataset
from src.features.feature_engineering import create_churn_features
from src.data.split import split_dataset
from src.models.train import train_and_validate
from src.evaluation.evaluate import evaluate_all_models, get_feature_importance
from src.evaluation.visualize import (
    plot_metrics_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    plot_cv_results
)
from src.config.settings import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, FIGURES_DIR


def run_churn_pipeline():
    """
    Pipeline completo de prediccion de churn.
    Construye dataset, crea features, entrena modelos con cross-validation,
    evalua en test set y genera visualizaciones.
    """
    print("=" * 60)
    print("PIPELINE DE PREDICCION DE CHURN")
    print("=" * 60)

    # paso 1: construir dataset de churn desde datos raw
    print("\n[1/6] Construyendo dataset de churn...")
    churn_data = build_churn_dataset()

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    churn_data.to_csv(os.path.join(DATA_PROCESSED, "churn_data.csv"), index=False)
    print(f"  Dataset: {churn_data.shape[0]} clientes, {churn_data.shape[1]} columnas")
    print(f"  Tasa de churn: {churn_data['churn'].mean():.2%}")

    # paso 2: crear features
    print("\n[2/6] Creando features...")
    features, target, feature_names = create_churn_features(churn_data)
    print(f"  Features: {features.shape[1]} variables")

    # paso 3: dividir dataset
    print("\n[3/6] Dividiendo dataset (80/20, stratified)...")
    X_train, X_test, y_train, y_test = split_dataset(features, target)
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")

    # paso 4: entrenar y validar con cross-validation
    print("\n[4/6] Entrenando modelos con cross-validation...")
    trained_models, cv_results = train_and_validate(X_train, y_train)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cv_results.to_csv(os.path.join(RESULTS_DIR, "churn_cv_results.csv"))

    # paso 5: evaluar en test set
    print("\n[5/6] Evaluando modelos en test set...")
    metrics_df, confusion_matrices, reports, predictions, probabilities = evaluate_all_models(
        trained_models, X_test, y_test,
        output_csv=os.path.join(RESULTS_DIR, "churn_metrics.csv")
    )

    # paso 6: generar visualizaciones
    print("\n[6/6] Generando visualizaciones...")
    plot_metrics_comparison(metrics_df)
    plot_confusion_matrices(confusion_matrices)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_cv_results(cv_results)

    importance_data = get_feature_importance(trained_models, feature_names)
    if importance_data:
        plot_feature_importance(importance_data)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print(f"  Modelos guardados en: {MODELS_DIR}")
    print(f"  Resultados en: {RESULTS_DIR}")
    print(f"  Graficas en: {FIGURES_DIR}")
    print("=" * 60)

    return trained_models, metrics_df, cv_results


if __name__ == "__main__":
    run_churn_pipeline()
