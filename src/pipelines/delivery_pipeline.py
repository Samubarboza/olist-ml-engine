import os

from sklearn.base import clone

from src.config.settings import DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR
from src.data.split import split_dataset
from src.delivery.dataset import build_delivery_dataset
from src.delivery.evaluation import evaluate_delivery_models, get_delivery_feature_importance, identify_orders_with_delay_risk
from src.delivery.features import create_delivery_features
from src.delivery.training import train_and_validate_delivery_models
from src.delivery.visualization import plot_delivery_cv_results, plot_delivery_feature_importance, plot_delivery_metrics_comparison, plot_predicted_vs_actual, plot_residuals


def _ensure_output_directories():
    """Crea los directorios de salida del pipeline si no existen."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def run_delivery_pipeline():
    """
    Pipeline completo de estimacion de plazo de entrega.
    Construye dataset, crea features, entrena modelos de regresion,
    evalua resultados e identifica pedidos con riesgo de retraso.
    """
    print("=" * 60)
    print("PIPELINE DE ESTIMACION DE PLAZO DE ENTREGA")
    print("=" * 60)

    _ensure_output_directories()

    # paso 1: construir dataset de delivery
    print("\n[1/6] Construyendo dataset de delivery...")
    delivery_data = build_delivery_dataset()
    delivery_data.to_csv(os.path.join(DATA_PROCESSED, "delivery_data.csv"), index=False)
    print(f"  Dataset: {delivery_data.shape[0]} pedidos, {delivery_data.shape[1]} columnas")
    print(f"  Tasa historica de retraso: {delivery_data['is_late'].mean():.2%}")

    # paso 2: crear features
    print("\n[2/6] Creando features...")
    features, target, feature_names = create_delivery_features(delivery_data)
    print(f"  Features: {features.shape[1]} variables")

    # paso 3: dividir dataset
    print("\n[3/6] Dividiendo dataset (80/20)...")
    X_train, X_test, y_train, y_test = split_dataset(features, target, use_stratify=False)
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Test: {X_test.shape[0]} muestras")

    # paso 4: entrenar y validar modelos
    print("\n[4/6] Entrenando modelos con cross-validation...")
    trained_models, cv_results = train_and_validate_delivery_models(X_train, y_train)
    cv_results.to_csv(os.path.join(RESULTS_DIR, "delivery_cv_results.csv"))

    # paso 5: evaluar y detectar riesgo de retraso
    print("\n[5/6] Evaluando modelos e identificando riesgo de retraso...")
    metrics_df, predictions = evaluate_delivery_models(
        trained_models,
        X_test,
        y_test,
        output_csv=os.path.join(RESULTS_DIR, "delivery_metrics.csv"),
    )

    best_model_name = metrics_df.index[0]
    best_model = trained_models[best_model_name]
    best_predictions = predictions[best_model_name].copy()
    best_predictions.insert(0, "order_id", delivery_data.loc[X_test.index, "order_id"].values)
    best_predictions.to_csv(os.path.join(RESULTS_DIR, "delivery_test_predictions.csv"), index=False)

    production_model = clone(best_model)
    production_model.fit(features, target)

    risk_orders = identify_orders_with_delay_risk(
        production_model,
        features,
        delivery_data[
            [
                "order_id",
                "delivery_days",
                "promised_delivery_days",
                "delay_days",
                "is_late",
            ]
        ],
    )
    risk_orders.to_csv(os.path.join(RESULTS_DIR, "delivery_risk_orders.csv"), index=False)
    print(f"  Mejor modelo: {best_model_name}")
    print(f"  Pedidos con riesgo estimado de retraso: {risk_orders['delay_risk'].sum()}")

    # paso 6: generar visualizaciones
    print("\n[6/6] Generando visualizaciones...")
    plot_delivery_metrics_comparison(metrics_df)
    plot_delivery_cv_results(cv_results)
    plot_predicted_vs_actual(best_predictions, best_model_name)
    plot_residuals(best_predictions, best_model_name)

    importance_data = get_delivery_feature_importance(trained_models, feature_names)
    if importance_data:
        plot_delivery_feature_importance(importance_data)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print(f"  Mejor modelo: {best_model_name}")
    print(f"  Datos procesados en: {DATA_PROCESSED}")
    print(f"  Resultados en: {RESULTS_DIR}")
    print(f"  Graficas en: {FIGURES_DIR}")
    print("=" * 60)

    return trained_models, metrics_df, risk_orders


if __name__ == "__main__":
    run_delivery_pipeline()
