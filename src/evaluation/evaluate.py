import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# función para evaluar un modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# función para evaluar todos los modelos guardados
def evaluate_all_models(X_test, y_test, models_folder="models", output_csv="reports/results/model_metrics.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results = {}
    
    # buscar todos los modelos .pkl en la carpeta
    for file in os.listdir(models_folder):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            model = joblib.load(os.path.join(models_folder, file))
            metrics = evaluate_model(model, X_test, y_test)
            results[model_name] = metrics
    
    # pasar resultados a dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by="f1", ascending=False)
    
    # guardar CSV con resultados
    results_df.to_csv(output_csv)
    
    # mostrar en consola
    print("Resultados de todos los modelos:")
    print(results_df)
    
    return results_df

# ejemplo de uso (se puede comentar en producción)
if __name__ == "__main__":
    df = pd.read_csv("data/processed/churn_data.csv")
    from src.features.feature_engineering import create_features
    X, y = create_features(df, target_column="churn")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    evaluate_all_models(X_test, y_test)