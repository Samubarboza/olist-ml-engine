import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.settings import RESULTS_DIR, FIGURES_DIR

def plot_model_metrics(metrics_csv=os.path.join(RESULTS_DIR, "churn_metrics.csv")):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # cargar métricas
    df = pd.read_csv(metrics_csv, index_col=0)

    # convertir índices a columna para gráficas
    df = df.reset_index().rename(columns={"index": "Model"})

    # graficar todas las métricas en un gráfico de barras
    metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="Model", y=metric)
        plt.title(f"Comparación de {metric} por modelo")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        figure_path = os.path.join(FIGURES_DIR, f"{metric}_comparison.png")
        plt.savefig(figure_path)
        plt.close()
        print(f"{metric} guardado en {figure_path}")

    print("Todas las gráficas generadas correctamente.")
    return df

# ejemplo de uso
if __name__ == "__main__":
    plot_model_metrics()