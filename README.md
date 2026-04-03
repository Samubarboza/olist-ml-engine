# Olist ML Engine

Proyecto de machine learning sobre datos del marketplace Olist. El objetivo es analizar el comportamiento de los clientes y predecir quienes tienen probabilidad de abandonar la plataforma (churn).

## Que hace este proyecto

- Construye un dataset de churn a nivel de cliente a partir de los datos raw de Olist (ordenes, pagos, reviews, productos)
- Define churn como clientes sin compras en los ultimos 180 dias
- Genera features basadas en RFM (recency, frequency, monetary), reviews, entregas y pagos
- Entrena y evalua 7 modelos de clasificacion con cross-validation (StratifiedKFold, 5 folds)
- Aplica PCA para reduccion de dimensionalidad
- Segmenta clientes con K-Means
- Genera visualizaciones automaticas (metricas, matrices de confusion, curvas ROC, feature importance)
- Incluye un dashboard interactivo con Streamlit

## Estructura del proyecto

```
olist-ml-engine/
├── app.py                          # dashboard Streamlit
├── Dockerfile
├── requirements.txt
├── data/raw/                       # datasets originales de Olist
├── notebooks/
│   └── 01_churn_prediction.ipynb   # notebook con analisis completo
└── src/
    ├── config/settings.py          # configuracion y constantes
    ├── data/
    │   ├── load_data.py            # carga de CSVs
    │   ├── preprocess.py           # merge de tablas y etiqueta de churn
    │   └── split.py                # division train/test
    ├── features/
    │   └── feature_engineering.py  # escalado, encoding y PCA
    ├── models/
    │   └── train.py                # entrenamiento con cross-validation
    ├── evaluation/
    │   ├── evaluate.py             # metricas y feature importance
    │   └── visualize.py            # graficas
    └── pipelines/
        └── churn_pipeline.py       # orquesta todo el flujo
```

## Requisitos

- Docker

## Como correr

**1. Construir la imagen:**

```bash
docker build -t olist-ml-engine .
```

**2. Ejecutar el pipeline de churn:**

```bash
docker run --rm -v $(pwd):/app olist-ml-engine python -m src.pipelines.churn_pipeline
```

Esto genera los modelos en `models/`, las metricas en `reports/results/` y las graficas en `reports/figures/`.

**3. Levantar el dashboard:**

```bash
docker run --rm -p 8501:8501 -v $(pwd):/app olist-ml-engine
```

Abrir `http://localhost:8501` en el navegador. Si no se ejecuto el pipeline antes, el dashboard tiene un boton para ejecutarlo.

**4. Abrir el notebook (opcional):**

```bash
docker run --rm -p 8888:8888 -v $(pwd):/app olist-ml-engine jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

## Modelos utilizados

| Modelo | Tipo |
|---|---|
| Logistic Regression | Clasificacion |
| Decision Tree | Clasificacion |
| Random Forest | Clasificacion |
| Gradient Boosting | Clasificacion |
| SVM | Clasificacion |
| KNN | Clasificacion |
| Naive Bayes | Clasificacion |
| PCA | Reduccion de dimensionalidad |
| K-Means | Segmentacion / Clustering |

## Estado del proyecto

Este proyecto se encuentra en desarrollo activo. Por ahora se implemento el pipeline de prediccion de churn como primer caso de uso. Se planea agregar mas pipelines y funcionalidades sobre los datos de Olist en las proximas iteraciones.
