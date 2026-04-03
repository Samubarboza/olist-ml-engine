# Olist ML Engine

Proyecto de machine learning sobre datos del marketplace Olist. El objetivo es analizar el comportamiento de los clientes, predecir quienes tienen probabilidad de abandonar la plataforma (churn), recomendar productos en base a clusters y estimar plazos de entrega.

## Que hace este proyecto

El proyecto tiene tres casos de uso principales, cada uno con su propio pipeline, notebook y seccion en el dashboard:

### Caso 1 - Prediccion de Churn

- Construye un dataset de churn a nivel de cliente a partir de los datos raw de Olist (ordenes, pagos, reviews, productos)
- Define churn como clientes sin compras en los ultimos 180 dias
- Genera features basadas en RFM (recency, frequency, monetary), reviews, entregas y pagos
- Entrena y evalua 7 modelos de clasificacion con cross-validation (StratifiedKFold, 5 folds)
- Aplica PCA para reduccion de dimensionalidad
- Segmenta clientes con K-Means
- Genera visualizaciones automaticas (metricas, matrices de confusion, curvas ROC, feature importance)

### Caso 2 - Recomendacion de Productos

- Construye un dataset a nivel de producto con metricas de venta, reviews y atributos fisicos
- Agrupa productos con K-Means usando features de precio, reviews, volumen de ventas y categorias
- Busca el k optimo con silhouette score
- Analiza preferencias de clientes por historial de compras
- Genera recomendaciones basadas en clusters
- Aplica PCA para visualizar clusters en 2D y 3D
- Genera heatmaps de categorias por cluster y perfiles de clusters

### Caso 3 - Estimacion de Plazo de Entrega

- Construye un dataset a nivel de orden con features de ubicacion, producto, volumen y distancia geografica
- Entrena 5 modelos de regresion: Regresion Lineal, Random Forest, Gradient Boosting, SVR y KNN
- Evalua con MAE, RMSE y R2 usando cross-validation (KFold, 5 folds)
- Identifica pedidos con riesgo de retraso comparando la prediccion contra el plazo prometido
- Genera visualizaciones de predicho vs actual, residuales y feature importance

### Dashboard

- Dashboard interactivo con Streamlit que conecta los 3 casos de uso
- Navegacion por sidebar para cambiar entre Churn, Recomendacion y Delivery
- Cada seccion permite ejecutar su pipeline desde el navegador si no hay resultados previos

## Estructura del proyecto

```
olist-ml-engine/
├── app.py                              # dashboard Streamlit (3 casos de uso)
├── Dockerfile
├── requirements.txt
├── data/raw/                           # datasets originales de Olist
├── notebooks/
│   ├── 01_churn_prediction.ipynb       # notebook de churn
│   ├── 02_product_recommendations.ipynb # notebook de recomendacion
│   └── 03_delivery_estimation.ipynb    # notebook de delivery
└── src/
    ├── config/
    │   └── settings.py                 # configuracion y constantes
    ├── data/
    │   ├── load_data.py                # carga de CSVs
    │   ├── preprocess.py               # construccion de datasets (churn, productos, historial)
    │   └── split.py                    # division train/test
    ├── features/
    │   └── feature_engineering.py      # escalado, encoding y PCA
    ├── models/
    │   └── train.py                    # entrenamiento de clasificacion con cross-validation
    ├── evaluation/
    │   ├── evaluate.py                 # metricas de clasificacion y feature importance
    │   └── visualize.py                # graficas de churn
    ├── recommendation/
    │   ├── clustering.py               # K-Means, silhouette, PCA de productos
    │   ├── engine.py                   # preferencias de clientes y recomendaciones
    │   ├── utils.py                    # utilidades de normalizacion
    │   └── visualization.py            # graficas de recomendacion
    ├── delivery/
    │   ├── dataset.py                  # construccion del dataset de delivery
    │   ├── features.py                 # features de regresion (escalado, one-hot)
    │   ├── training.py                 # entrenamiento de regresion con cross-validation
    │   ├── evaluation.py               # metricas de regresion y deteccion de riesgo
    │   ├── utils.py                    # calculo de distancia haversine
    │   └── visualization.py            # graficas de delivery
    ├── utils/
    │   └── helpers.py                  # utilidades generales
    └── pipelines/
        ├── churn_pipeline.py           # pipeline completo de churn
        ├── recommendation_pipeline.py  # pipeline completo de recomendacion
        └── delivery_pipeline.py        # pipeline completo de delivery
```

## Requisitos

- Docker

## Como correr

**1. Construir la imagen:**

```bash
docker build -t olist-ml-engine .
```

**2. Levantar el dashboard:**

```bash
docker run --rm -p 8501:8501 -v $(pwd):/app olist-ml-engine
```

Abrir `http://localhost:8501` en el navegador. El dashboard tiene un boton para ejecutar cada pipeline si todavia no se corrieron.

**3. Ejecutar un pipeline por separado (opcional):**

```bash
docker run --rm -v $(pwd):/app olist-ml-engine python -m src.pipelines.churn_pipeline
docker run --rm -v $(pwd):/app olist-ml-engine python -m src.pipelines.recommendation_pipeline
docker run --rm -v $(pwd):/app olist-ml-engine python -m src.pipelines.delivery_pipeline
```

Esto genera los modelos en `models/`, las metricas en `reports/results/` y las graficas en `reports/figures/`.

**4. Abrir los notebooks (opcional):**

```bash
docker run --rm -p 8888:8888 -v $(pwd):/app olist-ml-engine jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

## Modelos utilizados

| Modelo | Tipo | Caso de uso |
|---|---|---|
| Logistic Regression | Clasificacion | Churn |
| Decision Tree | Clasificacion | Churn |
| Random Forest | Clasificacion | Churn |
| Gradient Boosting | Clasificacion | Churn |
| SVM | Clasificacion | Churn |
| KNN | Clasificacion | Churn |
| Naive Bayes | Clasificacion | Churn |
| K-Means | Clustering | Churn, Recomendacion |
| PCA | Reduccion de dimensionalidad | Churn, Recomendacion |
| Linear Regression | Regresion | Delivery |
| Random Forest Regressor | Regresion | Delivery |
| Gradient Boosting Regressor | Regresion | Delivery |
| SVR | Regresion | Delivery |
| KNN Regressor | Regresion | Delivery |

## Datasets de Olist

El proyecto usa los siguientes datasets del marketplace Olist:

- `olist_customers_dataset.csv` - datos de clientes
- `olist_orders_dataset.csv` - ordenes
- `olist_order_items_dataset.csv` - items por orden
- `olist_order_payments_dataset.csv` - pagos
- `olist_order_reviews_dataset.csv` - reviews
- `olist_products_dataset.csv` - productos
- `olist_sellers_dataset.csv` - vendedores
- `olist_geolocation_dataset.csv` - geolocalizacion
- `product_category_name_translation.csv` - traduccion de categorias
