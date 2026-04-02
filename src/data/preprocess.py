import pandas as pd
from src.data.load_data import load_csv
from src.config.settings import CHURN_THRESHOLD_DAYS


def build_churn_dataset():
    """
    Construye el dataset de churn a nivel de cliente unico.
    Carga las tablas raw de Olist, las mergea, agrega metricas por cliente
    y crea la etiqueta binaria de churn.
    """
    # cargar tablas raw
    customers = load_csv("olist_customers_dataset.csv")
    orders = load_csv("olist_orders_dataset.csv")
    order_items = load_csv("olist_order_items_dataset.csv")
    payments = load_csv("olist_order_payments_dataset.csv")
    reviews = load_csv("olist_order_reviews_dataset.csv")
    products = load_csv("olist_products_dataset.csv")

    # filtrar solo ordenes entregadas
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])
    delivered["order_delivered_customer_date"] = pd.to_datetime(delivered["order_delivered_customer_date"])
    delivered["order_estimated_delivery_date"] = pd.to_datetime(delivered["order_estimated_delivery_date"])

    # vincular ordenes con clientes (customer_unique_id es el identificador real)
    order_customer = delivered.merge(
        customers[["customer_id", "customer_unique_id", "customer_state"]],
        on="customer_id"
    )

    # agregar items a nivel de orden (evitar duplicacion al mergear)
    items_per_order = order_items.groupby("order_id").agg(
        order_total_price=("price", "sum"),
        order_total_freight=("freight_value", "sum"),
        order_item_count=("order_item_id", "count")
    ).reset_index()

    # agregar pagos a nivel de orden
    payments_per_order = payments.groupby("order_id").agg(
        order_payment=("payment_value", "sum"),
        order_installments=("payment_installments", "mean")
    ).reset_index()

    # tipo de pago principal por orden (el de mayor monto)
    main_payment = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id", keep="first")[["order_id", "payment_type"]]
    )

    # review promedio por orden
    review_per_order = reviews.groupby("order_id").agg(
        review_score=("review_score", "mean")
    ).reset_index()

    # construir tabla a nivel de orden con todas las metricas
    order_data = order_customer[
        ["order_id", "customer_unique_id", "customer_state",
         "order_purchase_timestamp", "order_delivered_customer_date",
         "order_estimated_delivery_date"]
    ]
    order_data = order_data.merge(items_per_order, on="order_id", how="left")
    order_data = order_data.merge(payments_per_order, on="order_id", how="left")
    order_data = order_data.merge(main_payment, on="order_id", how="left")
    order_data = order_data.merge(review_per_order, on="order_id", how="left")

    # calcular metricas de entrega
    order_data["delivery_days"] = (
        order_data["order_delivered_customer_date"] - order_data["order_purchase_timestamp"]
    ).dt.days
    order_data["is_late"] = (
        order_data["order_delivered_customer_date"] > order_data["order_estimated_delivery_date"]
    ).astype(int)

    # rellenar payment_type nulo antes de agregar
    order_data["payment_type"] = order_data["payment_type"].fillna("unknown")

    # fecha de referencia (ultima compra en el dataset)
    reference_date = order_data["order_purchase_timestamp"].max()

    # agregar a nivel de cliente unico
    customer_data = order_data.groupby("customer_unique_id").agg(
        recency_days=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary_total=("order_total_price", "sum"),
        monetary_avg=("order_total_price", "mean"),
        avg_freight=("order_total_freight", "mean"),
        avg_review_score=("review_score", "mean"),
        review_count=("review_score", "count"),
        avg_installments=("order_installments", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        late_deliveries=("is_late", "sum"),
        total_items=("order_item_count", "sum"),
        customer_state=("customer_state", "first")
    ).reset_index()

    # diversidad de productos por cliente (desde items raw, para conteo real de unicos)
    items_with_customer = order_items.merge(
        order_customer[["order_id", "customer_unique_id"]].drop_duplicates(),
        on="order_id"
    ).merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left"
    )
    product_diversity = items_with_customer.groupby("customer_unique_id").agg(
        unique_products=("product_id", "nunique"),
        unique_categories=("product_category_name", "nunique")
    ).reset_index()
    customer_data = customer_data.merge(product_diversity, on="customer_unique_id", how="left")

    # tipo de pago mas comun por cliente
    payment_mode = (
        order_data.groupby("customer_unique_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .reset_index()
        .rename(columns={"payment_type": "most_common_payment"})
    )
    customer_data = customer_data.merge(payment_mode, on="customer_unique_id", how="left")

    # etiqueta de churn (sin compra en los ultimos N dias)
    customer_data["churn"] = (
        customer_data["recency_days"] > CHURN_THRESHOLD_DAYS
    ).astype(int)

    # ratio de entregas tardias
    customer_data["late_delivery_ratio"] = (
        customer_data["late_deliveries"] / customer_data["frequency"]
    )

    # limpiar valores faltantes
    customer_data = clean_missing(customer_data)

    return customer_data


def clean_missing(df):
    """
    Limpia valores faltantes del dataset.
    Elimina columnas con mas del 50% de nulos.
    Rellena numericas con la mediana y categoricas con la moda.
    """
    df = df.loc[:, df.isnull().mean() < 0.5]

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            df[col] = df[col].fillna("unknown")

    return df
