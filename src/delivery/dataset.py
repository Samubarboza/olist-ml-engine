import numpy as np
import pandas as pd

from src.data.load_data import load_csv
from src.data.preprocess import clean_missing
from src.delivery.utils import haversine_distance_km


def build_delivery_dataset():
    """
    Construye un dataset a nivel de orden para estimar el plazo de entrega.

    Usa:
    - ubicacion del cliente y del vendedor principal
    - categoria principal del pedido
    - volumen y peso total del pedido
    - proxy operativa de carrier via shipping_limit_date
    - distancia geografica cliente-vendedor
    """
    customers = load_csv("olist_customers_dataset.csv")
    sellers = load_csv("olist_sellers_dataset.csv")
    orders = load_csv("olist_orders_dataset.csv")
    order_items = load_csv("olist_order_items_dataset.csv")
    products = load_csv("olist_products_dataset.csv")
    geolocation = load_csv("olist_geolocation_dataset.csv")
    translations = load_csv("product_category_name_translation.csv")
    translations.columns = translations.columns.str.replace("\ufeff", "")

    delivered_orders = _prepare_delivered_orders(orders)
    geolocation_reference = _build_geolocation_reference(geolocation)
    order_features = _build_order_level_features(order_items, products, sellers, translations)

    delivery_data = delivered_orders.merge(customers, on="customer_id", how="inner")
    delivery_data = delivery_data.merge(order_features, on="order_id", how="inner")
    delivery_data = _attach_geolocation_features(delivery_data, geolocation_reference)

    delivery_data["distance_km"] = haversine_distance_km(
        delivery_data["seller_lat"],
        delivery_data["seller_lng"],
        delivery_data["customer_lat"],
        delivery_data["customer_lng"],
    )
    delivery_data["seller_customer_same_state"] = (
        delivery_data["seller_state"] == delivery_data["customer_state"]
    ).astype(int)

    delivery_data = _finalize_delivery_dataset(delivery_data)
    delivery_data = clean_missing(delivery_data)

    return delivery_data


def _prepare_delivered_orders(orders):
    """Filtra ordenes entregadas y calcula variables temporales base."""
    delivered_orders = orders[orders["order_status"] == "delivered"].copy()

    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in datetime_columns:
        delivered_orders[column] = pd.to_datetime(delivered_orders[column])

    delivered_orders = delivered_orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )

    delivered_orders["delivery_days"] = (
        delivered_orders["order_delivered_customer_date"] - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivered_orders["promised_delivery_days"] = (
        delivered_orders["order_estimated_delivery_date"] - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivered_orders["approval_days"] = (
        delivered_orders["order_approved_at"] - delivered_orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"] > delivered_orders["order_estimated_delivery_date"]
    ).astype(int)

    delivered_orders = delivered_orders[delivered_orders["delivery_days"] >= 0].copy()

    return delivered_orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "delivery_days",
            "promised_delivery_days",
            "approval_days",
            "is_late",
        ]
    ]


def _build_geolocation_reference(geolocation):
    """Agrega geolocalizacion promedio por zip code prefix."""
    geolocation["geolocation_zip_code_prefix"] = (
        geolocation["geolocation_zip_code_prefix"].astype(str).str.zfill(5)
    )

    return geolocation.groupby("geolocation_zip_code_prefix").agg(
        latitude=("geolocation_lat", "mean"),
        longitude=("geolocation_lng", "mean"),
    ).reset_index()


def _build_order_level_features(order_items, products, sellers, translations):
    """Agrega metricas del pedido y recupera vendedor/categoria principales."""
    order_items = order_items.copy()
    products = products.copy()
    sellers = sellers.copy()

    order_items["shipping_limit_date"] = pd.to_datetime(order_items["shipping_limit_date"])
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    products = products.merge(translations, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )

    item_level = order_items.merge(
        products[
            [
                "product_id",
                "category",
                "product_weight_g",
                "product_volume_cm3",
            ]
        ],
        on="product_id",
        how="left",
    )
    item_level = item_level.merge(
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ],
        on="seller_id",
        how="left",
    )

    order_aggregates = item_level.groupby("order_id").agg(
        order_item_count=("order_item_id", "count"),
        product_count=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        order_total_price=("price", "sum"),
        order_total_freight=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("product_volume_cm3", "sum"),
        last_shipping_limit_date=("shipping_limit_date", "max"),
    ).reset_index()

    primary_seller = _get_primary_value_by_order(
        item_level,
        value_column="seller_id",
        count_column="order_item_id",
        amount_column="price",
    ).rename(columns={"seller_id": "primary_seller_id"})
    primary_seller = primary_seller.merge(
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ],
        left_on="primary_seller_id",
        right_on="seller_id",
        how="left",
    ).drop(columns=["seller_id"])

    primary_category = _get_primary_value_by_order(
        item_level,
        value_column="category",
        count_column="order_item_id",
        amount_column="price",
    ).rename(columns={"category": "primary_category"})

    order_features = order_aggregates.merge(primary_seller, on="order_id", how="left")
    order_features = order_features.merge(primary_category, on="order_id", how="left")

    return order_features


def _get_primary_value_by_order(item_level, value_column, count_column, amount_column):
    """Selecciona el valor dominante por orden usando cantidad y monto como criterio."""
    ranked_values = item_level.groupby(["order_id", value_column]).agg(
        item_count=(count_column, "count"),
        total_amount=(amount_column, "sum"),
    ).reset_index()

    ranked_values = ranked_values.sort_values(
        ["order_id", "item_count", "total_amount"],
        ascending=[True, False, False],
    )

    return ranked_values.drop_duplicates("order_id")[["order_id", value_column]]


def _attach_geolocation_features(delivery_data, geolocation_reference):
    """Adjunta lat/lng de cliente y vendedor principal."""
    geolocation_reference = geolocation_reference.rename(
        columns={
            "geolocation_zip_code_prefix": "zip_code_prefix",
            "latitude": "lat",
            "longitude": "lng",
        }
    )

    delivery_data["customer_zip_code_prefix"] = (
        delivery_data["customer_zip_code_prefix"].astype(str).str.zfill(5)
    )
    delivery_data["seller_zip_code_prefix"] = (
        delivery_data["seller_zip_code_prefix"].astype(str).str.zfill(5)
    )

    customer_geo = geolocation_reference.rename(
        columns={
            "zip_code_prefix": "customer_zip_code_prefix",
            "lat": "customer_lat",
            "lng": "customer_lng",
        }
    )
    seller_geo = geolocation_reference.rename(
        columns={
            "zip_code_prefix": "seller_zip_code_prefix",
            "lat": "seller_lat",
            "lng": "seller_lng",
        }
    )

    delivery_data = delivery_data.merge(customer_geo, on="customer_zip_code_prefix", how="left")
    delivery_data = delivery_data.merge(seller_geo, on="seller_zip_code_prefix", how="left")

    return delivery_data


def _finalize_delivery_dataset(delivery_data):
    """Calcula columnas finales y conserva solo las variables relevantes."""
    delivery_data["carrier_sla_days"] = (
        delivery_data["last_shipping_limit_date"] - delivery_data["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    delivery_data["carrier_sla_days"] = delivery_data["carrier_sla_days"].clip(lower=0)
    delivery_data["delay_days"] = (
        delivery_data["delivery_days"] - delivery_data["promised_delivery_days"]
    )
    delivery_data["order_density_g_cm3"] = (
        delivery_data["total_weight_g"] / delivery_data["total_volume_cm3"].replace(0, np.nan)
    )
    delivery_data["multiple_sellers"] = (delivery_data["seller_count"] > 1).astype(int)

    selected_columns = [
        "order_id",
        "delivery_days",
        "promised_delivery_days",
        "delay_days",
        "is_late",
        "approval_days",
        "carrier_sla_days",
        "order_item_count",
        "product_count",
        "seller_count",
        "multiple_sellers",
        "order_total_price",
        "order_total_freight",
        "avg_item_price",
        "total_weight_g",
        "total_volume_cm3",
        "order_density_g_cm3",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
        "primary_category",
        "customer_lat",
        "customer_lng",
        "seller_lat",
        "seller_lng",
        "distance_km",
        "seller_customer_same_state",
    ]

    return delivery_data[selected_columns]
