import numpy as np


def haversine_distance_km(lat1, lng1, lat2, lng2):
    """
    Calcula distancia geografica en kilometros entre dos pares lat/lng.
    Soporta escalares o arrays de numpy/pandas.
    """
    lat1 = np.radians(lat1)
    lng1 = np.radians(lng1)
    lat2 = np.radians(lat2)
    lng2 = np.radians(lng2)

    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lng / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0
    return earth_radius_km * c
