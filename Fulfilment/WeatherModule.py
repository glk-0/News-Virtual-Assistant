#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""# Weater intent fulfillment"""

import requests
from datetime import datetime
from Fulfilment.Helpers import _parse_date

def get_coordinates(location: str):
    """Geocodes a location string into lat/lon coordinates."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": location, "count": 1}
    response = requests.get(url, params=params)

    if response.status_code == 200 and "results" in response.json():
        return response.json()["results"][0]
    return None

def GetWeather(slots: dict) -> dict:
    """
    Fulfills the GetWeather intent by querying Open-Meteo.
    Slots: LOCATION (optional list), defaults to Ottawa.
    """
    # Consistency: treat slot as a list and join it
    location_list = slots.get("LOCATION", ["Ottawa"])
    location = " ".join(location_list) if isinstance(location_list, list) else location_list

    raw_date = slots.get("DATE", datetime.date.today().strftime("%Y-%m-%d") )
    asked_date = _parse_date(raw_date, format=1)
    print(asked_date)
    coords = get_coordinates(location)
    if not coords:
        return {"intent": "GetWeather", "error": f"Could not find location '{location}'."}

    lat, lon = coords["latitude"], coords["longitude"]
    city = coords["name"]

    # Clean Open-Meteo call (No API key needed)
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code",
        "timezone": "auto",
        "start_date": asked_date,
        "end_date": asked_date
    })

    if resp.status_code != 200:
        return {"intent": "GetWeather", "error": "Weather API is currently unreachable."}

    data = resp.json()

    return {
        "intent": "GetWeather",
        "location": city,
        "temperature": data["current"]["temperature_2m"],
        "weather_code": data["current"]["weather_code"],
        "unit": "Celsius"
    }

