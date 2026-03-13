import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv('CLIMATE_API_KEY', '')

INDIAN_CITIES = [
    "New Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Ahmedabad", "Pune", "Surat", "Jaipur",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Surat"
]

_DATA_CACHE = {}
_CACHE_TTL = 600  # Increase to 10 minutes

def get_live_city_data(city_query):
    query_key = str(city_query).lower().strip()
    
    # Return cached data if available and fresh
    if query_key in _DATA_CACHE:
        cache_entry = _DATA_CACHE[query_key]
        if time.time() - cache_entry['timestamp'] < _CACHE_TTL:
            return cache_entry['data']

    # Fetch fresh data
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city_query}&aqi=yes"
    try:
        response = requests.get(url, timeout=5) # Shorter timeout for faster failover
        if response.status_code == 200:
            data = response.json()
            loc, current = data['location'], data['current']
            aqi_data = current.get('air_quality', {})
            
            epa_index = aqi_data.get('us-epa-index', 1)
            aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250, 6: 400}
            estimated_aqi = aqi_map.get(epa_index, 100)
            
            processed_data = {
                'City': loc['name'],
                'Region': loc['region'],
                'Country': loc['country'],
                'LocalTime': loc['localtime'],
                'Latitude': loc['lat'],
                'Longitude': loc['lon'],
                'Temperature': current['temp_c'],
                'FeelsLike': current['feelslike_c'],
                'Humidity': current['humidity'],
                'Condition': current['condition']['text'],
                'Icon': current['condition']['icon'],
                'Wind_kph': current['wind_kph'],
                'UV': current['uv'],
                'AQI': estimated_aqi,
                'PM25': round(aqi_data.get('pm2_5', 0), 1),
                'PM10': round(aqi_data.get('pm10', 0), 1),
                'NO2': round(aqi_data.get('no2', 0), 1),
                'Ozone': round(aqi_data.get('o3', 0), 1),
                'CO': round(aqi_data.get('co', 0), 1),
            }
            
            heat_risk = min(10, max(0, ((processed_data['FeelsLike'] - 25) / 2) + (processed_data['UV'] * 0.5)))
            processed_data['Heat_Risk_Score'] = round(heat_risk, 1)
            processed_data['Heat_Risk_Category'] = 'High Risk' if processed_data['FeelsLike'] > 40 else 'Medium Risk' if processed_data['FeelsLike'] > 32 else 'Low Risk'
            
            _DATA_CACHE[query_key] = {'timestamp': time.time(), 'data': processed_data}
            return processed_data
            
    except Exception as e:
        print(f"Fetch failed for {city_query}: {e}")
        # Fallback to expired cache if available
        if query_key in _DATA_CACHE:
            return _DATA_CACHE[query_key]['data']
            
    return {"error": "Connection Latency detected", "City": city_query}

def get_bulk_indian_cities_data():
    """Parallelized fetching to drastically reduce latency."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_live_city_data, INDIAN_CITIES))
    return [r for r in results if "error" not in r]

