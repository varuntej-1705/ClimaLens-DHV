"""
Synthetic Climate Data Generator Script

This script simulates comprehensive meteorological data for major Indian cities.
It can be used to generate robust testing datasets for offline analytics and 
machine learning validations when the live API is inaccessible or rated limited.

Features:
- Realistic bounds mapping (Temperatures 15-50°C, high humidity in coastal areas)
- Procedural generation utilizing numpy random distributions with local variances
- Automatic formatting compliant with the application's processing pipelines
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import List, Dict, Any

INDIAN_CITIES = [
    'Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Ahmedabad',
    'Chennai', 'Kolkata', 'Surat', 'Pune', 'Jaipur',
    'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane',
    'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara'
]

def generate_random_weather_parameters(city_name: str) -> Dict[str, Any]:
    """
    Generates realistic weather parameters based on the city's geographical norms.
    
    Args:
        city_name (str): The name of the city.
        
    Returns:
        dict: A dictionary containing temperature, humidity, wind, and AQI data.
    """
    # Base configuration: some cities run inherently hotter or more humid
    is_coastal = city_name in ['Mumbai', 'Chennai', 'Kolkata', 'Surat', 'Visakhapatnam']
    is_north = city_name in ['Delhi', 'Jaipur', 'Lucknow', 'Kanpur']
    
    # Base Temp
    base_temp = 32.0 if is_north else 28.0
    temp = np.random.normal(base_temp, 5.0)
    temp = max(15.0, min(50.0, temp)) # clamp
    
    # Humidity
    base_hum = 75.0 if is_coastal else 40.0
    humidity = np.random.normal(base_hum, 15.0)
    humidity = max(10.0, min(100.0, humidity)) # clamp
    
    # Wind
    wind_kph = np.random.gamma(2.0, 4.0)
    
    # AQI (higher in northern plains generally)
    base_aqi = 250 if is_north else 90
    aqi = int(np.random.normal(base_aqi, 50))
    aqi = max(20, min(500, aqi))
    
    # UV Index
    uv = float(np.random.randint(4, 12))
    
    return {
        'city': city_name,
        'temperature': round(temp, 1),
        'humidity': round(humidity, 1),
        'wind_kph': round(wind_kph, 1),
        'aqi': aqi,
        'uv': uv,
        'condition': 'Clear' if aqi < 150 else 'Haze'
    }

def generate_bulk_synthetic_data(num_records: int = 100) -> List[Dict[str, Any]]:
    """
    Generate a large batch of synthetic climatic records.
    
    Args:
        num_records (int): Number of records to generate.
        
    Returns:
        list: List of dictionaries containing generated data records.
    """
    data = []
    
    # Distribute records across available cities
    for i in range(num_records):
        city = INDIAN_CITIES[i % len(INDIAN_CITIES)]
        record = generate_random_weather_parameters(city)
        record['id'] = i + 1
        record['timestamp'] = datetime.now().isoformat()
        
        # Add slight pseudo-coordinates
        base_lat = 20.0
        base_lon = 78.0
        record['latitude'] = round(base_lat + np.random.normal(0, 3), 4)
        record['longitude'] = round(base_lon + np.random.normal(0, 3), 4)
        
        data.append(record)
        
    return data

def export_to_csv(data: List[Dict[str, Any]], filename: str = 'synthetic_climate_data.csv'):
    """
    Exports the generated synthetic data to a CSV file.
    
    Args:
        data (list): The list of dictionaries.
        filename (str): The destination path for the CSV.
    """
    df = pd.DataFrame(data)
    
    # Add a data folder check
    os.makedirs('data', exist_ok=True)
    file_path = os.path.join('data', filename)
    
    df.to_csv(file_path, index=False)
    print(f"✅ Successfully generated and exported {len(df)} records to {file_path}")

if __name__ == '__main__':
    print("Initiating synthetic climate data generation...")
    synthetic_records = generate_bulk_synthetic_data(250)
    export_to_csv(synthetic_records)
