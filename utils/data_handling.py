"""
ClimaLens — In-Memory Data Pipeline
Prepares live WeatherAPI data using Pandas for preprocessing and feature engineering
before it reaches the ML and visualization layers.
"""

import pandas as pd
import numpy as np
from scipy import stats

def process_live_data(bulk_data):
    """
    Advanced Data Pipeline: API -> Pandas -> Cleaning -> Normalization -> Feature Engineering.
    Meets course requirements for data handling and statistical analysis.
    """
    if not bulk_data:
        return pd.DataFrame()

    # 1. API to DataFrame
    df = pd.DataFrame(bulk_data)

    # 2. Cleaning: Remove duplicates and handle missing values
    df = df.drop_duplicates(subset=['City'])
    
    numeric_cols = ['Temperature', 'FeelsLike', 'Humidity', 'AQI', 'PM25', 'PM10', 'NO2', 'Ozone', 'UV', 'Wind_kph', 'CO']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Handle missing values via mean imputation (standard course technique)
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = 0.0

    # 3. Filtering unrealistic values
    df = df[(df['Temperature'] >= -10) & (df['Temperature'] <= 55)]
    df = df.reset_index(drop=True)

    # 4. Outlier Detection (Z-score > 3)
    # Course requirement: Demonstrate statistical outlier detection
    for col in ['Temperature', 'AQI']:
        z_col = f'{col}_ZScore'
        df[z_col] = np.abs(stats.zscore(df[col]))
        df[f'Is_{col}_Outlier'] = df[z_col] > 3

    # 5. Min-Max Normalization (Feature Scaling)
    for col in ['Temperature', 'Humidity', 'AQI']:
        c_min = df[col].min()
        c_max = df[col].max()
        if c_max > c_min:
            df[f'{col}_Normalized'] = (df[col] - c_min) / (c_max - c_min)
        else:
            df[f'{col}_Normalized'] = 0.5

    # 6. FEATURE ENGINEERING (Derived Features)
    
    # Heatwave Flag: 1 if Temp > 35, else 0
    df['Heatwave_Flag'] = (df['Temperature'] > 35).astype(int)
    
    # Heat Risk Score: (Temperature * 0.5) + (AQI * 0.3) + (Humidity * 0.2)
    # We use normalized values for a consistent 0-1 range, then scale to 0-100
    df['Heat_Risk_Score'] = (
        (df['Temperature_Normalized'] * 0.5) + 
        (df['AQI_Normalized'] * 0.3) + 
        (df['Humidity_Normalized'] * 0.2)
    ) * 100
    df['Heat_Risk_Score'] = df['Heat_Risk_Score'].round(1)

    # Pollution Category (Mapping AQI)
    conditions = [
        df['AQI'] <= 50,
        df['AQI'] <= 100,
        df['AQI'] <= 200,
        df['AQI'] > 200
    ]
    labels = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']
    df['Pollution_Category'] = np.select(conditions, labels, default='Moderate')

    # Temperature Anomaly: Deviation from current national mean
    national_mean = df['Temperature'].mean()
    df['Temperature_Anomaly'] = (df['Temperature'] - national_mean).round(2)

    return df


# ─── Leaderboard Formatting (Using Processed DF) ──────────────────────

def prepare_live_leaderboards(df):
    """
    Takes the fully processed Pandas DataFrame and
    prepares sorted leaderboards for the UI.
    """
    if df.empty:
        return {"highest_risk": [], "lowest_risk": [], "hottest": [], "coolest": []}
        
    df_sorted_risk = df.sort_values(by='Heat_Risk_Score', ascending=False)
    
    highest_risk = df_sorted_risk.head(5).to_dict('records')
    lowest_risk = df_sorted_risk.tail(5).iloc[::-1].to_dict('records')
    
    df_sorted_temp = df.sort_values(by='Temperature', ascending=False)
    hottest = df_sorted_temp.head(5).to_dict('records')
    coolest = df_sorted_temp.tail(5).iloc[::-1].to_dict('records')
    
    def format_card(city_row, primary_metric, secondary_metric_label):
        val = city_row[primary_metric]
        value_suffix = " / 100" if primary_metric == 'Heat_Risk_Score' else " °C"
        return {
            "city": city_row['City'],
            "state": city_row['Region'],
            "value": f"{val}",
            "secondary": f"{secondary_metric_label}: {city_row[secondary_metric_label]}",
            "condition": city_row['Condition']
        }
    
    return {
        "highest_risk": [format_card(c, 'Heat_Risk_Score', 'AQI') for c in highest_risk],
        "lowest_risk": [format_card(c, 'Heat_Risk_Score', 'AQI') for c in lowest_risk],
        "hottest": [format_card(c, 'Temperature', 'Humidity') for c in hottest],
        "coolest": [format_card(c, 'Temperature', 'Humidity') for c in coolest]
    }


def get_live_summary_stats(df):
    """Calculate high-level averages from the processed DataFrame."""
    if df.empty:
        return {}
        
    avg_temp = df['Temperature'].mean()
    avg_aqi = df['AQI'].mean()
    high_risk_count = len(df[df['Heat_Risk_Category'] == 'High Risk'])
    
    return {
        "avg_temp": round(avg_temp, 1),
        "avg_aqi": round(avg_aqi, 0),
        "monitored_cities": len(df),
        "high_risk_cities": high_risk_count
    }
