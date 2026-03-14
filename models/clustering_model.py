"""
Climora — Machine Learning Clustering Engine (Live Data Adaptation)
===================================================================

This module implements the core artificial intelligence engine for the 
Climora application. It utilizes unsupervised machine learning techniques 
to classify urban areas into dynamic heat vulnerability risk levels 
based on a composite synthesis of meteorological and ambient data.

Algorithm Pipeline:
1. Feature Scaling (Standardization) to normalize disparate metrics (Temp, AQI, Wind).
2. Principal Component Analysis (PCA) for dimensionality reduction to isolate the 
   most significant variance axes (usually Heat/Wind vs Humidity/AQI).
3. K-Means Clustering on the reduced feature space to dynamically group 
   cities sharing similar risk profiles without relying on static thresholds.

Version: 2.1.0
Author: Climora Data Science Team
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Any, Optional
import json



def prepare_live_features(df):
    """
    Prepare features for K-Means clustering from the preprocessed DataFrame.
    Features: Temperature, Humidity, AQI, UV, Wind_kph
    """
    if df is None or df.empty:
        return None, None, None
            
    feature_cols = ['Temperature', 'Humidity', 'AQI', 'UV', 'Wind_kph']
    # Ensure columns exist (they should from process_live_data)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    features = df[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return df, features_scaled, feature_cols


def apply_pca(features_scaled, n_components=2):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    explained_variance = pca.explained_variance_ratio_

    return pca_result, explained_variance


def apply_kmeans(features_scaled, n_clusters=3):
    """Apply K-Means clustering."""
    # Handle case with fewer cities than clusters
    actual_clusters = min(n_clusters, len(features_scaled))
    if actual_clusters < 1:
        return [], None
        
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return cluster_labels, kmeans


def map_clusters_to_risk(df, cluster_labels):
    """
    Map numeric cluster labels to risk categories based on
    average Heat_Risk_Score per cluster.
    """
    df = df.copy()
    df['Cluster'] = cluster_labels

    # Calculate mean heat risk score per cluster
    cluster_risk = df.groupby('Cluster')['Heat_Risk_Score'].mean()

    # Sort clusters by risk: highest risk score → 'High Risk'
    sorted_clusters = cluster_risk.sort_values(ascending=False).index.tolist()
    risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']

    cluster_to_risk = {}
    for i, cluster_id in enumerate(sorted_clusters):
        if i < len(risk_labels):
            cluster_to_risk[cluster_id] = risk_labels[i]
        else:
            cluster_to_risk[cluster_id] = 'Medium Risk'

    df['ML_Risk_Level'] = df['Cluster'].map(cluster_to_risk)

    return df, cluster_to_risk


def get_clustering_scatter(df, pca_result):
    """
    Generate scatter plot of PCA-reduced clustering results using Plotly.
    Uses the new light premium theme.
    Returns Plotly JSON.
    """
    color_map = {'High Risk': '#EF4444', 'Medium Risk': '#F59E0B', 'Low Risk': '#10B981'}

    fig = go.Figure()

    for risk_level in ['High Risk', 'Medium Risk', 'Low Risk']:
        mask = df['ML_Risk_Level'] == risk_level
        if len(df[mask]) > 0:
            fig.add_trace(go.Scatter(
                x=pca_result[mask, 0],
                y=pca_result[mask, 1],
                mode='markers+text',
                name=risk_level,
                text=df[mask]['City'],
                textposition='top center',
                textfont={'size': 11, 'color': 'rgba(255,255,255,0.7)'},
                marker={
                    'size': 16,
                    'color': color_map[risk_level],
                    'line': {'width': 1, 'color': 'rgba(255,255,255,0.3)'},
                    'opacity': 0.9,
                },
            ))

    fig.update_layout(
        template='plotly_white', 
        paper_bgcolor='white',
        plot_bgcolor='white',
        title={
            'text': 'K-Means Clustering: Heat Vulnerability Dimensions (PCA)',
            'font': {'size': 18, 'color': '#1F2937', 'family': 'Inter'}
        },
        xaxis={
            'title': 'Principal Component 1 (Heat vs Wind)',
            'gridcolor': '#E5E7EB',
            'color': '#6B7280',
            'zeroline': False,
        },
        yaxis={
            'title': 'Principal Component 2 (Humidity vs AQI)',
            'gridcolor': '#E5E7EB',
            'color': '#6B7280',
            'zeroline': False,
        },
        legend={'font': {'color': '#6B7280', 'size': 13}},
        margin={'l': 40, 'r': 20, 't': 50, 'b': 40},
        height=480,
    )

    return json.loads(fig.to_json())


def get_risk_by_temperature_aqi(df):
    """
    Scatter plot of Live Temperature vs AQI colored by ML risk level.
    """
    color_map = {'High Risk': '#EF4444', 'Medium Risk': '#F59E0B', 'Low Risk': '#10B981'}

    fig = go.Figure()

    for risk_level in ['High Risk', 'Medium Risk', 'Low Risk']:
        mask = df['ML_Risk_Level'] == risk_level
        if len(df[mask]) > 0:
            fig.add_trace(go.Scatter(
                x=df[mask]['Temperature'],
                y=df[mask]['AQI'],
                mode='markers+text',
                name=risk_level,
                text=df[mask]['City'],
                textposition='top center',
                textfont={'size': 11, 'color': 'rgba(255,255,255,0.7)'},
                marker={
                    'size': 18,
                    'color': color_map[risk_level],
                    'line': {'width': 1, 'color': 'rgba(255,255,255,0.3)'},
                    'symbol': 'circle',
                },
            ))

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        title={
            'text': 'Live Temperature vs AQI mapped to ML Clusters',
            'font': {'size': 18, 'color': '#1F2937', 'family': 'Inter'}
        },
        xaxis={'title': 'Current Temperature (°C)', 'gridcolor': '#E5E7EB', 'color': '#6B7280'},
        yaxis={'title': 'Current AQI Estimate', 'gridcolor': '#E5E7EB', 'color': '#6B7280'},
        legend={'font': {'color': '#6B7280', 'size': 13}},
        margin={'l': 40, 'r': 20, 't': 50, 'b': 40},
        height=480,
    )

    return json.loads(fig.to_json())


def generate_live_clustering_charts(df):
    """
    Runs the full K-Means pipeline on the fly using live API data.
    Returns the two Plotly JSON charts for the frontend.
    """
    df, features_scaled, feature_cols = prepare_live_features(df)
    if df is None or df.empty:
        return {"error": "No data"}
        
    pca_result, explained_variance = apply_pca(features_scaled)
    cluster_labels, kmeans_model = apply_kmeans(features_scaled)
    
    if len(cluster_labels) == 0:
        return {"error": "Not enough data points for clustering"}
        
    df, cluster_map = map_clusters_to_risk(df, cluster_labels)

    scatter_chart = get_clustering_scatter(df, pca_result)
    temp_aqi_chart = get_risk_by_temperature_aqi(df)
    
    # Calculate group averages for the UI
    high_risk_cities = len(df[df['ML_Risk_Level'] == 'High Risk'])

    return {
        'status': 'success',
        'scatter_chart': scatter_chart,
        'temp_aqi_chart': temp_aqi_chart,
        'high_risk_count': high_risk_cities,
        'total_analyzed': len(df)
    }

class HistoricalBacktester:
    """
    Advanced offline tool for analyzing how the K-Means clustering model
    would have performed on historical heatwave data over previous years.
    
    This class is intended to be used by the `manage.py evaluate-model`
    invocation and is not part of the active web request lifecycle.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_historical_data(self) -> bool:
        """Attempt to load big data from local storage."""
        import os
        if not os.path.exists(self.data_path):
            return False
            
        try:
            self.raw_data = pd.read_csv(self.data_path)
            return True
        except Exception as e:
            print(f"Data load error: {str(e)}")
            return False
            
    def run_temporal_analysis(self, target_variable: str = 'Temperature') -> Dict[str, Any]:
        """
        Runs the K-Means pipeline iteratively over sliced temporal windows 
        to observe cluster drift over seasons.
        """
        if self.raw_data is None or self.raw_data.empty:
            return {"error": "No data loaded"}
            
        results = []
        
        # Simulate temporal chunking (assuming 'Date' or 'timestamp' exists)
        date_col = 'timestamp' if 'timestamp' in self.raw_data.columns else 'Date'
        
        if date_col in self.raw_data.columns:
            self.raw_data[date_col] = pd.to_datetime(self.raw_data[date_col], errors='coerce')
            self.raw_data = self.raw_data.dropna(subset=[date_col])
            self.raw_data = self.raw_data.sort_values(by=date_col)
            
            # Simple chunking logic
            chunk_size = len(self.raw_data) // 4
            chunks = [self.raw_data.iloc[i:i + chunk_size] for i in range(0, len(self.raw_data), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 3:
                    try:
                        features = chunk[['Temperature', 'Humidity', 'AQI', 'UV', 'Wind_kph']].fillna(0).values
                        scaler = StandardScaler()
                        features_scaled = scaler.fit_transform(features)
                        
                        labels, _ = apply_kmeans(features_scaled, n_clusters=3)
                        results.append({
                            "window": i,
                            "records": len(chunk),
                            "avg_temp": float(chunk['Temperature'].mean())
                        })
                    except Exception as e:
                        pass
                        
        return {"status": "success", "temporal_drift": results}
