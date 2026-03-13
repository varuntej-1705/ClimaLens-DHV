"""
Climora — ML Clustering Model (Live Data Adaptation)
PCA dimensionality reduction + K-Means clustering to classify
urban areas into heat vulnerability risk levels.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
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
        template='none', # Manual dark theme
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': 'K-Means Clustering: Heat Vulnerability Dimensions (PCA)',
            'font': {'size': 18, 'color': 'white', 'family': 'Inter'}
        },
        xaxis={
            'title': 'Principal Component 1 (Heat vs Wind)',
            'gridcolor': 'rgba(255,255,255,0.05)',
            'color': 'rgba(255,255,255,0.5)',
            'zeroline': False,
        },
        yaxis={
            'title': 'Principal Component 2 (Humidity vs AQI)',
            'gridcolor': 'rgba(255,255,255,0.05)',
            'color': 'rgba(255,255,255,0.5)',
            'zeroline': False,
        },
        legend={'font': {'color': 'rgba(255,255,255,0.6)', 'size': 13}},
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
        template='none', # Manual dark theme
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': 'Live Temperature vs AQI mapped to ML Clusters',
            'font': {'size': 18, 'color': 'white', 'family': 'Inter'}
        },
        xaxis={'title': 'Current Temperature (°C)', 'gridcolor': 'rgba(255,255,255,0.05)', 'color': 'rgba(255,255,255,0.5)'},
        yaxis={'title': 'Current AQI Estimate', 'gridcolor': 'rgba(255,255,255,0.05)', 'color': 'rgba(255,255,255,0.5)'},
        legend={'font': {'color': 'rgba(255,255,255,0.6)', 'size': 13}},
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
