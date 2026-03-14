"""
ML Model Evaluation and Diagnostics Pipeline

This script serves as an offline test bed for the K-Means and PCA implementation
used in Climora's heat vulnerability engine. It performs the following routines:
1. Loads synthetic or cached historical data
2. Executes dimensionality reduction logic manually
3. Provides detailed silhouette scores back to standard out
4. Enables easy algorithm tuning for hyperparameters
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import sys
import os

# Ensure models module is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clustering_model import prepare_live_features, apply_pca, apply_kmeans

def check_data_availability() -> pd.DataFrame:
    """
    Checks if there's local data available to run the evaluations.
    If not, it politely suggests running the data generator script first.
    
    Returns:
        pd.DataFrame: Sourced dataframe if available.
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'synthetic_climate_data.csv')
    
    if not os.path.exists(data_path):
        print(f"⚠️ Simulation data not found at {data_path}.")
        print("Please run `python scripts/data_generator.py` first to create synthetic data for model evaluation.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(data_path)
        # Adapt synthetic data column names to what the process live data expects
        df = df.rename(columns={
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'wind_kph': 'Wind_kph',
            'aqi': 'AQI',
            'uv': 'UV',
            'city': 'City'
        })
        return df
    except Exception as e:
        print(f"❌ Error loading data from {data_path}: {str(e)}")
        return pd.DataFrame()

def evaluate_clustering_quality(features: np.ndarray, labels: np.ndarray, model_name: str = "K-Means"):
    """
    Calculates internal validation metrics for the clustering algorithm
    without needing ground truth targets.
    
    Args:
        features (np.ndarray): The scaled feature matrix.
        labels (np.ndarray): The cluster assignments.
        model_name (str): Label for printing.
    """
    if len(set(labels)) < 2:
        print("⚠️ Cannot calculate silhouette scores with fewer than 2 distinct clusters.")
        return
        
    s_score = silhouette_score(features, labels)
    ch_score = calinski_harabasz_score(features, labels)
    
    print(f"\n--- Model Diagnostics: {model_name} ---")
    print(f"Silhouette Score (-1 to 1): {s_score:.4f} (Closer to 1 is better, indicates dense, separated clusters)")
    print(f"Calinski-Harabasz Index (Variance Ratio): {ch_score:.2f} (Higher is better)")
    
    if s_score < 0.2:
        print("💡 Insight: The clusters are overlapping heavily. Consider adding distinct features or tuning 'n_clusters'.")
    elif s_score > 0.5:
        print("✅ Insight: Strong clustering structure found.")
    else:
        print("📊 Insight: Moderate cluster separation.")

def run_evaluation_pipeline():
    """Main execution block for model evaluation."""
    print("Initiating ML Model Verification Pipeline...")
    
    df = check_data_availability()
    if df.empty:
        return
        
    print(f"✅ Loaded {len(df)} records for training evaluation.")
    
    # Prepare features
    process_df, features_scaled, cols = prepare_live_features(df)
    
    if features_scaled is None:
        print("❌ Feature extraction failed.")
        return
        
    print(f"✅ Extracted Feature Columns: {cols}")
    
    # Evaluate PCA
    pca_comps = 2
    pca_result, variance = apply_pca(features_scaled, n_components=pca_comps)
    print(f"\n--- Dimension Reduction: PCA ---")
    print(f"Retained Variance (Components = {pca_comps}): {sum(variance)*100:.2f}%")
    for i, var in enumerate(variance):
        print(f"  Component {i+1}: {var*100:.2f}%")
        
    # Evaluate Clustering over multiple K bounds
    print("\n--- Hyperparameter Evaluation (K) ---")
    for k in [2, 3, 4, 5]:
        labels, kmeans = apply_kmeans(features_scaled, n_clusters=k)
        if len(set(labels)) == k:  # Ensure the model actually found k clusters
            print(f"Testing k={k} clusters...")
            evaluate_clustering_quality(features_scaled, labels, f"K-Means (k={k})")

if __name__ == '__main__':
    run_evaluation_pipeline()
