import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clustering_model import (
    prepare_live_features,
    apply_pca,
    apply_kmeans,
    map_clusters_to_risk,
    generate_live_clustering_charts
)

class TestClusteringModel(unittest.TestCase):
    """
    Test suite for the K-Means clustering model pipeline.
    Ensures that data is properly scaled, reduced, clustered, and mapped
    to categorical risk levels accurately.
    """

    def setUp(self):
        """Prepare mock features for testing."""
        self.mock_df = pd.DataFrame({
            'City': ['A', 'B', 'C', 'D'],
            'Temperature': [45, 42, 30, 28],
            'Humidity': [20, 25, 80, 85],
            'AQI': [300, 250, 50, 40],
            'UV': [12, 10, 4, 3],
            'Wind_kph': [5, 6, 20, 25],
            'Heat_Risk_Score': [90, 80, 20, 10]  # Used by map_clusters_to_risk
        })

    def test_prepare_live_features_valid(self):
        """Test scaling and feature extraction."""
        df, features_scaled, cols = prepare_live_features(self.mock_df.copy())
        
        self.assertIsNotNone(df)
        self.assertEqual(features_scaled.shape, (4, 5))  # 4 rows, 5 features
        
        # Standard scaler should mean-center to approx 0
        self.assertAlmostEqual(np.mean(features_scaled[:, 0]), 0, places=5)

    def test_prepare_live_features_empty(self):
        """Test handling of empty DataFrames."""
        df, features, cols = prepare_live_features(pd.DataFrame())
        self.assertIsNone(df)
        self.assertIsNone(features)

    def test_apply_pca(self):
        """Test PCA dimensionality reduction correctly yields requested components."""
        _, features_scaled, _ = prepare_live_features(self.mock_df.copy())
        
        pca_result, variance = apply_pca(features_scaled, n_components=2)
        self.assertEqual(pca_result.shape, (4, 2))
        self.assertEqual(len(variance), 2)

    def test_apply_kmeans(self):
        """Test K-Means clustering returns valid cluster indices."""
        _, features_scaled, _ = prepare_live_features(self.mock_df.copy())
        
        labels, model = apply_kmeans(features_scaled, n_clusters=2)
        # Should return labels 0 or 1 for the 4 rows
        self.assertEqual(len(labels), 4)
        self.assertTrue(set(labels).issubset({0, 1}))

    def test_apply_kmeans_few_samples(self):
        """Test K-Means when n_clusters > n_samples."""
        df = self.mock_df.head(2).copy()
        _, features_scaled, _ = prepare_live_features(df)
        
        # Request 3 clusters, but only 2 samples exist
        labels, model = apply_kmeans(features_scaled, n_clusters=3)
        # It should cap clusters to 2 implicitly
        self.assertEqual(len(set(labels)), 2)

    def test_map_clusters_to_risk(self):
        """Test mapping from abstract cluster IDs to qualitative risk levels."""
        df, features_scaled, _ = prepare_live_features(self.mock_df.copy())
        labels, _ = apply_kmeans(features_scaled, n_clusters=2)
        
        mapped_df, cluster_to_risk = map_clusters_to_risk(df, labels)
        
        self.assertIn('ML_Risk_Level', mapped_df.columns)
        # Because we provided Heat_Risk_Score (90/80 for generic "High Risk" candidates vs 20/10)
        # it should successfully assign 'High Risk' to the top cluster.
        self.assertIn('High Risk', mapped_df['ML_Risk_Level'].values)

    def test_full_pipeline_generate_live_clustering_charts(self):
        """Test the end-to-end chart generation function."""
        result = generate_live_clustering_charts(self.mock_df.copy())
        
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'success')
        self.assertIn('scatter_chart', result)
        self.assertIn('temp_aqi_chart', result)
        self.assertEqual(result['total_analyzed'], 4)

    def test_pipeline_handles_missing_columns(self):
        """Test the pipeline's robustness against missing dataset columns."""
        # Drop a feature column to simulate API data loss
        df = self.mock_df.drop(columns=['UV']).copy()
        
        # prepare_live_features should impute 0.0 for missing features
        processed_df, features_scaled, cols = prepare_live_features(df)
        
        self.assertIsNotNone(processed_df)
        self.assertIn('UV', processed_df.columns)
        self.assertTrue((processed_df['UV'] == 0.0).all())

    def test_pca_variance_bounds(self):
        """Verify PCA explained variance ratios are mathematically sound."""
        _, features_scaled, _ = prepare_live_features(self.mock_df.copy())
        
        # Max components limited by min(n_samples, n_features)
        components = min(features_scaled.shape[0], features_scaled.shape[1])
        pca_result, variance = apply_pca(features_scaled, n_components=components)
        
        # Total variance explained should sum to approximately 1.0
        self.assertAlmostEqual(sum(variance), 1.0, places=2)

    def test_kmeans_deterministic_random_state(self):
        """Verify K-Means model produces consistent clusters due to random_state."""
        _, features_scaled, _ = prepare_live_features(self.mock_df.copy())
        
        labels_run_1, _ = apply_kmeans(features_scaled, n_clusters=2)
        labels_run_2, _ = apply_kmeans(features_scaled, n_clusters=2)
        
        # Should be identical assignments
        np.testing.assert_array_equal(labels_run_1, labels_run_2)

    def test_clustering_charts_error_propagation(self):
        """Verify chart generator gracefully returns errors for empty data."""
        result = generate_live_clustering_charts(pd.DataFrame())
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No data')

if __name__ == '__main__':
    unittest.main()
