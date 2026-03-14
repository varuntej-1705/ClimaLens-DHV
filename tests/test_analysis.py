import unittest
import pandas as pd
import json
import os
import sys

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis import (
    get_temperature_trends,
    get_humidity_trends,
    detect_heatwaves,
    get_risk_distribution,
    get_monthly_temperature_distribution
)

class TestAnalysisTools(unittest.TestCase):
    """
    Test suite for the data visualization and analysis functions in utils/analysis.py.
    Validates that the Plotly JSON structures and data processing handle various
    DataFrame shapes correctly, including empty sets and edge cases.
    """

    def setUp(self):
        """Prepare mock data for tests."""
        self.mock_data = pd.DataFrame({
            'City': ['Delhi', 'Delhi', 'Mumbai', 'Mumbai'],
            'Date': ['2023-05-01', '2023-05-02', '2023-05-01', '2023-05-02'],
            'Temperature': [40, 42, 35, 36],
            'Humidity': [30, 25, 60, 65],
            'Heat_Risk_Category': ['High Risk', 'High Risk', 'Medium Risk', 'Medium Risk']
        })

    def test_temperature_trends_all_cities(self):
        """Test temperature trend chart generation for all cities."""
        result = get_temperature_trends(self.mock_data, city='All')
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertIn('layout', result)
        
        # We expect 2 traces (Delhi, Mumbai)
        self.assertEqual(len(result['data']), 2)

    def test_temperature_trends_single_city(self):
        """Test temperature trend chart generation for a specific city."""
        result = get_temperature_trends(self.mock_data, city='Delhi')
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['data']), 1)
        self.assertEqual(result['data'][0]['name'], 'Temperature')

    def test_humidity_trends_all_cities(self):
        """Test humidity trends aggregate calculation."""
        result = get_humidity_trends(self.mock_data, city='All')
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['data']), 1)

    def test_detect_heatwaves_threshold(self):
        """Test heatwave detection thresholding logic."""
        # 38 threshold: Delhi has 2, Mumbai has 0
        result = detect_heatwaves(self.mock_data, threshold=38)
        self.assertIsInstance(result, dict)
        
        # Check that x-axis data for heatwaves is correctly calculated
        bars = result['data'][0]
        self.assertIn('x', bars)
        # Delhi should have 2 days >= 38
        self.assertIn(2, bars['x'])

    def test_risk_distribution_valid_data(self):
        """Test pie chart generation for risk distribution."""
        result = get_risk_distribution(self.mock_data)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['data'][0]['type'], 'pie')

    def test_risk_distribution_missing_column(self):
        """Test graceful degradation if risk category column is missing."""
        bad_df = self.mock_data.drop(columns=['Heat_Risk_Category'])
        result = get_risk_distribution(bad_df)
        self.assertIsNone(result)

    def test_monthly_temp_distribution(self):
        """Test box plot generation logic."""
        result = get_monthly_temperature_distribution(self.mock_data)
        self.assertIsInstance(result, dict)
        # Only 'May' is present in the mock_data, so only 1 trace is created
        self.assertEqual(len(result['data']), 1)

    def test_massive_dataset_ingestion(self):
        """
        Validates the visualization layers against an artificially inflated
        dataset simulating years of historical data to ensure plotly json
        rendering remains stable and does not crash the layout properties.
        """
        # Generate 5000 rows of mock data
        dates = pd.date_range(start='2020-01-01', periods=5000, freq='D')
        large_df = pd.DataFrame({
            'City': ['Delhi'] * 2500 + ['Mumbai'] * 2500,
            'Date': dates,
            'Temperature': np.random.normal(30, 10, 5000),
            'Humidity': np.random.uniform(20, 90, 5000),
            'Heat_Risk_Category': np.random.choice(['Low Risk', 'Medium Risk', 'High Risk'], 5000)
        })
        
        trends = get_temperature_trends(large_df, city='All')
        self.assertIsNotNone(trends)
        self.assertIn('data', trends)
        
        dist = get_risk_distribution(large_df)
        self.assertIsNotNone(dist)
        self.assertIn('data', dist)

    def test_malformed_date_formats(self):
        """
        Ensures the analysis tools can cast weird date formats that 
        might be returned by unpredictable upstream API providers.
        """
        bad_df = pd.DataFrame({
            'City': ['Chennai'],
            'Date': ['12/31/2023 23:59:59 PM'], # non-standard string
            'Temperature': [39],
            'Humidity': [80]
        })
        
        # It should coerce or handle it without crashing
        result = get_temperature_trends(bad_df, city='Chennai')
        self.assertIn('data', result)

    def test_zero_variance_datasets(self):
        """
        Checks rendering behavior when there is absolutely zero variance
        in weather over time (static simulator inputs).
        """
        flat_df = pd.DataFrame({
            'City': ['Pune'] * 10,
            'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'Temperature': [25.0] * 10,
            'Humidity': [50.0] * 10,
            'Heat_Risk_Category': ['Low Risk'] * 10
        })
        
        result = get_temperature_trends(flat_df)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
