import unittest
from unittest.mock import patch, MagicMock
from app import app
import json
import logging

class TestFlaskAPI(unittest.TestCase):
    """
    Test suite for the primary Flask server endpoints.
    These tests execute requests against the test-configured Flask client
    to ensure that all HTML endpoints and JSON APIs respond correctly.
    """

    def setUp(self):
        """Prepare the Flask test client before each test."""
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.get_bulk_indian_cities_data')
    @patch('app.process_live_data')
    def test_api_live_summary_success(self, mock_process, mock_get_bulk):
        """
        Verify the main summary API endpoint correctly aggregates and 
        returns JSON leaderboard and static data.
        """
        mock_get_bulk.return_value = {"mock": "data"}
        mock_process.return_value = MagicMock()
        mock_process.return_value.empty = False
        
        with patch('app.prepare_live_leaderboards') as mock_prep:
            with patch('app.get_live_summary_stats') as mock_stats:
                mock_prep.return_value = {"hottest": [], "coolest": []}
                mock_stats.return_value = {"total_cities": 50}
                
                response = self.app.get('/api/live/summary')
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                
                self.assertEqual(data['status'], 'success')
                self.assertIn('leaderboards', data)
                self.assertIn('stats', data)
                self.assertEqual(data['stats']['total_cities'], 50)

    @patch('app.get_live_city_data')
    def test_api_live_city_valid_request(self, mock_get_city):
        """
        Verify that fetching live data for a specific city returns
        the expected JSON payload.
        """
        mock_get_city.return_value = {
            "city": "Mumbai",
            "temperature": 35.5,
            "humidity": 60,
            "aqi": 120
        }
        
        response = self.app.get('/api/live/city?q=Mumbai')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
        self.assertEqual(data['data']['city'], 'Mumbai')

    @patch('app.get_live_city_data')
    def test_api_live_city_error_propagation(self, mock_get_city):
        """
        Verify that downstream errors from the weather API are
        caught and returned as 400 Bad Request to the client.
        """
        mock_get_city.return_value = {"error": "City API unreachable"}
        
        response = self.app.get('/api/live/city?q=BadCityName')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'City API unreachable')

    def test_index_route(self):
        """Ensure the root route returns the HTML dashboard."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())

    def test_climate_dashboard_route(self):
        """Ensure the climate analysis dashboard route returns correctly."""
        response = self.app.get('/climate')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())

    def test_national_dashboard_route(self):
        """Ensure the national dashboard returns successfully."""
        response = self.app.get('/national')
        self.assertEqual(response.status_code, 200)

    def test_aqi_dashboard_route(self):
        """Ensure the AQI dashboard route returns successfully."""
        response = self.app.get('/aqi')
        self.assertEqual(response.status_code, 200)

    def test_map_dashboard_route(self):
        """Ensure the Leaflet map dashboard route returns successfully."""
        response = self.app.get('/map')
        self.assertEqual(response.status_code, 200)

    def test_heat_dashboard_route(self):
        """Ensure the AI heat vulnerability dashboard returns successfully."""
        response = self.app.get('/heat')
        self.assertEqual(response.status_code, 200)

class TestFlaskEndpointsAdvanced(unittest.TestCase):
    """
    Additional advanced endpoints tests verifying marker generation
    and clustering integration routes to ensure absolute robustness.
    """

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.get_bulk_indian_cities_data')
    @patch('app.process_live_data')
    def test_api_live_map_markers(self, mock_process, mock_get_bulk):
        """Verify Leaflet map marker generation."""
        import pandas as pd
        mock_get_bulk.return_value = []
        mock_df = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Latitude': [28.6, 19.0],
            'Longitude': [77.2, 72.8],
            'Temperature': [40, 35],
            'AQI': [150, 80],
            'Heat_Risk_Category': ['High Risk', 'Low Risk'],
            'Condition': ['Sunny', 'Clear']
        })
        mock_process.return_value = mock_df
        
        response = self.app.get('/api/live/map-markers')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('markers', data)
        self.assertEqual(len(data['markers']), 2)
        self.assertEqual(data['markers'][0]['city'], 'Delhi')
        # High risk -> #ff4757
        self.assertEqual(data['markers'][0]['color'], '#ff4757')

    @patch('app.get_bulk_indian_cities_data')
    @patch('app.process_live_data')
    @patch('app.generate_live_clustering_charts')
    def test_api_live_clustering_success(self, mock_cluster, mock_process, mock_get_bulk):
        """Verify the clustering visualization endpoint handles valid data."""
        mock_get_bulk.return_value = []
        mock_process.return_value = MagicMock()
        mock_cluster.return_value = {
            "status": "success",
            "scatter_chart": {"data": []},
            "temp_aqi_chart": {"data": []}
        }
        
        response = self.app.get('/api/live/clustering')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
        self.assertIn('temp_aqi_chart', data['data'])

    @patch('app.get_bulk_indian_cities_data')
    @patch('app.process_live_data')
    @patch('app.generate_live_clustering_charts')
    def test_api_live_clustering_error(self, mock_cluster, mock_process, mock_get_bulk):
        """Verify the clustering visualization endpoint handles internal model errors."""
        mock_get_bulk.return_value = []
        mock_process.return_value = MagicMock()
        mock_cluster.return_value = {"error": "Insufficient data"}
        
        response = self.app.get('/api/live/clustering')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Insufficient data')

    @patch('app.get_bulk_indian_cities_data')
    @patch('app.process_live_data')
    def test_api_viz_data(self, mock_process, mock_get_bulk):
        """Verify the visualization data dump endpoint."""
        import pandas as pd
        mock_get_bulk.return_value = []
        mock_df = pd.DataFrame({'City': ['Delhi'], 'Temperature': [40]})
        mock_process.return_value = mock_df
        
        response = self.app.get('/api/viz/data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['data']), 1)
        self.assertEqual(data['data'][0]['City'], 'Delhi')

if __name__ == '__main__':
    unittest.main()
