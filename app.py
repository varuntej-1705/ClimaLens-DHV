"""
Climora — Enterprise Web Interface and API layer
================================================

This module acts as the primary FastAPI/Flask hybrid routing gateway for
the Climora climate intelligence application. It serves both the rendered
HTML templates for the frontend dashboards and provides a robust json API
for asynchronous live data fetching.

Endpoints are broadly grouped into:
1. Standard Page Routes (/, /climate, /heat, etc.)
2. API Routes (/api/live/*) which interact with the upstream data 
   retrieval layers, handle aggregation, and serve JSON.

Version: 2.1.0
"""
import os
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from typing import Dict, Any, Tuple, Union
from dotenv import load_dotenv

from utils.api_fetch import get_live_city_data, get_bulk_indian_cities_data, INDIAN_CITIES
from utils.data_handling import process_live_data, prepare_live_leaderboards, get_live_summary_stats
from models.clustering_model import generate_live_clustering_charts

load_dotenv()

app = Flask(__name__)

# ─── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    """
    New Premium 'Air Secure' Landing Page.
    We just render the template; the JS will immediately fetch /api/live
    to populate the leaderboards dynamically to support auto-refresh.
    """
    return render_template('index.html', cities=sorted(INDIAN_CITIES))


@app.route('/climate')
def climate_dashboard():
    """City-specific Climate Analysis."""
    return render_template('climate_dashboard.html', cities=sorted(INDIAN_CITIES))


@app.route('/national')
def national_dashboard():
    """National Climate Overview."""
    return render_template('national_dashboard.html', cities=sorted(INDIAN_CITIES))


@app.route('/aqi')
def aqi_dashboard():
    """Real-time Air Quality Analysis."""
    return render_template('aqi_dashboard.html', cities=sorted(INDIAN_CITIES))


@app.route('/map')
def map_dashboard():
    """Live Spatial Map."""
    return render_template('map_dashboard.html', cities=sorted(INDIAN_CITIES))

@app.route('/heat')
def heat_dashboard():
    """AI clustering and Heat Vulnerability Dashboard."""
    return render_template('heat_dashboard.html', cities=sorted(INDIAN_CITIES))



# ─── New API Endpoints for Auto-Refresh ───────────────────────

@app.route('/api/live/summary')
def api_live_summary():
    """
    Called every 10 seconds by the homepage to update leaderboards.
    Uses the underlying 5-minute cache in api_fetch to avoid rate limits.
    """
    # Force a quick start timestamp for performance tracking
    start_time = time.time()
    
    bulk_data = get_bulk_indian_cities_data()
    # ── Pipeline: API -> Pandas Processing -> Leaderboards ──
    processed_df = process_live_data(bulk_data)
    
    leaderboards = prepare_live_leaderboards(processed_df)
    stats = get_live_summary_stats(processed_df)
    
    response_data = {
        "status": "success",
        "timestamp": datetime.now().strftime('%I:%M:%S %p'),
        "latency_ms": round((time.time() - start_time) * 1000),
        "leaderboards": leaderboards,
        "stats": stats
    }
    
    return jsonify(response_data)


@app.route('/api/live/city')
def api_live_city():
    """
    Fetch comprehensive live data for a specific searched city.
    """
    city_query = request.args.get('q', 'New Delhi')
    
    data = get_live_city_data(city_query)
    
    if "error" in data:
        return jsonify({"status": "error", "message": data["error"]}), 400
        
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().strftime('%I:%M:%S %p'),
        "data": data
    })


@app.route('/api/live/map-markers')
def api_live_map_markers():
    """
    Returns real-time data specifically formatted for the Leaflet Map.
    """
    bulk_data = get_bulk_indian_cities_data()
    processed_df = process_live_data(bulk_data)
    
    markers = []
    if not processed_df.empty:
        for _, city in processed_df.iterrows():
            risk = city['Heat_Risk_Category']
            color = '#ff4757' if risk == 'High Risk' else '#ffa502' if risk == 'Medium Risk' else '#2ed573'
            
            markers.append({
                'city': city['City'],
                'lat': city['Latitude'],
                'lon': city['Longitude'],
                'temp': city['Temperature'],
                'aqi': city['AQI'],
                'risk': risk,
                'color': color,
                'condition': city['Condition']
            })
        
    return jsonify({"markers": markers})

@app.route('/api/live/clustering')
def api_live_clustering():
    """
    Runs K-Means clustering dynamically on the live bulk data and returns
    two Plotly figure JSONs.
    """
    bulk_data = get_bulk_indian_cities_data()
    processed_df = process_live_data(bulk_data)
    result = generate_live_clustering_charts(processed_df)
    
    if "error" in result:
        return jsonify({"status": "error", "message": result["error"]}), 400
        
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().strftime('%I:%M:%S %p'),
        "data": result
    })

@app.route('/api/viz/data')
def api_viz_data():
    """Returns the full processed dataset for the Dynamic Visualization Generator."""
    bulk_data = get_bulk_indian_cities_data()
    processed_df = process_live_data(bulk_data)
    return jsonify({
        "status": "success",
        "data": processed_df.to_dict('records')
    })



# ─── Run ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Local development server
    app.run(debug=True, port=5000)

# Export for Vercel serverless
app = app
