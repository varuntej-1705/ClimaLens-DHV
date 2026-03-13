import os
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
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
    """Real-time Climate Analysis."""
    return render_template('climate_dashboard.html', cities=sorted(INDIAN_CITIES))


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

@app.route('/insights')
def insights_dashboard():
    """Adaptation Strategies and Recommendations."""
    return render_template('insights.html', cities=sorted(INDIAN_CITIES))

@app.route('/settings')
def settings_page():
    """System Settings Page."""
    return render_template('settings.html', cities=sorted(INDIAN_CITIES))


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

@app.route('/api/live/insights')
def api_live_insights():
    """
    Returns data quality/outlier analysis and general summary statistics
    for the insights & adaptation strategies page.
    """
    bulk_data = get_bulk_indian_cities_data()
    processed_df = process_live_data(bulk_data)
    
    if processed_df.empty:
        return jsonify({"status": "error", "message": "No data unavailable"}), 400
        
    stats = get_live_summary_stats(processed_df)
    
    # Calculate Z-score outlier summaries for the UI table
    outliers = {}
    for metric in ['Temperature', 'AQI']:
        if f'Is_Outlier' in processed_df.columns:
            # We flagged it globally, but for summary we can recalculate quickly 
            # to show metric-specific counts if we wanted. 
            # For simplicity, we'll just check if the metric itself is > 3 std devs
            mean = processed_df[metric].mean()
            std = processed_df[metric].std()
            if std > 0:
                metric_outliers = processed_df[abs(processed_df[metric] - mean) > (3 * std)]
                count = len(metric_outliers)
            else:
                count = 0
                
            percent = round((count / len(processed_df)) * 100, 1)
            outliers[metric] = {"count": count, "percentage": percent}
            
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().strftime('%I:%M:%S %p'),
        "stats": stats,
        "outliers": outliers
    })


# ─── Run ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Local development server
    app.run(debug=True, port=5000)

# Export for Vercel serverless
app = app
