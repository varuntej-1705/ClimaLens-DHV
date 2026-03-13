# Climora — Climate Risk Monitoring & Urban Heat Vulnerability

Climora is a professional climate risk monitoring system that analyzes real-time climate data (Temperature, Humidity) and Air Quality Index (AQI) data (PM2.5, PM10, NO2, Ozone) to detect urban heat vulnerability zones across major Indian cities using Machine Learning.

## Features
- **Real-time Data Collection**: Integrated with OpenWeatherMap and AQICN APIs for live environmental monitoring.
- **Data Preprocessing**: Full pipeline including cleaning, normalization, Z-score outlier detection, and feature engineering.
- **Heatwave Analysis**: Trend detection and frequency monitoring of extreme heat events.
- **ML Clustering**: PCA + K-Means clustering to classify cities into High, Medium, and Low risk vulnerability zones.
- **Geospatial Mapping**: Interactive Leaflet.js map with color-coded risk markers.
- **Professional Dashboards**: 6 specialized views for Climate, AQI, Heat Analysis, and Adaptation Strategies.

## Tech Stack
- **Backend**: Python Flask
- **Data Analysis**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Leaflet.js
- **Frontend**: HTML5, CSS3 (Modern Dark Theme), JavaScript

## Project Structure
- `app.py`: Main Flask application server.
- `utils/`: API fetch, data handling, and trend analysis modules.
- `models/`: ML clustering models (PCA + KMeans).
- `templates/`: HTML5 dashboard templates.
- `static/`: CSS3 styling and interactive JS.
- `data/`: Raw and processed CSV datasets.

## How to Run Locally

### 1. Requirements
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Keys
The system comes with a valid API key. You can update it in the `.env` file:
```env
CLIMATE_API_KEY=56be9ff42b564dbfa7074815250112
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

## Modules Overview
- **Climate Dashboard**: Line charts and box plots for temperature/humidity.
- **AQI Dashboard**: Distribution and trend analysis of gaseous pollutants.
- **Heat Analysis**: PCA-based clustering visualization and risk scorecards.
- **Heat Map**: Geospatial visualization of vulnerability zones.
- **Adaptation Strategies**: Scientific recommendations for urban heat mitigation.
