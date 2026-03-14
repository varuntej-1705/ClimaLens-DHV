/**
 * Climora Charts & Map logic
 * Updated for Light Weather-App Theme
 */

// Global chart settings for light theme
const lightChartLayout = {
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    font: { family: "Inter, sans-serif", color: "#6B7280" },
    xaxis: { gridcolor: "#E5E7EB", zeroline: false },
    yaxis: { gridcolor: "#E5E7EB", zeroline: false }
};

/**
 * Initialize the Leaflet Map with Light Theme tiles
 */
function initMap(containerId, data) {
    const map = L.map(containerId, {
        zoomControl: false,
        attributionControl: false
    }).setView([20.5937, 78.9629], 5);

    // CartoDB Positron (Light)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19
    }).addTo(map);

    L.control.zoom({ position: 'bottomleft' }).addTo(map);

    if (data) {
        addMapMarkers(map, data);
    }
    return map;
}

/**
 * Adds circle markers with light theme popups
 */
function addMapMarkers(map, data) {
    data.forEach(city => {
        const color = city.Heat_Risk_Score > 70 ? '#EF4444' : (city.Heat_Risk_Score > 40 ? '#F59E0B' : '#22C55E');
        const marker = L.circleMarker([city.Latitude, city.Longitude], {
            radius: 12,
            fillColor: color,
            color: 'white',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.85
        }).addTo(map);

        const popupContent = `
            <div class="custom-popup">
                <div class="popup-title">${city.City}</div>
                <div class="popup-stat"><span>Temperature</span> <strong>${city.Temperature}°C</strong></div>
                <div class="popup-stat"><span>AQI Index</span> <strong>${city.AQI}</strong></div>
                <div class="popup-badge" style="background:${color}">${city.ML_Risk_Level} Risk</div>
            </div>
        `;

        marker.bindPopup(popupContent, { className: 'custom-popup-light' });
    });
}

/**
 * Common Plotly responsive wrapper
 */
function renderPlotly(containerId, data, layout, options = {}) {
    const finalLayout = { ...lightChartLayout, ...layout };
    Plotly.newPlot(containerId, data, finalLayout, { 
        responsive: true, 
        displayModeBar: false,
        ...options 
    });
}
