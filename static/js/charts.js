/**
 * ClimaLens — Chart Rendering & Interactivity
 * Plotly chart helpers, Leaflet map init, and UI controls.
 */

// ─── Render Plotly Chart ──────────────────────────────────
function renderChart(containerId, chartData) {
    const container = document.getElementById(containerId);
    if (!container || !chartData) return;

    try {
        const data = chartData.data || [];
        const layout = chartData.layout || {};

        // Ensure responsive
        layout.autosize = true;
        layout.font = { family: 'Inter, sans-serif' };

        Plotly.newPlot(containerId, data, layout, {
            responsive: true,
            displayModeBar: false,
        });
    } catch (e) {
        console.error('Chart render error:', e);
        container.innerHTML = '<p style="color:#8b949e;text-align:center;padding:40px;">Chart unavailable</p>';
    }
}

// ─── Initialize Leaflet Map ───────────────────────────────
function initHeatMap(markers) {
    const mapContainer = document.getElementById('heat-map');
    if (!mapContainer) return;

    const map = L.map('heat-map', {
        zoomControl: true,
        scrollWheelZoom: true,
    }).setView([20.5937, 78.9629], 5); // Center on India

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 18,
    }).addTo(map);

    // Add markers
    if (markers && markers.length > 0) {
        markers.forEach(function(m) {
            const markerColor = m.color || '#ffa726';
            const riskClass = m.risk === 'High Risk' ? 'risk-high' :
                              m.risk === 'Medium Risk' ? 'risk-medium' : 'risk-low';

            // Custom circle marker
            const circle = L.circleMarker([m.lat, m.lon], {
                radius: 12,
                fillColor: markerColor,
                color: '#ffffff',
                weight: 2,
                opacity: 0.9,
                fillOpacity: 0.7,
            }).addTo(map);

            // Popup content
            const popupHTML = `
                <div class="popup-content">
                    <h3>${m.city}</h3>
                    <div class="popup-row">
                        <span class="label">🌡 Temperature</span>
                        <span class="val" style="color:#ffa726">${m.temp}°C</span>
                    </div>
                    <div class="popup-row">
                        <span class="label">🌫 AQI</span>
                        <span class="val" style="color:#bb86fc">${m.aqi}</span>
                    </div>
                    <div class="popup-row">
                        <span class="label">🔥 Heat Score</span>
                        <span class="val" style="color:${markerColor}">${m.heat_score}</span>
                    </div>
                    <div class="popup-row">
                        <span class="label">⚠ Risk Level</span>
                        <span class="val" style="color:${markerColor}">${m.risk}</span>
                    </div>
                </div>
            `;

            circle.bindPopup(popupHTML, {
                className: 'custom-popup',
                maxWidth: 250,
            });

            // City label
            L.marker([m.lat, m.lon], {
                icon: L.divIcon({
                    className: 'city-label',
                    html: `<span style="color:${markerColor};font-size:11px;font-weight:600;
                            text-shadow:0 0 4px rgba(0,0,0,0.8);white-space:nowrap;">${m.city}</span>`,
                    iconSize: [100, 20],
                    iconAnchor: [50, -10],
                })
            }).addTo(map);
        });
    }

    // Legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function() {
        const div = L.DomUtil.create('div', 'map-legend');
        div.style.cssText = `
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 16px;
            color: #e6edf3;
            font-family: 'Inter', sans-serif;
            font-size: 12px;
        `;
        div.innerHTML = `
            <div style="font-weight:600;margin-bottom:8px;">Risk Level</div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <span style="width:12px;height:12px;border-radius:50%;background:#ff6b6b;display:inline-block;"></span> High Risk
            </div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <span style="width:12px;height:12px;border-radius:50%;background:#ffa726;display:inline-block;"></span> Medium Risk
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:12px;height:12px;border-radius:50%;background:#66bb6a;display:inline-block;"></span> Low Risk
            </div>
        `;
        return div;
    };
    legend.addTo(map);

    return map;
}

// ─── City Selector ────────────────────────────────────────
function onCityChange(selectElement) {
    const city = selectElement.value;
    const currentPath = window.location.pathname;
    const url = new URL(window.location);
    url.searchParams.set('city', city);
    window.location.href = url.toString();
}

// ─── Refresh Data ─────────────────────────────────────────
function refreshData() {
    const btn = document.getElementById('refresh-btn');
    if (btn) {
        btn.innerHTML = '<span class="spinner"></span> Refreshing...';
        btn.disabled = true;
    }

    fetch('/api/refresh-data', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            showToast(data.message, data.status === 'success' ? 'success' : 'warning');
            if (data.status === 'success') {
                setTimeout(() => location.reload(), 1500);
            }
        })
        .catch(err => {
            showToast('Failed to refresh data', 'error');
        })
        .finally(() => {
            if (btn) {
                btn.innerHTML = '🔄 Refresh Data';
                btn.disabled = false;
            }
        });
}

// ─── Download CSV ─────────────────────────────────────────
function downloadCSV() {
    window.location.href = '/api/download-csv';
    showToast('Downloading processed dataset...', 'success');
}

// ─── Toast Notification ───────────────────────────────────
function showToast(message, type = 'success') {
    // Remove existing toasts
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 50);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ─── Auto-Refresh Timer ───────────────────────────────────
let autoRefreshInterval = null;

function startAutoRefresh(intervalMinutes = 10) {
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    autoRefreshInterval = setInterval(() => {
        refreshData();
    }, intervalMinutes * 60 * 1000);
}

// ─── Sidebar Mobile Toggle ────────────────────────────────
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) sidebar.classList.toggle('open');
}

// ─── Initialize on Load ──────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
    // Animate stat cards
    const cards = document.querySelectorAll('.stat-card, .dash-card, .strategy-card');
    cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100 + i * 80);
    });

    // Start auto-refresh
    startAutoRefresh(10);
});
