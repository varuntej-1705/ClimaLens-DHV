"""
ClimaLens — Heatwave Trend Analysis Module
Analyzes temperature trends, seasonal patterns, and heatwave events.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json


def get_temperature_trends(df, city=None):
    """
    Generate temperature trend data for line chart.
    Returns Plotly JSON for rendering.
    """
    if city and city != 'All':
        data = df[df['City'] == city].copy()
    else:
        data = df.copy()

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    if city and city != 'All':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['Date'].dt.strftime('%b %Y'),
            y=data['Temperature'],
            mode='lines+markers',
            name='Temperature',
            line={'color': '#ff6b6b', 'width': 3},
            marker={'size': 8},
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)',
        ))
    else:
        fig = go.Figure()
        for c in data['City'].unique()[:5]:  # Top 5 cities
            city_data = data[data['City'] == c]
            fig.add_trace(go.Scatter(
                x=city_data['Date'].dt.strftime('%b %Y'),
                y=city_data['Temperature'],
                mode='lines+markers',
                name=c,
                line={'width': 2},
                marker={'size': 6},
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={'text': 'Temperature Trend Analysis', 'font': {'size': 18, 'color': '#e6edf3'}},
        xaxis={'title': 'Month', 'gridcolor': 'rgba(48,54,61,0.5)', 'color': '#8b949e'},
        yaxis={'title': 'Temperature (°C)', 'gridcolor': 'rgba(48,54,61,0.5)', 'color': '#8b949e'},
        legend={'font': {'color': '#8b949e'}},
        margin={'l': 40, 'r': 20, 't': 50, 'b': 40},
        height=380,
    )

    return json.loads(fig.to_json())


def get_humidity_trends(df, city=None):
    """Generate humidity trend chart data."""
    if city and city != 'All':
        data = df[df['City'] == city].copy()
    else:
        data = df.groupby('Date')['Humidity'].mean().reset_index()

    data['Date'] = pd.to_datetime(data['Date']) if 'Date' in data.columns else data.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'].dt.strftime('%b %Y') if hasattr(data['Date'], 'dt') else data['Date'],
        y=data['Humidity'],
        mode='lines+markers',
        name='Humidity',
        line={'color': '#00c2ff', 'width': 3},
        marker={'size': 8},
        fill='tozeroy',
        fillcolor='rgba(0, 194, 255, 0.1)',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={'text': 'Humidity Trend', 'font': {'size': 18, 'color': '#e6edf3'}},
        xaxis={'title': 'Month', 'gridcolor': 'rgba(48,54,61,0.5)', 'color': '#8b949e'},
        yaxis={'title': 'Humidity (%)', 'gridcolor': 'rgba(48,54,61,0.5)', 'color': '#8b949e'},
        margin={'l': 40, 'r': 20, 't': 50, 'b': 40},
        height=380,
    )

    return json.loads(fig.to_json())


def detect_heatwaves(df, threshold=38):
    """
    Detect heatwave events (days above threshold temperature).
    Returns heatwave frequency chart data.
    """
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.strftime('%b')

    # Count days above threshold per city
    heatwave_days = data[data['Temperature'] >= threshold].groupby('City').size().reset_index(name='Heatwave_Days')
    heatwave_days = heatwave_days.sort_values('Heatwave_Days', ascending=True)

    colors = ['#ff6b6b' if x >= 4 else '#ffa726' if x >= 2 else '#66bb6a'
              for x in heatwave_days['Heatwave_Days']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=heatwave_days['Heatwave_Days'],
        y=heatwave_days['City'],
        orientation='h',
        marker={'color': colors, 'line': {'color': 'rgba(255,255,255,0.1)', 'width': 1}},
        text=heatwave_days['Heatwave_Days'],
        textposition='outside',
        textfont={'color': '#e6edf3'},
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={'text': 'Heatwave Frequency by City (Days ≥ 38°C)', 'font': {'size': 18, 'color': '#e6edf3'}},
        xaxis={'title': 'Number of Extreme Heat Days', 'gridcolor': 'rgba(48,54,61,0.5)', 'color': '#8b949e'},
        yaxis={'color': '#8b949e'},
        margin={'l': 120, 'r': 40, 't': 50, 'b': 40},
        height=450,
    )

    return json.loads(fig.to_json())


def get_risk_distribution(df):
    """Pie chart of heat risk category distribution."""
    if 'Heat_Risk_Category' not in df.columns:
        return None

    risk_counts = df['Heat_Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Category', 'Count']

    color_map = {'High Risk': '#ff6b6b', 'Medium Risk': '#ffa726', 'Low Risk': '#66bb6a'}
    colors = [color_map.get(cat, '#888') for cat in risk_counts['Category']]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=risk_counts['Category'],
        values=risk_counts['Count'],
        marker={'colors': colors, 'line': {'color': '#161b22', 'width': 2}},
        textinfo='label+percent',
        textfont={'size': 14, 'color': '#e6edf3'},
        hole=0.45,
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={'text': 'Heat Risk Distribution', 'font': {'size': 18, 'color': '#e6edf3'}},
        legend={'font': {'color': '#8b949e'}},
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
        height=380,
    )

    return json.loads(fig.to_json())


def get_monthly_temperature_distribution(df):
    """Box plot of temperature by month across all cities."""
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data['Month'] = pd.Categorical(data['Month'], categories=month_order, ordered=True)

    fig = go.Figure()
    for month in month_order:
        month_data = data[data['Month'] == month]
        if not month_data.empty:
            fig.add_trace(go.Box(
                y=month_data['Temperature'],
                name=month,
                marker_color='#00c2ff',
                line={'color': '#00c2ff'},
            ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={'text': 'Monthly Temperature Distribution', 'font': {'size': 18, 'color': '#e6edf3'}},
        yaxis={'title': 'Temperature (°C)', 'gridcolor': 'rgba(48,54,61,0.5)', 'color': '#8b949e'},
        xaxis={'color': '#8b949e'},
        margin={'l': 40, 'r': 20, 't': 50, 'b': 40},
        height=380,
        showlegend=False,
    )

    return json.loads(fig.to_json())
