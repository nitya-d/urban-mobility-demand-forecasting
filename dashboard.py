"""
Bike Demand Forecasting Dashboard

Interactive Streamlit app for exploring demand predictions.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Bike Demand Forecasting",
    page_icon="🚲",
    layout="wide"
)

# Constants
MODEL_PATH = Path("models/xgboost_demand.joblib")
AVG_HOURLY_DEMAND = 1150
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def create_features(hour: int, day_of_week: int, month: int, is_weekend: bool) -> pd.DataFrame:
    """Create feature DataFrame for prediction"""
    features = {
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': 15,
        'month': month,
        'year': datetime.now().year,
        'week_of_year': 26,
        'is_weekend': int(is_weekend),
        'is_rush_hour': int(hour in [7, 8, 9, 17, 18, 19]),
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
        'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'demand_lag_1h': AVG_HOURLY_DEMAND,
        'demand_lag_24h': AVG_HOURLY_DEMAND,
        'demand_lag_168h': AVG_HOURLY_DEMAND,
        'demand_rolling_24h': AVG_HOURLY_DEMAND,
        'demand_rolling_7d': AVG_HOURLY_DEMAND,
    }
    return pd.DataFrame([features])


def predict_demand(model, hour: int, day_of_week: int, month: int) -> int:
    """Get demand prediction for specific time"""
    is_weekend = day_of_week >= 5
    features = create_features(hour, day_of_week, month, is_weekend)
    prediction = model.predict(features)[0]
    return max(0, int(round(prediction)))


def predict_day(model, day_of_week: int, month: int) -> list:
    """Get predictions for all 24 hours of a day"""
    return [predict_demand(model, h, day_of_week, month) for h in range(24)]


# Load model
model = load_model()

# Header
st.title("🚲 London Bike Demand Forecasting")
st.markdown("*Predict hourly bike hire demand using ML trained on 31M TfL Santander Cycles journeys*")

if model is None:
    st.error("⚠️ Model not found. Please run `python train_model.py` first.")
    st.stop()

# Sidebar controls
st.sidebar.header("🎛️ Prediction Controls")

selected_month = st.sidebar.selectbox(
    "Month",
    options=list(range(1, 13)),
    format_func=lambda x: MONTHS[x-1],
    index=5  # June default
)

selected_day = st.sidebar.selectbox(
    "Day of Week",
    options=list(range(7)),
    format_func=lambda x: DAYS_OF_WEEK[x],
    index=0  # Monday default
)

selected_hour = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=8,
    format="%d:00"
)

# Main prediction
is_weekend = selected_day >= 5
prediction = predict_demand(model, selected_hour, selected_day, selected_month)

# Display main prediction
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Predicted Demand",
        value=f"{prediction:,} journeys",
        delta=f"{prediction - AVG_HOURLY_DEMAND:+,} vs average" if prediction != AVG_HOURLY_DEMAND else None
    )

with col2:
    demand_level = "🔴 Very High" if prediction > 2000 else "🟠 High" if prediction > 1500 else "🟡 Medium" if prediction > 800 else "🟢 Low"
    st.metric(label="Demand Level", value=demand_level)

with col3:
    st.metric(
        label="Time Selected",
        value=f"{DAYS_OF_WEEK[selected_day]} {selected_hour}:00",
        delta="Weekend" if is_weekend else "Weekday"
    )

st.divider()

# 24-hour forecast
st.subheader("📊 24-Hour Demand Forecast")

hourly_predictions = predict_day(model, selected_day, selected_month)
hours = list(range(24))

fig_hourly = go.Figure()

# Add bar chart
colors = ['#ff6b6b' if h in [7, 8, 9, 17, 18, 19] else '#4dabf7' for h in hours]
fig_hourly.add_trace(go.Bar(
    x=[f"{h}:00" for h in hours],
    y=hourly_predictions,
    marker_color=colors,
    hovertemplate="<b>%{x}</b><br>Predicted: %{y:,} journeys<extra></extra>"
))

# Highlight selected hour
fig_hourly.add_vline(
    x=selected_hour,
    line_dash="dash",
    line_color="green",
    annotation_text=f"Selected: {prediction:,}"
)

fig_hourly.update_layout(
    xaxis_title="Hour of Day",
    yaxis_title="Predicted Journeys",
    height=400,
    showlegend=False,
    hovermode="x unified"
)

# Add legend annotation
fig_hourly.add_annotation(
    x=0.02, y=0.98, xref="paper", yref="paper",
    text="🔴 Rush Hours | 🔵 Off-Peak",
    showarrow=False,
    bgcolor="white",
    borderpad=4
)

st.plotly_chart(fig_hourly, width='stretch')

# Weekday vs Weekend comparison
st.subheader("📈 Weekday vs Weekend Comparison")

col1, col2 = st.columns(2)

# Average weekday (Monday)
weekday_preds = predict_day(model, 0, selected_month)
# Average weekend (Saturday)
weekend_preds = predict_day(model, 5, selected_month)

fig_comparison = go.Figure()

fig_comparison.add_trace(go.Scatter(
    x=hours,
    y=weekday_preds,
    mode='lines+markers',
    name='Weekday (Mon)',
    line=dict(color='#228be6', width=3),
    hovertemplate="<b>%{x}:00</b><br>Weekday: %{y:,}<extra></extra>"
))

fig_comparison.add_trace(go.Scatter(
    x=hours,
    y=weekend_preds,
    mode='lines+markers',
    name='Weekend (Sat)',
    line=dict(color='#fa5252', width=3),
    hovertemplate="<b>%{x}:00</b><br>Weekend: %{y:,}<extra></extra>"
))

fig_comparison.update_layout(
    xaxis_title="Hour of Day",
    yaxis_title="Predicted Journeys",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)

st.plotly_chart(fig_comparison, width='stretch')

# Key insights
with st.expander("💡 Key Patterns from Training Data"):
    st.markdown("""
    | Pattern | Insight | Operational Implication |
    |---------|---------|------------------------|
    | **Morning Rush** | 7-9 AM weekdays see 2x average demand | Pre-position bikes by 6:30 AM |
    | **Evening Rush** | 5-7 PM weekdays peak at ~2,800/hour | Redistribute during lunch lull |
    | **Weekend Shift** | Peak moves to 12-3 PM | Different staffing schedule needed |
    | **Night Minimum** | 2-5 AM < 100 journeys/hour | Maintenance window |
    | **Seasonal Boost** | Summer months +40% vs winter | Plan fleet expansion May-Sep |
    """)

# Station Analysis Section
st.divider()
st.subheader("🗺️ Station Analysis")

# Tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Station Imbalance", "Station Popularity", "Round Trip Hotspots"])

with tab1:
    st.markdown("""
    **Station Imbalance** shows which stations accumulate bikes (sinks) vs lose bikes (sources).
    - 🔴 Red = More arrivals than departures (bikes accumulate)
    - 🔵 Blue = More departures than arrivals (bikes drain)
    
    *Operational insight: Target red stations for bike collection, pre-stock blue stations before rush hour.*
    """)
    imbalance_path = Path("plots/interactive plots/station_imbalance_map.html")
    if imbalance_path.exists():
        with open(imbalance_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=500, scrolling=True)
    else:
        st.warning("Station imbalance map not found. Run the EDA notebook to generate it.")

with tab2:
    st.markdown("""
    **Station Popularity** shows the most heavily used stations by total journey count.
    
    *Use this to identify high-demand locations for capacity planning and infrastructure investment.*
    """)
    popularity_path = Path("plots/interactive plots/station_popularity_map.html")
    if popularity_path.exists():
        with open(popularity_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=500, scrolling=True)
    else:
        st.warning("Station popularity map not found. Run the EDA notebook to generate it.")

with tab3:
    st.markdown("""
    **Round Trip Hotspots** shows stations where users frequently return bikes to the same location (leisure/tourist activity).
    
    *These stations may benefit from premium leisure pricing or tourist-focused marketing.*
    """)
    roundtrip_path = Path("plots/interactive plots/round_trip_hotspots_map.html")
    if roundtrip_path.exists():
        with open(roundtrip_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=500, scrolling=True)
    else:
        st.warning("Round trip hotspots map not found. Run the EDA notebook to generate it.")

# Experimentation ideas (shows product thinking for Monzo)
st.divider()
st.subheader("🧪 A/B Testing Opportunities")

st.markdown("""
This forecasting model enables several **data-driven experiments**:

| Experiment | Hypothesis | Metrics |
|------------|------------|---------|
| **Dynamic Pricing** | Surge pricing during peak hours reduces demand spikes by 15% | Peak-to-average ratio, revenue per journey |
| **Redistribution Alerts** | Proactive bike repositioning reduces "empty station" events by 30% | Station availability rate, user complaints |
| **Commuter Subscriptions** | Flat-rate monthly pass increases weekday retention by 20% | Monthly active users, journey frequency |
| **Weekend Promotions** | 10% discount on Sat/Sun increases weekend utilisation by 25% | Weekend journey count, new user acquisition |

**Implementation**: Split stations into control/treatment groups, track metrics for 4-6 weeks, analyse with statistical significance testing.
""")

# Model info
st.sidebar.divider()
st.sidebar.subheader("📊 Model Info")
st.sidebar.markdown(f"""
- **Algorithm**: XGBoost Regressor
- **Training Data**: 31M journeys (2019-2021)
- **Test R²**: 0.966
- **Test MAE**: 199.7 journeys
""")

st.sidebar.divider()
st.sidebar.markdown("*Built for TfL Santander Cycles analysis*")
