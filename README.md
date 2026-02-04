# Bike Demand Forecasting 🚴

Machine learning model for predicting London bike-sharing demand using time series analysis and seasonal decomposition.

**TLDR**: Analysed 31M TfL bike journeys → Built XGBoost demand forecaster (R² = 0.966) → Delivered actionable redistribution and pricing recommendations.

## Table of Contents
- [Bike Demand Forecasting 🚴](#bike-demand-forecasting-)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Findings](#key-findings)
  - [Technology Stack](#technology-stack)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Analytical Approach](#analytical-approach)
  - [Demand Forecasting Model](#demand-forecasting-model)
  - [Business Recommendations](#business-recommendations)
    - [1. Optimise Bike Redistribution](#1-optimise-bike-redistribution)
    - [2. Customer-Segmented Pricing](#2-customer-segmented-pricing)
  - [Future Enhancements](#future-enhancements)

## Overview

This project analyses Transport for London (TfL) Santander Cycles usage data from 2019-2021 (~31M journey records) to demonstrate how data science can be applied to real-world operational and strategic decisions in urban mobility.

The work combines exploratory data analysis, geospatial analytics, and machine learning to support demand forecasting, fleet optimisation, and pricing strategy for a hypothetical bike-sharing operator.

### Key Findings

| Metric | Value | Business Implication |
|--------|-------|---------------------|
| Unique Bikes | 17,766 | Infrastructure baseline |
| Annual Journeys | ~10M | Market size validation |
| Peak Hours | 8AM, 6PM | Redistribution timing |
| Short Trips (<30min) | 87% | Pricing tier opportunity |
| Day-to-Day Variation | 11% | Consistent demand |
| Post-COVID Growth | +5.5% | Expansion timing validated |

## Technology Stack

| Category | Tools |
|----------|-------|
| Core Analysis | Python, pandas, numpy, Jupyter |
| Machine Learning | scikit-learn, XGBoost |
| Visualisation | Plotly, Folium |
| Data Collection | Selenium, requests |
| Data Engineering | Parquet, categorical typing, ThreadPoolExecutor |

## Project Structure

```
├── scrape_data.py          # Data collection from TfL API
├── cycling_EDA.ipynb       # Exploratory data analysis & visualisation
├── forecast_model.ipynb    # Time series forecasting models
├── interactive plots/      # Plotly & Folium outputs
├── requirements.txt        # Python dependencies
├── client recommendations  # A simple stakeholder summary powerpoint
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python scrape_data.py          # Downloads ~400MB of data
```

Then open `cycling_EDA.ipynb` for analysis, `forecast_model.ipynb` for modeling.

**Requirements**: Python 3.10+, Chrome browser, ~500MB disk space

## Analytical Approach

The dataset contains over 30 million journeys, so the analysis was structured to progress from behavioural understanding to operational decision support:

1. **Behaviour Characterisation** — Separated predictable commuter demand from irregular leisure demand using temporal and spatial patterns

2. **Operational Alignment** — Aggregated data at hourly resolution to match real operational decisions (staffing, fleet positioning, maintenance)

3. **Forecasting Formulation** — Treated demand prediction as supervised regression with cyclical time encoding, lag features, and rolling averages

4. **Decision Integration** — Translated model outputs into actionable recommendations rather than standalone predictive metrics

## Demand Forecasting Model

**Objective**: Predict hourly system-wide demand

Data was aggregated from 31M journeys into ~26K hourly observations. Time-aware splitting (train: 2019–2020, test: 2021) prevented leakage.

**Features**: Cyclical time encoding • Lag features (1h, 24h, 168h) • Rolling means (24h, 7-day)

| Model | Test MAE | Test R² | Test RMSE |
|-------|----------|---------|-----------|
| **XGBoost** | **199.7** | **0.966** | **285.3** |
| Random Forest | 205.1 | 0.963 | 298.7 |
| Gradient Boosting | 208.4 | 0.961 | 306.2 |

**Validation**: No data leakage (`.shift()` for lags) • Time-based split • Feature importance analysis • Visual inspection of predictions vs actuals

## Business Recommendations

### 1. Optimise Bike Redistribution
Combine station imbalance data with demand forecasting:
- Pre-position bikes at source stations before 7AM rush
- Deploy collection crews at sink stations overnight (10PM-6AM)
- Prioritise top 10 imbalanced stations (80% of redistribution need)

### 2. Customer-Segmented Pricing
Two products for two distinct user types:
- **Commuter Pass**: Flat monthly fee, unlimited 30-min rides (87% qualify)
- **Leisure Premium**: Higher per-ride pricing for round-trips/park stations
- **Dynamic Surge**: Peak-hour pricing (6-8PM weekdays)

## Future Enhancements

| Category | Enhancement |
|----------|-------------|
| Model | Weather API integration, station-level forecasting, real-time REST API |
| Analytics | K-means customer segmentation, predictive maintenance, route optimisation |
| Production | MLOps pipeline, A/B testing framework, monitoring dashboard |

---

**Author**: Nitya Devaraj  
**Dataset**: [Transport for London Cycling Data](https://cycling.data.tfl.gov.uk/) (2019-2021)