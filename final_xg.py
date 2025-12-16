# predict_2025_risk_maps.py
import pandas as pd
import numpy as np
import random
import joblib
import folium
import os
import glob
from datetime import datetime, timedelta

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# -----------------------------
# Define zones and coordinates
# -----------------------------
zones = {
    "Coimbatore": {
        "Mettupalayam": (11.29968, 76.94244), "Avinashi": (11.20233, 77.29211),
        "Thiruppur": (11.10319, 77.39866), "Palladam": (11.005831, 77.287908),
        "Pollachi": (10.66167, 77.00667), "Udumalaipet": (10.58100, 77.26408),
        "Valparai": (10.32814, 76.95580), "Coimbatore North": (11.060874, 76.947387),
        "Coimbatore South": (10.982548, 76.989055)
    },
    "Kanyakumari": {
        "Agastheeswaram": (8.101711, 77.538529), "Kalkulam": (8.184250, 77.315512),
        "Killiyoor": (8.265311, 77.217375), "Thiruvattar": (8.335010, 77.265150),
        "Thovalai": (8.231171, 77.505970), "Vilavancode": (8.313338, 77.206699)
    },
    "Tiruchirapalli": {
        "Srirangam": (10.854603, 78.713548), "Thillai Nagar": (10.834443, 78.660012),
        "Tiruchirappalli Fort": (10.825416, 78.687250), "BHEL Township": (10.779601, 78.802013),
        "Puthur": (10.816613, 78.675481), "K.K. Nagar": (10.754823, 78.697035),
        "Golden Rock": (10.786705, 78.727490), "Bharathidasan Nagar": (10.815605, 78.692403),
        "Abhishekapuram": (10.814276, 78.680441), "Cantonment": (10.801732, 78.693002),
        "Ariyamangalam": (10.810064, 78.720986), "Thuvakudi": (10.752956, 78.830112),
        "Edamalaipatti Pudur": (10.776544, 78.673610), "Senthaneerpuram": (10.819999, 78.684148),
        "Manikandam": (10.738831, 78.637647)
    },
    "Salem": {
        "Mettur": (11.783860, 77.796605), "Omalur": (11.743163, 78.046521),
        "Yercadu": (11.775337, 78.211078), "Salem": (11.667494, 78.150663),
        "Edappadi": (11.585847, 77.839697), "Sankari": (11.474599, 77.869162),
        "Vazhappadi": (11.657002, 78.401244), "Attur": (11.599410, 78.599066),
        "Gangavalli": (11.497384, 78.652020)
    },
    "Chennai": {
        "Thiruvotriyur": (13.164773, 80.300387), "Manali": (13.177337, 80.268712),
        "Madhavaram": (13.148179, 80.230612), "Tondiarpet": (13.126294, 80.288429),
        "Royapuram": (13.114361, 80.295138), "Sholinganallur": (12.894777, 80.218789),
        "Ambattur": (13.118490, 80.156760), "Anna Nagar": (13.088520, 80.206771),
        "Teynampet": (13.045034, 80.250918), "Kodambakkam": (13.051635, 80.221131),
        "Valasaravakkam": (13.041316, 80.175038), "Alandur": (12.999424, 80.199751),
        "Adyar": (13.001462, 80.254135), "Perungudi": (12.964866, 80.240739),
        "Kotturpuram": (13.015840, 80.239412)
    }
}

# -----------------------------
# Generate weekly datasets for a half-year (H1: Jan–Jun, H2: Jul–Dec)
# -----------------------------
def generate_half_year_weeks(year=2025, half: str = "H2"):
    """Generate exactly 26 weekly records per zone for the given half-year.
    half: 'H1' for Jan–Jun, 'H2' for Jul–Dec
    """
    start_month = 1 if half.upper() == "H1" else 7
    rows = []
    date = datetime(year, start_month, 1)

    for week_idx in range(26):  # 26 weeks per half
        month = date.month
        week_in_half = week_idx + 1

        for city, city_zones in zones.items():
            for zone, (lat, lon) in city_zones.items():
                temp = round(random.uniform(22, 38), 1)
                humidity = round(random.uniform(45, 95), 1)
                # Seasonality: more rain in H2 to mimic monsoon
                rain_scale = 1.2 if half.upper() == "H2" else 0.8
                rainfall = round(random.uniform(0, 350) * rain_scale, 1)
                pop_density = random.randint(1000, 15000)
                prev_incidents = random.randint(0, 50)

                rows.append([
                    city, zone, lat, lon, date.year, month, week_in_half,
                    temp, humidity, rainfall, pop_density, prev_incidents
                ])
        date += timedelta(weeks=1)

    df = pd.DataFrame(rows, columns=[
        "city","zone","latitude","longitude","year","month","week",
        "temperature","humidity","rainfall","pop_density","previous_incidents"
    ])
    return df

def generate_future_weeks(year=2025):
    """Backward-compatible alias: generate H2 (Jul–Dec)."""
    return generate_half_year_weeks(year=year, half="H2")

# -----------------------------
# Predict risk
# -----------------------------
def find_latest_model(models_dir="models"):
    """Return the path to the latest saved XGBoost model in models_dir."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    candidates = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.startswith("xgboost_model_") and f.endswith(".pkl")
    ]
    if not candidates:
        raise FileNotFoundError("No saved models found in 'models/'")
    # Pick most recently modified
    return max(candidates, key=os.path.getmtime)

def predict_risk(df, model_path=None):
    if model_path is None:
        model_path = find_latest_model()
    model = joblib.load(model_path)
    # Prepare features to match the trained model
    df = df.copy()
    # Normalize schema differences
    if "population_density" not in df.columns and "pop_density" in df.columns:
        df["population_density"] = df["pop_density"]
    if "sanitation_score" not in df.columns:
        # If not provided for future weeks, synthesize a realistic sanitation score
        # Higher population density and higher rainfall generally reduce sanitation effectiveness
        pop = df["population_density"].astype(float)
        hum = df.get("humidity", pd.Series(70.0, index=df.index)).astype(float)
        rain = df.get("rainfall", pd.Series(100.0, index=df.index)).astype(float)
        base = 80.0 - 0.0008 * pop + 0.02 * hum - 0.01 * rain
        noise = np.random.normal(0, 3, size=len(df))
        df["sanitation_score"] = np.clip(base + noise, 45.0, 95.0)

    # Ensure month exists for cyclic features; generate if missing from a date
    # (our generator already provides 'month')

    # Engineer features similar to train_xgboost.prepare_data
    def engineer_features(_df: pd.DataFrame) -> pd.DataFrame:
        _df = _df.copy()
        # Cyclic encoding
        _df['month_sin'] = np.sin(2 * np.pi * _df['month']/12)
        _df['month_cos'] = np.cos(2 * np.pi * _df['month']/12)
        # Interaction terms
        _df['temp_humidity'] = _df['temperature'] * _df['humidity']
        _df['rainfall_pop_density'] = _df['rainfall'] * (_df['population_density'] / 1000.0)
        # Rolling stats by zone over time (best-effort ordering by year, month, week)
        sort_cols = [c for c in ['city','zone','year','month','week'] if c in _df.columns]
        if sort_cols:
            _df = _df.sort_values(sort_cols)
        for window in [4, 8, 12]:
            _df[f'rolling_avg_temp_{window}w'] = _df.groupby('zone')['temperature'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        return _df

    df = engineer_features(df)

    # Exact feature list used by the trained model
    features = [
        'temperature', 'humidity', 'rainfall',
        'sanitation_score', 'population_density',
        'month_sin', 'month_cos',
        'temp_humidity', 'rainfall_pop_density',
        'rolling_avg_temp_4w', 'rolling_avg_temp_8w', 'rolling_avg_temp_12w'
    ]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features for prediction: {missing}")

    raw_scores = model.predict(df[features])

    # Calibrate to a more realistic 0–100 distribution
    def calibrate_scores(scores: np.ndarray,
                         target_mean: float = 50.0,
                         target_std: float = 18.0) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        # Robust scale using IQR to avoid extreme influence
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        iqr = max(q75 - q25, 1e-6)
        approx_std = iqr / 1.349  # IQR to std approximation
        if approx_std < 1e-6:
            approx_std = np.std(s) if np.std(s) > 1e-6 else 1.0
        z = (s - q50) / approx_std
        calibrated = z * target_std + target_mean
        # Percentile-based widening to ensure a mix of low/medium/high
        p20, p80 = np.percentile(calibrated, [20, 80])
        # Piecewise linear mapping: below p20 compress towards 20, above p80 towards 80
        low_mask = calibrated <= p20
        high_mask = calibrated >= p80
        mid_mask = ~(low_mask | high_mask)
        # Map lows from [min, p20] -> [5, 30]
        cmin = calibrated.min()
        if p20 - cmin < 1e-6:
            low_scaled = np.full_like(calibrated, 20.0)
        else:
            low_scaled = 5.0 + (calibrated - cmin) / (p20 - cmin) * (30.0 - 5.0)
        # Map mids from [p20, p80] -> [30, 70]
        if p80 - p20 < 1e-6:
            mid_scaled = np.full_like(calibrated, 50.0)
        else:
            mid_scaled = 30.0 + (calibrated - p20) / (p80 - p20) * (70.0 - 30.0)
        # Map highs from [p80, max] -> [70, 95]
        cmax = calibrated.max()
        if cmax - p80 < 1e-6:
            high_scaled = np.full_like(calibrated, 85.0)
        else:
            high_scaled = 70.0 + (calibrated - p80) / (cmax - p80) * (95.0 - 70.0)
        calibrated = np.where(low_mask, low_scaled,
                      np.where(high_mask, high_scaled, mid_scaled))
        # Simple heuristics to dampen unrealistically high risk
        # - Low rain and low humidity generally reduce risk
        low_env_mask = (df['rainfall'] < 50) & (df['humidity'] < 60)
        calibrated = np.where(low_env_mask, calibrated - 8.0, calibrated)
        # - Very good sanitation reduces risk
        calibrated = np.where(df['sanitation_score'] >= 85, calibrated - 5.0, calibrated)
        # - Extremely low temperature (< 22C) or very high (> 38C) reduce risk (vector activity bounds)
        temp_mask = (df['temperature'] < 22) | (df['temperature'] > 38)
        calibrated = np.where(temp_mask, calibrated - 6.0, calibrated)
        return np.clip(calibrated, 0.0, 100.0)

    df["risk_score"] = calibrate_scores(raw_scores)
    return df

# -----------------------------
# Public helpers to return datasets
# -----------------------------
def get_future_dataset(year=2025):
    """Return the generated future (Jul–Dec) weekly dataset for the given year."""
    return generate_future_weeks(year=year)

def get_predictions(year=2025, model_path=None):
    """Return a tuple: (future_df, predictions_df) without saving files.
    If model_path is None, the latest model in the 'models/' directory is used.
    """
    future_df = generate_future_weeks(year=year)
    pred_df = predict_risk(future_df.copy(), model_path=model_path)
    return future_df, pred_df

# -----------------------------
# Generate Folium map
# -----------------------------
def generate_map(df, city_name, save_path):
    city_df = df[df["city"] == city_name]
    if city_df.empty:
        raise ValueError(f"No rows for city: {city_name}")
    m = folium.Map(
        location=[city_df["latitude"].mean(), city_df["longitude"].mean()],
        zoom_start=10
    )
    def get_color(score):
        if score < 30:
            return "green"
        elif score < 70:
            return "orange"
        else:
            return "red"
    for _, row in city_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=get_color(row["risk_score"]),
            fill=True,
            fill_color=get_color(row["risk_score"]),
            fill_opacity=0.6,
            popup=f"{row['zone']}<br>Risk: {row['risk_score']:.1f}<br>Week {row['week']}, {row['month']}/{row['year']}"
        ).add_to(m)
    m.save(save_path)
    print(f"✅ Map saved as {save_path}")

# -----------------------------
# Run all
# -----------------------------
if __name__ == "__main__":
    # Generate weekly data for Jan–Jun 2025 (input) and Jul–Dec 2025 (pred output)
    df_input = generate_half_year_weeks(year=2025, half="H1")
    df_future = generate_half_year_weeks(year=2025, half="H2")

    # Predict risk
    df_pred = predict_risk(df_future)

    # Save input (H1) and predictions (H2) CSVs
    df_input.to_csv("dataset/risk_2025_H1_weekly_input.csv", index=False)
    print("✅ H1 input (Jan–Jun) saved as 'dataset/risk_2025_H1_weekly_input.csv'")
    df_pred.to_csv("dataset/risk_2025_H2_weekly_predictions.csv", index=False)
    print("✅ H2 predictions (Jul–Dec) saved as 'dataset/risk_2025_H2_weekly_predictions.csv'")
    # Print distribution summary to ensure mix of low/medium/high
    def category(score):
        if score < 30: return 'low'
        if score < 70: return 'medium'
        return 'high'
    dist = df_pred.groupby('city')['risk_score'].apply(lambda s: pd.Series({
        'low': (s < 30).mean()*100,
        'medium': ((s >= 30) & (s < 70)).mean()*100,
        'high': (s >= 70).mean()*100
    })).unstack(fill_value=0).round(1)
    print("\nRisk distribution by city (%):")
    for city_name, row in dist.iterrows():
        print(f"- {city_name}: low {row['low']}%, medium {row['medium']}%, high {row['high']}%")

    for city_name in zones.keys():
        map_path = f"dataset/{city_name.lower().replace(' ', '_')}_risk_2025_weekly_map.html"
        generate_map(df_pred, city_name=city_name, save_path=map_path)