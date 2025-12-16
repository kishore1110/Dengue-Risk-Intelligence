import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# -----------------------------
# Define zones and coordinates
# -----------------------------
# -----------------------------
# Define zones and coordinates
# -----------------------------
zones = {
    "Coimbatore": {
        "Mettupalayam": (11.29968, 76.94244),
        "Avinashi": (11.20233, 77.29211),
        "Thiruppur": (11.10319, 77.39866),
        "Palladam": (11.005831, 77.287908),
        "Pollachi": (10.66167, 77.00667),
        "Udumalaipet": (10.58100, 77.26408),
        "Valparai": (10.32814, 76.95580),
        "Coimbatore North": (11.060874454593293, 76.94738732503518),
        "Coimbatore South": (10.982548569803779, 76.98905572666018),
    },
    "Chennai": {
        "Thiruvotriyur": (13.164773844031352, 80.30038727033461),
        "Manali": (13.177337013345936, 80.26871272264003),
        "Madhavaram": (13.148179758690757, 80.23061223333873),
        "Tondiarpet": (13.1262949400739, 80.28842991846733),
        "Royapuram": (13.114361770441773, 80.29513825798645),
        "Sholinganallur": (12.894777917601955, 80.21878910415437),
        "Ambattur": (13.118490265427598, 80.1567603795202),
        "Anna Nagar": (13.088520556973387, 80.206771086125),
        "Teynampet": (13.045034718459263, 80.25091867594695),
        "Kodambakkam": (13.051635411882724, 80.22113168248616),
        "Valasaravakkam": (13.041316946926463, 80.1750380044828),
        "Alandur": (12.999424518344618, 80.19975197828738),
        "Adyar": (13.001462136345742, 80.25413517121761),
        "Perungudi": (12.96486693365694, 80.24073981976898),
        "Kotturpuram": (13.0158405619389, 80.23941254952075)
    },
    "Kanyakumari": {
        "Agastheeswaram": (8.101711001418199, 77.53852942379115),
        "Kalkulam": (8.184250604647879, 77.31551227451052),
        "Killiyoor": (8.265311626195535, 77.21737532270488),
        "Thiruvattar": (8.335010342231953, 77.26515061898453),
        "Thovalai": (8.231171776027947, 77.50597006674505),
        "Vilavancode": (8.313338680527695, 77.20669942995275)
    },
    "Tiruchirapalli": {
        "Srirangam": (10.854603680177885, 78.71354876350011),
        "Thillai Nagar": (10.834443055949594, 78.6600120927843),
        "Tiruchirappalli Fort": (10.825416706341354, 78.68725053920024),
        "BHEL Township": (10.779601260132063, 78.8020134465777),
        "Puthur": (10.816613926431312, 78.67548178421278),
        "K.K. Nagar": (10.754823739897988, 78.69703503252225),
        "Golden Rock": (10.786705928343814, 78.72749050086485),
        "Bharathidasan Nagar": (10.815605620780243, 78.6924032940072),
        "Abhishekapuram": (10.814276110961908, 78.68044154832714),
        "Cantonment": (10.801732520397492, 78.6930029284964),
        "Ariyamangalam": (10.810064973676322, 78.72098667200379),
        "Thuvakudi": (10.752956248763715, 78.83011220920471),
        "Edamalaipatti Pudur": (10.776544783166937, 78.67361068970932),
        "Senthaneerpuram": (10.819999333384466, 78.68414874613045),
        "Manikandam": (10.738831764536368, 78.63764762746621)
    },
    "Salem": {
        "Mettur": (11.78386032420848, 77.79660596982747),
        "Omalur": (11.743163682110064, 78.04652127028925),
        "Yercadu": (11.775337941924352, 78.21107854167128),
        "Salem": (11.667494933572193, 78.15066371370854),
        "Edappadi": (11.585847030527223, 77.83969762580683),
        "Sankari": (11.474599174184794, 77.86916242450349),
        "Vazhappadi": (11.657002286254166, 78.40124410527613),
        "Attur": (11.599410748474934, 78.59906606474027),
        "Gangavalli": (11.497384898316948, 78.65202007750166)
    }
}


def generate_weekly_data(start_year=2020, end_year=2025):
    """Generate more realistic weekly dengue risk data with clear patterns."""
    rows = []
    
    for city, city_zones in zones.items():
        for zone, (lat, lon) in city_zones.items():
            current_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)
            
            # Zone-specific baseline risk (some areas are inherently higher risk)
            zone_risk = np.random.normal(0, 10)  # Base risk modifier for this zone
            
            # Track previous values for temporal patterns
            prev_weeks = []
            
            while current_date <= end_date:
                year = current_date.year
                week = current_date.isocalendar()[1]
                month = current_date.month
                
                # 1. Generate environmental features with clear seasonal patterns
                # Temperature (seasonal pattern)
                base_temp = 25 + 10 * np.sin(2 * np.pi * (week + 10)/52)  # Peaks in summer
                temp = np.clip(np.random.normal(base_temp, 2), 15, 40)
                
                # Humidity (higher in monsoon)
                base_humidity = 60 + 20 * np.sin(2 * np.pi * (week - 20)/52)
                humidity = np.clip(np.random.normal(base_humidity, 5), 40, 100)
                
                # Rainfall (monsoon pattern)
                if month in [6, 7, 8, 9, 10]:  # Monsoon months
                    rainfall = np.random.gamma(shape=2, scale=25)  # Higher in monsoon
                else:
                    rainfall = np.random.gamma(shape=1, scale=5)  # Lower otherwise
                
                # 2. Generate other features
                sanitation = np.clip(np.random.normal(75, 10), 40, 100)
                pop_density = random.randint(1000, 15000)
                
                # 3. Generate dengue cases based on features (with clear relationships)
                # Base risk factors
                temp_effect = 0.5 * (temp - 25)  # Optimal around 25°C
                humidity_effect = 0.3 * (humidity - 50)  # Higher humidity increases risk
                rain_effect = 0.2 * np.log1p(rainfall)  # Log scale for rainfall
                sanitation_effect = -0.4 * (100 - sanitation)  # Lower sanitation increases risk
                
                # Seasonal effect (peaks in post-monsoon)
                seasonal_effect = 10 * np.sin(2 * np.pi * (week - 35)/52)  # Peaks around September
                
                # Temporal autocorrelation (carry-over effect from previous weeks)
                prev_effect = 0
                if prev_weeks:
                    prev_effect = 0.3 * np.mean([w['dengue_cases'] for w in prev_weeks[-4:]])  # Last 4 weeks
                
                # Combine all effects to get dengue cases
                dengue_cases = max(0, 
                    20 +  # Base level
                    temp_effect +
                    humidity_effect +
                    rain_effect +
                    sanitation_effect +
                    seasonal_effect +
                    prev_effect +
                    zone_risk +
                    np.random.normal(0, 3)  # Some random noise
                )
                dengue_cases = min(dengue_cases, 100)  # Cap at 100
                
                # 4. Calculate risk score (0-100) with clear relationship to features
                risk_score = (
                    0.20 * (temp - 20) / 15 +  # 20-35°C -> 0-20 points
                    0.20 * (humidity - 40) / 50 +  # 40-90% -> 0-20 points
                    0.15 * np.log1p(rainfall) / 2 +  # 0-200mm -> 0-15 points
                    0.10 * (100 - sanitation) / 50 +  # 50-100% -> 0-10 points
                    0.15 * dengue_cases / 50 +  # 0-100 cases -> 0-15 points
                    0.20 * prev_effect / 10  # Previous cases -> 0-20 points
                ) * 100  # Scale to 0-100
                
                risk_score = np.clip(risk_score, 0, 100)
                
                # Store this week's data
                row = {
                    'city': city,
                    'zone': zone,
                    'latitude': lat,
                    'longitude': lon,
                    'year': year,
                    'week': week,
                    'month': month,
                    'temperature': round(temp, 1),
                    'humidity': round(humidity, 1),
                    'rainfall': round(rainfall, 1),
                    'sanitation_score': round(sanitation, 1),
                    'population_density': pop_density,
                    'dengue_cases': int(round(dengue_cases)),
                    'risk_score': round(risk_score, 1)
                }
                
                # Keep track of recent weeks for temporal patterns
                prev_weeks.append(row.copy())
                if len(prev_weeks) > 4:  # Keep only last 4 weeks
                    prev_weeks.pop(0)
                
                rows.append(row)
                current_date += timedelta(weeks=1)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

def split_dataset(df):
    """Split data into train/validation/test sets."""
    train = df[df["year"].between(2020, 2022)]
    val = df[df["year"] == 2023]
    test = df[df["year"] == 2024]
    
    # Prediction input (first half 2025)
    pred_input = df[(df["year"] == 2025) & (df["week"] <= 26)]
    pred_target = df[(df["year"] == 2025) & (df["week"] > 26)]
    
    return train, val, test, pred_input, pred_target

def get_datasets(start_year=2020, end_year=2025):
    """Convenience function to generate data and return all dataset splits.
    Returns: train, val, test, pred_input, pred_target
    """
    df = generate_weekly_data(start_year=start_year, end_year=end_year)
    return split_dataset(df)

if __name__ == "__main__":
    print("Generating synthetic dengue risk data...")
    df = generate_weekly_data()
    
    print("\nSplitting data into train/validation/test sets...")
    train, val, test, pred_input, pred_target = split_dataset(df)
    
    # Save datasets
    os.makedirs("dataset", exist_ok=True)
    train.to_csv("dataset/train.csv", index=False)
    val.to_csv("dataset/val.csv", index=False)
    test.to_csv("dataset/test.csv", index=False)
    pred_input.to_csv("dataset/pred_input.csv", index=False)
    pred_target.to_csv("dataset/pred_target.csv", index=False)
    
    print("\n✅ Data generation complete!")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    print(f"Prediction Input (2025 H1): {pred_input.shape}, Target (2025 H2): {pred_target.shape}")
    print("\nSummary statistics of risk_score:")
    print(df['risk_score'].describe())
    print("\nFirst few rows of generated data:")
    print(df[['year', 'week', 'temperature', 'humidity', 'rainfall', 'risk_score']].head())