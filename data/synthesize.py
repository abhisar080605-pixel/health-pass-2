"""
HealthPass - Synthetic Kiosk Data Generator
Generates realistic physiological proxy signals mimicking the HealthPass IoT kiosk:
- MAX30102 (heart_rate, spo2)
- MLX90614 (body_temp)
- HX711 Load Cell (weight -> bmi)
- GSR sensor (gsr_stress proxy)
- MPU6050 (activity_level proxy)
- Blood pressure proxy (systolic_bp)
- Deliberate noise features (ambient_lux, rfid_device_hash, kiosk_uptime_mins)
  used to validate that the Adaptive Genetic Algorithm effectively prunes uninformative inputs.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_kiosk_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic sensor records representing patient screenings.
    Target label 'elevated_risk' (0 or 1) is derived from non-linear combinations
    of physiological proxies (BMI, age, systolic BP, activity, GSR, heart rate).
    """
    rng = np.random.default_rng(random_state)

    # 1. Primary physiological features
    age = rng.integers(18, 80, size=n_samples)
    
    # BMI: log-normal distribution centered around 27
    bmi = np.clip(rng.normal(loc=26.8, scale=5.2, size=n_samples), 16.0, 48.0)
    
    # Systolic Blood Pressure: correlated with age and BMI
    systolic_bp = np.clip(
        100 + 0.35 * age + 0.7 * (bmi - 22) + rng.normal(0, 10, size=n_samples),
        90, 195
    )
    
    # Physical activity score (1 = sedentary, 5 = athletic/very active)
    # Sedentary more likely in older/higher BMI
    activity_logits = 3.5 - 0.02 * age - 0.04 * (bmi - 22) + rng.normal(0, 0.8, size=n_samples)
    activity_level = np.clip(np.round(activity_logits), 1, 5).astype(int)
    
    # Heart rate: MAX30102 (bpm)
    heart_rate = np.clip(
        68 + 0.3 * (bmi - 22) + 0.15 * (age - 30) - 2.0 * (activity_level - 1) + rng.normal(0, 8, size=n_samples),
        50, 130
    )
    
    # SpO2: MAX30102 (%)
    spo2 = np.clip(
        98.5 - 0.03 * (age - 20) - 0.05 * np.maximum(0, bmi - 30) + rng.normal(0, 1.0, size=n_samples),
        90.0, 100.0
    )
    
    # Body temperature: MLX90614 (°C)
    body_temp = np.clip(rng.normal(loc=36.6, scale=0.35, size=n_samples), 35.5, 38.5)
    
    # GSR stress index (0.0 to 10.0 scale, sympathetic arousal)
    gsr_stress = np.clip(rng.normal(loc=4.5, scale=1.8, size=n_samples), 0.5, 9.8)

    # 2. Deliberate uninformative noise features (must be pruned by GA)
    ambient_lux = rng.uniform(50, 950, size=n_samples)          # Photoresistor kiosk lighting
    rfid_device_hash = rng.integers(1000, 9999, size=n_samples) # Hardware chip serial hash
    kiosk_uptime_mins = rng.integers(1, 1440, size=n_samples)   # Kiosk session duration

    # 3. Ground truth risk log-odds calculation (non-linear proxy modeling)
    # Risk factors: age (>45), BMI (>25, steep >30), Systolic BP (>130), low activity, elevated GSR
    z = (
        -1.8
        + 0.055 * (age - 40)
        + 0.18 * (bmi - 25)
        + 0.035 * (systolic_bp - 120)
        - 0.55 * (activity_level - 3)
        + 0.15 * (gsr_stress - 5)
        + 0.02 * (heart_rate - 72)
        + 0.005 * (bmi - 25) * (age - 40) # interaction term
        + rng.logistic(loc=0, scale=0.4, size=n_samples) # clinical noise
    )
    
    risk_prob = 1.0 / (1.0 + np.exp(-z))
    elevated_risk = (risk_prob >= 0.45).astype(int)

    df = pd.DataFrame({
        "age": np.round(age, 1),
        "bmi": np.round(bmi, 2),
        "heart_rate": np.round(heart_rate, 1),
        "spo2": np.round(spo2, 1),
        "body_temp": np.round(body_temp, 2),
        "gsr_stress": np.round(gsr_stress, 2),
        "activity_level": activity_level,
        "systolic_bp": np.round(systolic_bp, 1),
        "ambient_lux": np.round(ambient_lux, 1),
        "rfid_device_hash": rfid_device_hash,
        "kiosk_uptime_mins": kiosk_uptime_mins,
        "elevated_risk": elevated_risk
    })

    return df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_kiosk_data.csv")
    df = generate_synthetic_kiosk_data(n_samples=1200, random_state=42)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic kiosk records to {out_path}")
    print(f"Class distribution: {df['elevated_risk'].value_counts().to_dict()}")
    print("Features:", list(df.columns))
