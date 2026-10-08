"""
HealthPass - Standalone End-to-End Demo Script

Executes:
1. Verifies/generates synthetic kiosk cohort with realistic physiological proxies + noise.
2. Trains Baseline RandomForest on ALL features.
3. Runs Adaptive Genetic Algorithm (AGA) feature selector (w1=0.8 accuracy, w2=0.2 parsimony).
4. Trains GA-optimized RandomForest on selected feature subset.
5. Prints side-by-side performance comparison (Accuracy, F1, Precision, Recall, Feature Count).
6. Runs sample inference for:
   - High-risk kiosk reading
   - Low-risk kiosk reading
7. Displays local explainability breakdown and mandatory clinical disclaimer.
8. Saves trained model artifact and convergence plot.
"""

import os
import json
from data.synthesize import generate_synthetic_kiosk_data
from model import RiskModelPipeline, CLINICAL_DISCLAIMER

def run_demo():
    print("\n" + "="*70)
    print(" HEALTHPASS: ADAPTIVE FEATURE SELECTION FOR DIABETES RISK SCREENING")
    print(" Kiosk IoT Proof-of-Concept | VIT Bhopal University")
    print(" Inspired by Abdollahi et al. (2026), Intelligence-Based Medicine")
    print("="*70 + "\n")

    # 1. Prepare data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "synthetic_kiosk_data.csv")

    if not os.path.exists(data_path):
        print(f"[*] Generating synthetic kiosk screening dataset (N=1200)...")
        df = generate_synthetic_kiosk_data(n_samples=1200, random_state=42)
        df.to_csv(data_path, index=False)
    else:
        print(f"[*] Using existing synthetic kiosk screening dataset: {data_path}")

    # 2. Initialize and train pipeline
    pipeline = RiskModelPipeline(data_path=data_path)
    print("\n[*] Starting GA Evolution & Model Training...")
    metrics = pipeline.train(n_population=30, n_generations=25, random_state=42)

    # 3. Print Comparison Report
    base = metrics["baseline"]
    ga = metrics["ga_selected"]
    comp = metrics["comparison"]

    print("\n" + "="*70)
    print("                   BASELINE vs. GA-OPTIMIZED COMPARISON")
    print("="*70)
    print(f"{'Metric':<25} | {'Baseline (All Features)':<23} | {'Adaptive GA Subset':<20}")
    print("-" * 75)
    print(f"{'Feature Count':<25} | {base['feature_count']:<23} | {ga['feature_count']:<20} ({comp['feature_reduction_percent']}% reduction)")
    print(f"{'Test Accuracy':<25} | {base['accuracy']*100:>6.2f}%{' '*16} | {ga['accuracy']*100:>6.2f}% (delta: {comp['accuracy_delta']*100:+.2f}%)")
    print(f"{'F1 Score':<25} | {base['f1_score']:>6.4f}{' '*17} | {ga['f1_score']:>6.4f} (delta: {comp['f1_delta']:+.4f})")
    print(f"{'Precision':<25} | {base['precision']:>6.4f}{' '*17} | {ga['precision']:>6.4f}")
    print(f"{'Recall':<25} | {base['recall']:>6.4f}{' '*17} | {ga['recall']:>6.4f}")
    print("="*70)
    print(f"Selected Features ({len(ga['features'])}): {ga['features']}")
    print(f"Pruned Noise/Redundant Features ({len(ga['pruned_features'])}): {ga['pruned_features']}")
    print("="*70)

    # Save model artifact
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthpass_model.joblib")
    pipeline.save(model_path)

    # 4. Demonstrate Sample Inferences
    print("\n" + "="*70)
    print("                       SAMPLE KIOSK INFERENCES")
    print("="*70)

    patient_a = {
        "age": 63,
        "bmi": 34.2,
        "systolic_bp": 152.0,
        "activity_level": 1,
        "gsr_stress": 7.8,
        "heart_rate": 86.0,
        "spo2": 95.0,
        "body_temp": 36.7,
        "ambient_lux": 420.0,
        "rfid_device_hash": 9812,
        "kiosk_uptime_mins": 340
    }

    patient_b = {
        "age": 28,
        "bmi": 21.8,
        "systolic_bp": 114.0,
        "activity_level": 4,
        "gsr_stress": 2.5,
        "heart_rate": 64.0,
        "spo2": 99.0,
        "body_temp": 36.5,
        "ambient_lux": 750.0,
        "rfid_device_hash": 1204,
        "kiosk_uptime_mins": 85
    }

    for name, sample in [("Patient A (Senior, High BMI & BP, Sedentary)", patient_a),
                         ("Patient B (Young adult, Normal BMI & BP, Active)", patient_b)]:
        print(f"\n--- Testing Inference: {name} ---")
        res = pipeline.predict_risk(sample)
        print(f"Risk Score:    {res['risk_score']}% (Probability: {res['risk_probability']})")
        print(f"Risk Label:    {res['risk_label']}")
        print(f"Guidance:      {res['clinical_recommendation']}")
        print(f"Top Drivers ({res['explainability_method']}):")
        for idx, feat in enumerate(res["top_contributing_features"][:3], 1):
            print(f"  {idx}. {feat['feature']} = {feat['value']}: {feat['impact']} ({feat['description']})")

    print("\n" + "="*70)
    print(" CLINICAL FRAMING & DISCLAIMER CHECK:")
    print("="*70)
    print(CLINICAL_DISCLAIMER)
    print("="*70 + "\n")
    print("[OK] Demo completed successfully! Convergence plot saved to static/ga_convergence.png")

if __name__ == "__main__":
    run_demo()
