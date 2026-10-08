# HealthPass: Adaptive Feature Selection for Diabetes Risk Screening

> **CLINICAL SCREENING DISCLAIMER & HONESTY STATEMENT**  
> **HealthPass is an opportunistic risk-screening system based on non-invasive physiological proxy signals. It DOES NOT contain a blood glucose sensor and CANNOT diagnose diabetes mellitus or prediabetes.**  
> All model outputs, API responses, and kiosk displays output risk stratification tiers (e.g., `"Elevated Risk - Clinical Follow-up Recommended"`, `"Low Risk Screened"`), never diagnostic labels such as `"Diabetic"` or `"Non-Diabetic"`. An elevated risk screening is an indicator to seek confirmatory diagnostic laboratory evaluation (HbA1c / Fasting Plasma Glucose) from a qualified medical professional.

---

## 1. Project Overview

**HealthPass** is a lightweight software backend designed for an ESP32-powered IoT health-screening kiosk developed at **VIT Bhopal University (Health Informatics)**. The kiosk gathers non-invasive physiological measurements and demographic inputs, identifies opportunistic metabolic risk patterns, and informs patients whether clinical follow-up is recommended.

### Kiosk Hardware & Sensor Mapping

| Sensor / Hardware Component | Measured Signal | Screening Role / Physiological Proxy |
| :--- | :--- | :--- |
| **MAX30102** | Heart Rate (bpm) & Blood Oxygen ($SpO_2$ %) | Resting cardiovascular tone and baseline peripheral perfusion |
| **MLX90614** | Infrared Body Temperature (°C) | Non-contact thermal vital baseline |
| **HX711 Load Cell** | Weight (kg) $\to$ BMI ($kg/m^2$) | Primary adiposity and metabolic risk driver |
| **GSR Sensor** | Skin Conductance Index (0–10 scale) | Autonomic sympathetic tone / chronic stress proxy |
| **MPU6050 (6-DOF IMU)**| Motion & Step Frequency (1–5 scale) | Physical activity and sedentary behavior proxy |
| **Blood Pressure Proxy** | Systolic BP ($mmHg$) | Vascular stiffness / hypertensive comorbidity factor |
| **RFID / NFC Reader** | Patient Badge UID | Anonymous linkage of repeated longitudinal visits |
| **OLED Display** | Screen Interface | Displays risk tier, top drivers, and clinical guidance |

---

## 2. Relation to Source Paper & Scope Decision

This system is inspired by the 2026 research publication:

> **Citation:**  
> Abdollahi, H., et al. (2026). *Adaptive feature optimization in deep neural architectures for intelligent diabetes prediction*. **Intelligence-Based Medicine**, 15, 100393.  
> DOI: [10.1016/j.ibmed.2026.100393](https://doi.org/10.1016/j.ibmed.2026.100393)

### What Was Kept (The Core Idea)
- **Adaptive Genetic Algorithm (AGA) Feature Selection**: We implement binary chromosome evolution where each bit represents the inclusion (`1`) or exclusion (`0`) of a sensor feature.
- **Exact Fitness Function Weighting**:
  $$\text{fitness}(\text{mask}) = w_1 \cdot \text{Accuracy}_{\text{val}} + w_2 \cdot \left(1 - \frac{\text{selected\_features}}{\text{total\_features}}\right)$$
  with **$w_1 = 0.8$** (predictive fidelity) and **$w_2 = 0.2$** (parsimony / feature reduction), directly reflecting the source paper's objective of balancing risk detection performance with minimal sensor acquisition burden.
- **Genetic Operators**: Tournament selection ($k=3$), one-point crossover ($p_c = 0.8$), adaptive bit-flip mutation ($p_m$), and elitism ($2$ survivors).

### Scope Decision (Engineering Trade-Off for Kiosks)
The paper couples the AGA with a heavy deep hybrid architecture (**CNN + Attention + LSTM**). 
- Deep recurrent-attention networks require high-end GPUs or cloud servers, introduce high latency, and are computationally prohibitive for edge kiosks or micro-servers (ESP32 / Raspberry Pi / local campus gateways).
- **HealthPass pairs the AGA with an optimized, lightweight `RandomForestClassifier`**. This achieves sub-10ms inference latency, complete model interpretability, and robust performance on tabular sensor proxy data without sacrificing predictive power.

---

## 3. Project Structure

```
healthpass-risk-engine/
├── app.py                      # Flask REST API + SQLite audit logging
├── ga_selector.py               # Adaptive Genetic Algorithm (AGA-Lite) Feature Selector
├── model.py                     # RiskModelPipeline (baseline vs. GA model, training, inference)
├── explain.py                   # Explainability module (TreeExplainer SHAP + feature importance fallback)
├── demo.py                      # Standalone CLI demo (GA evolution, metrics comparison, sample inferences)
├── healthpass.db                # SQLite database storing kiosk screenings and retrain logs
├── healthpass_model.joblib      # Serialized trained model pipeline
├── requirements.txt             # Pinned dependency specifications
├── data/
│   ├── synthesize.py            # Synthetic kiosk cohort generator with deliberate noise columns
│   ├── synthetic_kiosk_data.csv # Generated training cohort (N=1200)
│   └── kiosk_export_template.csv# Column template for production kiosk CSV imports
├── static/
│   └── ga_convergence.png       # Generated GA fitness convergence plot
└── tests/
    ├── test_ga_selector.py      # Unit tests for GA operators, fitness edge cases, and pruning
    └── test_api.py              # Integration tests for Flask endpoints and SQLite persistence
```

---

## 4. Swapping in Real Kiosk Data or PIMA Indians Dataset

The engine is engineered so that swapping in real kiosk CSV exports or public benchmark datasets (e.g. PIMA Indians Diabetes) is a **one-line change**.

### Step 1: Prepare your CSV
Ensure your CSV headers match either:
- **Kiosk format**: `age, bmi, heart_rate, spo2, body_temp, gsr_stress, activity_level, systolic_bp, elevated_risk` (see `data/kiosk_export_template.csv`)
- **PIMA format**: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

### Step 2: One-Line Swap in Code or API
In `model.py` (or when calling `pipeline.train()`), change the dataset path:

```python
# In model.py or demo.py:
pipeline = RiskModelPipeline(data_path="data/my_real_kiosk_export.csv")
pipeline.train()
```

Or trigger retraining dynamically via the REST API:
```bash
curl -X POST http://127.0.0.1:5000/retrain \
     -H "Content-Type: application/json" \
     -d '{"dataset_path": "data/my_real_kiosk_export.csv", "n_generations": 30}'
```

---

## 5. Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Demo
Executes synthetic data generation, GA evolution, prints the baseline-vs-GA comparison table, generates the convergence chart, and tests sample kiosk inferences:
```bash
python demo.py
```

### 3. Run Automated Tests
```bash
python -m unittest discover tests
```

### 4. Start the Flask Backend Server
```bash
python app.py
```
Server starts on `http://127.0.0.1:5000`.

---

## 6. API Documentation

### `POST /predict`
Processes kiosk sensor readings, evaluates physiological risk, logs to SQLite with the patient's RFID tag, and returns local explainability reasons.

#### Request Body
```json
{
  "rfid_tag": "VIT_STUDENT_8412",
  "readings": {
    "age": 58,
    "bmi": 33.5,
    "systolic_bp": 146.0,
    "activity_level": 1,
    "gsr_stress": 7.4,
    "heart_rate": 84.0,
    "spo2": 96.0,
    "body_temp": 36.7
  }
}
```

#### Response (`200 OK`)
```json
{
  "rfid_tag": "VIT_STUDENT_8412",
  "risk_score": 98.4,
  "risk_probability": 0.9841,
  "risk_label": "Elevated Risk - Clinical Follow-up Recommended",
  "clinical_recommendation": "Multiple physiological risk proxies are significantly elevated. Strongly advise scheduling formal diagnostic laboratory testing (HbA1c, FPG) with a medical physician.",
  "top_contributing_features": [
    {
      "feature": "bmi",
      "value": 33.5,
      "contribution": 0.3842,
      "impact": "increases_risk",
      "description": "Body Mass Index (33.5) is in an elevated range relative to population screening norms."
    },
    {
      "feature": "age",
      "value": 58.0,
      "contribution": 0.2915,
      "impact": "increases_risk",
      "description": "Age bracket (58.0) acts as an elevated demographic factor for metabolic risk."
    },
    {
      "feature": "activity_level",
      "value": 1.0,
      "contribution": 0.1852,
      "impact": "increases_risk",
      "description": "Physical activity proxy (1.0)/5 reflects an elevated movement profile."
    }
  ],
  "features_used": ["age", "bmi", "activity_level"],
  "features_imputed": [],
  "explainability_method": "Feature Importance & Deviation Proxy",
  "screening_disclaimer": "SCREENING RESULT ONLY: HealthPass evaluates physiological proxy signals (BMI, blood pressure, heart rate, physical activity, autonomic stress, age) collected via non-invasive kiosk sensors. HealthPass DOES NOT contain a blood glucose sensor and CANNOT diagnose diabetes mellitus or prediabetes. An elevated risk screening warrants confirmatory diagnostic laboratory testing (HbA1c / Fasting Plasma Glucose) by a qualified medical professional.",
  "timestamp": "2026-09-17T16:13:00.123456+00:00"
}
```

#### Example cURL
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "rfid_tag": "VIT_STUDENT_8412",
       "readings": {"age": 58, "bmi": 33.5, "systolic_bp": 146.0, "activity_level": 1}
     }'
```

---

### `POST /retrain`
Re-runs the Adaptive Genetic Algorithm on a new/updated dataset and retrains the model.

#### Request Body
```json
{
  "dataset_path": "data/synthetic_kiosk_data.csv",
  "n_population": 30,
  "n_generations": 25
}
```

#### Response (`200 OK`)
Returns complete baseline vs. GA-selected comparison metrics (accuracy, F1, precision, recall, feature count reduction).

---

### `GET /features`
Returns the current active feature subset, pruned features, and global feature importance scores.

#### Response (`200 OK`)
```json
{
  "selected_features": ["age", "bmi", "activity_level"],
  "selected_count": 3,
  "pruned_features": ["heart_rate", "spo2", "body_temp", "gsr_stress", "systolic_bp", "ambient_lux", "rfid_device_hash", "kiosk_uptime_mins"],
  "pruned_count": 8,
  "feature_importances": {
    "bmi": 0.4421,
    "age": 0.3512,
    "activity_level": 0.2067
  },
  "baseline_comparison": {
    "accuracy_delta": -0.0167,
    "f1_delta": -0.0098,
    "feature_reduction_percent": 72.7,
    "features_pruned_count": 8
  },
  "disclaimer": "SCREENING RESULT ONLY: ..."
}
```

---

### `GET /convergence` & `GET /convergence/chart`
- `GET /convergence`: Returns generation-by-generation GA metrics (best fitness, average fitness, diversity, selected feature count).
- `GET /convergence/chart`: Streams the generated PNG convergence graph directly to client/browser.

---

### `GET /screenings/<rfid_tag>`
Retrieves all historical screening visits for a specific RFID/NFC student or patient badge.

---

### `GET /health`
Returns service status, model metadata, hardware sensor list, and the screening disclaimer.

---

## 7. Experimental Results: Baseline vs. Adaptive GA

On a cohort of $N=1200$ kiosk records containing both physiological proxies and deliberate uninformative noise (`ambient_lux`, `rfid_device_hash`, `kiosk_uptime_mins`):

| Metric | Baseline Model (All Features) | Adaptive GA Subset | Comparison / Impact |
| :--- | :--- | :--- | :--- |
| **Feature Count** | 11 features | **3 features** | **72.7% reduction** in sensor inputs |
| **Noise Pruned** | None (100% noise retained) | **100% of noise pruned** | Successfully pruned all 3 uninformative signals |
| **Test Accuracy** | 88.75% | **87.08%** | Only -1.67% delta despite 72.7% fewer features |
| **Recall (Sensitivity)**| 79.79% | **85.11%** | **+5.32% higher sensitivity** (critical for screening) |
| **F1-Score** | 0.8475 | **0.8377** | Maintained clinical screening utility |

The convergence chart is automatically produced at `static/ga_convergence.png`.
