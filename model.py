"""
HealthPass - Risk Model Pipeline & Classification Engine

Implements:
1. Baseline Classifier (RandomForest on ALL sensor & proxy features)
2. Adaptive GA Classifier (RandomForest on GA-optimized feature subset)
3. Performance comparison reporting (Accuracy, F1, Precision, Recall, Feature Count)
4. Inference engine with risk tier stratification, explainability, and clinical disclaimer.

Framing Guarantee:
Outputs risk screening scores based on physiological proxies (no glucose sensor).
Strictly separates opportunistic screening from clinical diagnostic confirmation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ga_selector import AdaptiveGeneticSelector
from explain import RiskExplainer

# Clinical screening disclaimer required across all touchpoints
CLINICAL_DISCLAIMER = (
    "SCREENING RESULT ONLY: HealthPass evaluates physiological proxy signals "
    "(BMI, blood pressure, heart rate, physical activity, autonomic stress, age) "
    "collected via non-invasive kiosk sensors. HealthPass DOES NOT contain a blood glucose "
    "sensor and CANNOT diagnose diabetes mellitus or prediabetes. An elevated risk screening "
    "warrants confirmatory diagnostic laboratory testing (HbA1c / Fasting Plasma Glucose) "
    "by a qualified medical professional."
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_kiosk_data.csv")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthpass_model.joblib")


class RiskModelPipeline:
    """
    Manages end-to-end training, GA feature selection, baseline comparisons,
    explainability, and inference.
    """
    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.all_features: List[str] = []
        self.selected_features: List[str] = []
        self.pruned_features: List[str] = []
        
        self.baseline_model: Optional[RandomForestClassifier] = None
        self.ga_model: Optional[RandomForestClassifier] = None
        self.selector: Optional[AdaptiveGeneticSelector] = None
        self.explainer: Optional[RiskExplainer] = None
        
        self.metrics: Dict[str, Any] = {}
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.is_trained: bool = False
        self.trained_at: Optional[str] = None

    def load_dataset(self, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset. SWAP POINT NOTE:
        To swap to a real kiosk CSV export or PIMA dataset:
        Simply point 'path' to your CSV. Ensure the target column is named
        'elevated_risk' (or 'Outcome') and sensor columns match kiosk inputs.
        """
        target_path = path or self.data_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Dataset file not found at: {target_path}")

        df = pd.read_csv(target_path)
        
        # Support PIMA or real kiosk column conventions
        target_col = "elevated_risk"
        if target_col not in df.columns:
            if "Outcome" in df.columns:
                target_col = "Outcome"
            elif "target" in df.columns:
                target_col = "target"
            else:
                raise ValueError(f"Target column 'elevated_risk' (or 'Outcome'/'target') not found in {list(df.columns)}")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    def train(
        self,
        dataset_path: Optional[str] = None,
        n_population: int = 30,
        n_generations: int = 30,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Run GA feature selection, train baseline vs. GA-selected model,
        and compute evaluation metrics.
        """
        X, y = self.load_dataset(dataset_path)
        self.all_features = list(X.columns)

        # Compute baseline descriptive stats for imputations and explainability
        self.baseline_stats = {}
        for col in self.all_features:
            self.baseline_stats[col] = {
                "median": float(X[col].median()),
                "mean": float(X[col].mean()),
                "std": float(X[col].std()) if X[col].std() > 0 else 1.0,
                "min": float(X[col].min()),
                "max": float(X[col].max())
            }

        # Train/test split (80/20 stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        print(f"\n[HealthPass Pipeline] Starting GA Feature Selection on {len(self.all_features)} initial features...")
        # 1. Run Adaptive Genetic Selector
        self.selector = AdaptiveGeneticSelector(
            n_population=n_population,
            n_generations=n_generations,
            w1=0.8,
            w2=0.2,
            random_state=random_state,
            verbose=True
        )
        self.selector.fit(X_train, y_train)

        self.selected_features = self.selector.best_features_
        self.pruned_features = self.selector.pruned_features_

        # 2. Train Baseline Model (ALL features)
        print("\n[HealthPass Pipeline] Training Baseline Model on ALL features...")
        self.baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        )
        self.baseline_model.fit(X_train, y_train)
        y_pred_base = self.baseline_model.predict(X_test)

        # 3. Train GA-Selected Model (OPTIMIZED feature subset only)
        print(f"[HealthPass Pipeline] Training Adaptive Model on GA subset ({len(self.selected_features)} features)...")
        self.ga_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1
        )
        X_train_ga = X_train[self.selected_features]
        X_test_ga = X_test[self.selected_features]
        self.ga_model.fit(X_train_ga, y_train)
        y_pred_ga = self.ga_model.predict(X_test_ga)

        # 4. Compute Comprehensive Evaluation Metrics
        base_metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred_base)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_base, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred_base, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_base, zero_division=0)), 4),
            "feature_count": len(self.all_features),
            "features": self.all_features
        }

        ga_metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred_ga)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_ga, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred_ga, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_ga, zero_division=0)), 4),
            "feature_count": len(self.selected_features),
            "features": self.selected_features,
            "pruned_features": self.pruned_features
        }

        feature_reduction_pct = round((1.0 - (len(self.selected_features) / len(self.all_features))) * 100, 1)
        acc_delta = round(ga_metrics["accuracy"] - base_metrics["accuracy"], 4)
        f1_delta = round(ga_metrics["f1_score"] - base_metrics["f1_score"], 4)

        self.metrics = {
            "baseline": base_metrics,
            "ga_selected": ga_metrics,
            "comparison": {
                "accuracy_delta": acc_delta,
                "f1_delta": f1_delta,
                "features_pruned_count": len(self.pruned_features),
                "feature_reduction_percent": feature_reduction_pct,
                "pruned_features": self.pruned_features
            }
        }

        # 5. Initialize Explainability Module
        self.explainer = RiskExplainer(
            model=self.ga_model,
            feature_names=self.selected_features,
            baseline_stats=self.baseline_stats
        )

        self.is_trained = True
        self.trained_at = datetime.now(timezone.utc).isoformat()

        # Save convergence chart artifact
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        os.makedirs(static_dir, exist_ok=True)
        chart_path = os.path.join(static_dir, "ga_convergence.png")
        self.selector.plot_convergence(save_path=chart_path)

        return self.metrics

    def predict_risk(self, readings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate patient risk given kiosk sensor readings.
        """
        if not self.is_trained or self.ga_model is None or self.explainer is None:
            raise RuntimeError("Model pipeline has not been trained yet.")

        # Impute any missing selected features using cohort medians
        sample_dict = {}
        missing_imputed = []
        for feat in self.selected_features:
            if feat in readings and readings[feat] is not None:
                sample_dict[feat] = float(readings[feat])
            else:
                median_val = self.baseline_stats.get(feat, {}).get("median", 0.0)
                sample_dict[feat] = median_val
                missing_imputed.append(feat)

        sample_df = pd.DataFrame([sample_dict])[self.selected_features]

        # Model risk probability prediction
        probabilities = self.ga_model.predict_proba(sample_df)[0]
        # Class 1 is elevated risk
        risk_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        risk_score_100 = round(risk_probability * 100.0, 1)

        # Map to calibrated screening risk tiers
        if risk_score_100 < 35.0:
            risk_label = "Low Risk Screened"
            recommendation = "Maintain regular healthy lifestyle, diet, and physical activity. Opportunistic rescreening in 12 months."
        elif risk_score_100 < 65.0:
            risk_label = "Moderate Risk - Lifestyle Monitoring Advised"
            recommendation = "Borderline physiological indicators detected. Adopt preventive nutritional and exercise habits; consult campus health center for routine wellness check."
        else:
            risk_label = "Elevated Risk - Clinical Follow-up Recommended"
            recommendation = "Multiple physiological risk proxies are significantly elevated. Strongly advise scheduling formal diagnostic laboratory testing (HbA1c, FPG) with a medical physician."

        # Get local feature explanation
        explanation_res = self.explainer.explain_instance(sample_df, top_k=4)

        return {
            "risk_score": risk_score_100,
            "risk_probability": round(risk_probability, 4),
            "risk_label": risk_label,
            "clinical_recommendation": recommendation,
            "top_contributing_features": explanation_res["top_contributing_features"],
            "explainability_method": explanation_res["method"],
            "features_used_in_prediction": self.selected_features,
            "features_imputed": missing_imputed,
            "disclaimer": CLINICAL_DISCLAIMER,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def save(self, filepath: str = DEFAULT_MODEL_PATH):
        """Serialize trained pipeline to disk."""
        data = {
            "all_features": self.all_features,
            "selected_features": self.selected_features,
            "pruned_features": self.pruned_features,
            "baseline_model": self.baseline_model,
            "ga_model": self.ga_model,
            "metrics": self.metrics,
            "baseline_stats": self.baseline_stats,
            "selector_history": self.selector.history_ if self.selector else [],
            "trained_at": self.trained_at
        }
        joblib.dump(data, filepath)
        print(f"[HealthPass Pipeline] Saved model artifact to {filepath}")

    def load(self, filepath: str = DEFAULT_MODEL_PATH) -> bool:
        """Load trained pipeline from disk."""
        if not os.path.exists(filepath):
            return False
        data = joblib.load(filepath)
        self.all_features = data["all_features"]
        self.selected_features = data["selected_features"]
        self.pruned_features = data["pruned_features"]
        self.baseline_model = data["baseline_model"]
        self.ga_model = data["ga_model"]
        self.metrics = data["metrics"]
        self.baseline_stats = data["baseline_stats"]
        self.trained_at = data["trained_at"]

        self.explainer = RiskExplainer(
            model=self.ga_model,
            feature_names=self.selected_features,
            baseline_stats=self.baseline_stats
        )
        self.is_trained = True
        print(f"[HealthPass Pipeline] Loaded model artifact from {filepath}")
        return True
