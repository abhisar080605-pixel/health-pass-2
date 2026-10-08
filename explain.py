"""
HealthPass - Risk Explainability Module

Provides interpretable explanations for patient risk screening results:
1. SHAP (TreeExplainer) for local per-prediction attribution if `shap` is installed.
2. Built-in RandomForest `.feature_importances_` and baseline deviation fallback
   if SHAP is not present or throws an exception.

Strictly framed around physiological risk proxies (BMI, age, systolic BP, activity,
stress, vitals) rather than diagnostic markers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Try importing SHAP
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


class RiskExplainer:
    """
    Explains risk screening predictions using SHAP or tree feature importance fallback.
    """
    def __init__(self, model, feature_names: List[str], baseline_stats: Optional[Dict[str, Dict[str, float]]] = None):
        self.model = model
        self.feature_names = feature_names
        self.baseline_stats = baseline_stats or {}
        self.explainer = None
        self.has_shap = _HAS_SHAP

        if self.has_shap:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.has_shap = False
                self.explainer = None

    def get_global_importances(self) -> Dict[str, float]:
        """Return normalized global feature importances from random forest."""
        if hasattr(self.model, "feature_importances_"):
            raw = self.model.feature_importances_
            total = np.sum(raw)
            norm = raw / total if total > 0 else raw
            return {
                feat: round(float(imp), 4)
                for feat, imp in sorted(zip(self.feature_names, norm), key=lambda x: x[1], reverse=True)
            }
        return {feat: 1.0 / len(self.feature_names) for feat in self.feature_names}

    def explain_instance(self, sample_row: pd.DataFrame, top_k: int = 4) -> Dict[str, Any]:
        """
        Explain a single patient screening prediction.
        Returns top contributing features and directional risk impacts.
        """
        if self.has_shap and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(sample_row)
                # For binary classification, shap_values can be a list of arrays [class_0, class_1]
                # or a 3D array (n_samples, n_features, n_classes)
                if isinstance(shap_values, list) and len(shap_values) >= 2:
                    vals = np.asarray(shap_values[1])[0] # elevated risk class
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    vals = shap_values[0, :, 1]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                    vals = shap_values[0]
                else:
                    vals = np.asarray(shap_values).ravel()

                contributions = []
                for feat, val in zip(self.feature_names, vals):
                    val_float = float(val)
                    impact = "increases_risk" if val_float > 0 else "decreases_risk"
                    raw_val = sample_row.iloc[0][feat] if feat in sample_row.columns else None
                    contributions.append({
                        "feature": feat,
                        "value": float(raw_val) if raw_val is not None else None,
                        "contribution": round(val_float, 4),
                        "abs_contribution": abs(val_float),
                        "impact": impact,
                        "description": self._generate_feature_sentence(feat, raw_val, impact)
                    })

                contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
                for c in contributions:
                    del c["abs_contribution"]

                return {
                    "method": "SHAP (TreeExplainer)",
                    "top_contributing_features": contributions[:top_k],
                    "all_contributions": contributions
                }
            except Exception as e:
                # Fall through to fallback explanation
                pass

        # Fallback: Tree feature importances + deviation from population norm
        return self._fallback_explanation(sample_row, top_k=top_k)

    def _fallback_explanation(self, sample_row: pd.DataFrame, top_k: int = 4) -> Dict[str, Any]:
        """
        Fallback when SHAP is unavailable: calculates deviation-weighted tree importance.
        """
        importances = self.get_global_importances()
        contributions = []

        # Risk-increasing directions for typical kiosk proxies:
        # High BMI, high systolic BP, high age, low activity, high GSR stress increase risk
        higher_is_risk = {"bmi", "systolic_bp", "age", "gsr_stress", "heart_rate"}
        lower_is_risk = {"activity_level", "spo2"}

        for feat in self.feature_names:
            weight = importances.get(feat, 0.1)
            raw_val = float(sample_row.iloc[0][feat]) if feat in sample_row.columns else None

            # Get baseline norm if available, else standard clinical defaults
            norm_val = self.baseline_stats.get(feat, {}).get("median", self._default_norm(feat))
            std_val = self.baseline_stats.get(feat, {}).get("std", 1.0)
            if std_val == 0:
                std_val = 1.0

            if raw_val is not None:
                z_score = (raw_val - norm_val) / std_val
                if feat in higher_is_risk:
                    impact = "increases_risk" if z_score > 0.1 else "decreases_risk"
                    score = z_score * weight
                elif feat in lower_is_risk:
                    impact = "increases_risk" if z_score < -0.1 else "decreases_risk"
                    score = -z_score * weight
                else:
                    impact = "increases_risk" if z_score > 0 else "decreases_risk"
                    score = z_score * weight

                contributions.append({
                    "feature": feat,
                    "value": raw_val,
                    "contribution": round(float(score), 4),
                    "abs_contribution": abs(float(score)),
                    "impact": impact,
                    "description": self._generate_feature_sentence(feat, raw_val, impact)
                })

        contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
        for c in contributions:
            del c["abs_contribution"]

        return {
            "method": "Feature Importance & Deviation Proxy",
            "top_contributing_features": contributions[:top_k],
            "all_contributions": contributions
        }

    def _default_norm(self, feature: str) -> float:
        defaults = {
            "age": 42.0,
            "bmi": 24.5,
            "systolic_bp": 120.0,
            "activity_level": 3.0,
            "heart_rate": 72.0,
            "spo2": 98.0,
            "body_temp": 36.6,
            "gsr_stress": 4.0
        }
        return defaults.get(feature, 0.0)

    def _generate_feature_sentence(self, feature: str, value: Optional[float], impact: str) -> str:
        """Friendly human-readable explanation for kiosk display."""
        direction_word = "elevated" if impact == "increases_risk" else "healthy/protective"
        val_str = f"({value})" if value is not None else ""

        narratives = {
            "bmi": f"Body Mass Index {val_str} is in an {direction_word} range relative to population screening norms.",
            "age": f"Age bracket {val_str} acts as an {direction_word} demographic factor for metabolic risk.",
            "systolic_bp": f"Systolic Blood Pressure {val_str} mmHg represents an {direction_word} vascular reading.",
            "activity_level": f"Physical activity proxy {val_str}/5 reflects an {direction_word} movement profile.",
            "gsr_stress": f"Galvanic Skin Response {val_str} indicates {direction_word} sympathetic autonomic tone.",
            "heart_rate": f"Resting Heart Rate {val_str} bpm aligns with an {direction_word} cardiac pattern.",
            "spo2": f"Blood oxygen saturation {val_str}% is within an {direction_word} range."
        }
        return narratives.get(feature, f"Sensor feature '{feature}' {val_str} {direction_word.replace('/', ' or ')} risk.")
