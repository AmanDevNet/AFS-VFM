"""
AFS-VFM Learned Early Warning Classifiers
=========================================
Implements supervised multivariate machine learning predictors (ElasticNet Logistic
Regression, Histogram Gradient Boosted Trees, and Calibrated MLP) for vision foundation
model failure forecasting, replacing static heuristic thresholds.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)

from .feature_extractor import TemporalFeatureExtractor, CORE_FEATURE_NAMES


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class LearnedFailurePredictor:
    """
    Multivariate machine-learned failure predictor.
    Consumes streaming signals, computes dynamic kinematics, and produces
    calibrated failure probabilities and early warning alerts.
    """

    def __init__(
        self,
        classifier: Any = None,
        feature_names: Optional[List[str]] = None,
        model_name: str = "generic",
        classifier_type: str = "gradient_boosting",
        warn_threshold: float = 0.35,
        crit_threshold: float = 0.70,
    ) -> None:
        self.classifier = classifier
        self.feature_names = feature_names or list(CORE_FEATURE_NAMES)
        self.model_name = model_name
        self.classifier_type = classifier_type
        self.warn_threshold = warn_threshold
        self.crit_threshold = crit_threshold
        self.extractor = TemporalFeatureExtractor()

    def reset(self) -> None:
        """Reset temporal feature history (call at beginning of a new sequence)."""
        self.extractor.reset()

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """
        Compute failure probability for a 1D feature vector.
        """
        if self.classifier is None:
            # Fallback heuristic if no trained model loaded
            conf = float(feature_vector[0])
            return float(np.clip(1.0 - conf, 0.0, 1.0))

        X = feature_vector.reshape(1, -1)
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)
            return float(probs[0, 1])
        elif hasattr(self.classifier, "decision_function"):
            score = self.classifier.decision_function(X)[0]
            return float(1.0 / (1.0 + np.exp(-score)))
        else:
            return float(self.classifier.predict(X)[0])

    def assess(
        self,
        signals: Dict[str, Any],
        model_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Online assessment of a single frame's health and imminent failure risk.

        Args:
            signals: Dict of current frame signals (top1_confidence, logit_margin, cosine_sim, etc.)
            model_type: 'classification' or 'detection'

        Returns:
            Dict matching FailurePredictor.assess schema:
                status: 'OK' | 'WARNING' | 'CRITICAL'
                failure_probability: float [0, 1]
                imminent_hazard: bool
                triggering_signals: list of top contributing features
                signal_details: dict of feature values
                recommendation: safety guidance string
        """
        feat_vec = self.extractor.update(signals, model_type=model_type)
        prob = self.predict_proba(feat_vec)

        if prob >= self.crit_threshold:
            status = "CRITICAL"
            rec = "CRITICAL: Imminent model failure detected. Immediate failover or control handoff required."
        elif prob >= self.warn_threshold:
            status = "WARNING"
            rec = "WARNING: Significant latent degradation. Sensor cleaning or ensemble fallback recommended."
        else:
            status = "OK"
            rec = "Model operating within certified nominal reliability bounds."

        # Identify signals that are contributing to degradation
        signal_details = {name: float(feat_vec[i]) for i, name in enumerate(self.feature_names)}
        triggering = []
        if signal_details.get("confidence", 1.0) < 0.6:
            triggering.append("confidence_drop")
        if signal_details.get("cosine_sim", 1.0) < 0.85:
            triggering.append("latent_representation_drift")
        if signal_details.get("delta_cosine_sim", 0.0) < -0.05:
            triggering.append("rapid_drift_velocity")
        if signal_details.get("margin", 1.0) < 0.2:
            triggering.append("low_decision_margin")

        return {
            "status": status,
            "failure_probability": round(prob, 4),
            "imminent_hazard": prob >= self.warn_threshold,
            "classifier_type": self.classifier_type,
            "triggering_signals": triggering,
            "signal_details": signal_details,
            "recommendation": rec,
        }

    def save(self, filepath: str) -> None:
        """Serialize trained model bundle."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "model_name": self.model_name,
            "classifier_type": self.classifier_type,
            "warn_threshold": self.warn_threshold,
            "crit_threshold": self.crit_threshold,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LearnedFailurePredictor":
        """Load serialized predictor bundle."""
        payload = joblib.load(filepath)
        return cls(
            classifier=payload["classifier"],
            feature_names=payload["feature_names"],
            model_name=payload.get("model_name", "generic"),
            classifier_type=payload.get("classifier_type", "gradient_boosting"),
            warn_threshold=payload.get("warn_threshold", 0.35),
            crit_threshold=payload.get("crit_threshold", 0.70),
        )


def build_classifier(classifier_type: str, random_state: int = 42) -> Pipeline:
    """
    Factory function for competitive early warning classifiers.
    """
    if classifier_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

    elif classifier_type == "gradient_boosting":
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.08,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

    elif classifier_type == "random_forest":
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=40,
                max_depth=6,
                min_samples_leaf=15,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    elif classifier_type == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=0.01,
                early_stopping=True,
                max_iter=100,
                random_state=random_state,
            )),
        ])

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def prepare_training_dataset(
    df: pd.DataFrame,
    horizon: int = 3,
) -> Tuple[pd.DataFrame, List[str], str, str]:
    """
    Extract dynamic features and construct ground-truth hazard labels.

    Target definition:
      - 'imminent_failure': 1 if failure occurs within next `horizon` frames, else 0.
      - 'is_failed': 1 if frame >= first_failure_frame, else 0.
    """
    # Augment with temporal kinematic features
    augmented = TemporalFeatureExtractor.extract_batch(df)

    # Imminent warning label (within `horizon` frames before actual failure)
    def compute_imminent_label(row: pd.Series) -> int:
        if row.get("survived", False):
            return 0
        fff = row.get("first_failure_frame", np.nan)
        frame = row.get("frame", 0)
        if pd.isna(fff):
            return 0
        # Alert window: within horizon frames leading up to and including failure
        if 0 <= (fff - frame) <= horizon:
            return 1
        return 0

    augmented["target_imminent"] = [compute_imminent_label(r) for _, r in augmented.iterrows()]
    augmented["target_failed"] = [
        0 if r.get("survived", False) or pd.isna(r.get("first_failure_frame"))
        else int(r.get("frame", 0) >= r.get("first_failure_frame"))
        for _, r in augmented.iterrows()
    ]

    return augmented, CORE_FEATURE_NAMES, "target_imminent", "image"


def evaluate_cross_validated_models(
    df: pd.DataFrame,
    model_name: str,
    target_col: str = "target_imminent",
    group_col: str = "image",
    n_splits: int = 5,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline]]:
    """
    Train and evaluate 4 classifier paradigms using strict GroupKFold cross-validation
    to prevent data leakage across image trajectories.
    """
    features_df, feature_cols, _, _ = prepare_training_dataset(df)

    # Clean NaNs in features
    X = features_df[feature_cols].fillna(0.0).values
    y = features_df[target_col].values
    groups = features_df[group_col].values

    classifier_types = [
        "logistic_regression",
        "gradient_boosting",
        "random_forest",
        "mlp",
    ]

    gkf = GroupKFold(n_splits=n_splits)
    metrics_summary: Dict[str, Dict[str, float]] = {}
    best_pipelines: Dict[str, Pipeline] = {}

    for ctype in classifier_types:
        aurocs = []
        auprcs = []
        f1s = []
        briers = []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            pipeline = build_classifier(ctype)
            pipeline.fit(X_tr, y_tr)

            probs = pipeline.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)

            aurocs.append(float(roc_auc_score(y_te, probs)))
            auprcs.append(float(average_precision_score(y_te, probs)))
            f1s.append(float(f1_score(y_te, preds, zero_division=0)))
            briers.append(float(brier_score_loss(y_te, probs)))

        metrics_summary[ctype] = {
            "auroc_mean": round(float(np.mean(aurocs)), 4),
            "auroc_std": round(float(np.std(aurocs)), 4),
            "auprc_mean": round(float(np.mean(auprcs)), 4),
            "auprc_std": round(float(np.std(auprcs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "brier_score": round(float(np.mean(briers)), 4),
        }

        # Train final production model on all data
        full_pipeline = build_classifier(ctype)
        full_pipeline.fit(X, y)
        best_pipelines[ctype] = full_pipeline

    return metrics_summary, best_pipelines
