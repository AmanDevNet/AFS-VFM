"""
AFS-VFM Early Warning Feature Extraction
=========================================
Extracts static latent signals and kinematic temporal features (velocity,
acceleration, rolling window statistics) from trajectory sequences or streaming frames.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd


CORE_FEATURE_NAMES = [
    "confidence",
    "margin",
    "cosine_sim",
    "l2_distance",
    "delta_confidence",
    "delta_margin",
    "delta_cosine_sim",
    "delta_l2",
    "accel_cosine_sim",
    "rolling_mean_confidence",
    "rolling_std_confidence",
    "rolling_mean_cosine_sim",
]


class TemporalFeatureExtractor:
    """
    Extracts dynamic failure indicators from model signals.
    Supports both batch DataFrame processing and online streaming evaluation.
    """

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = window_size
        self.history: deque = deque(maxlen=max(window_size, 3))
        self.feature_names = list(CORE_FEATURE_NAMES)

    def reset(self) -> None:
        """Reset online stream history for a new trajectory/session."""
        self.history.clear()

    @staticmethod
    def _extract_base_signals(signals: Dict[str, Any], model_type: str = "classification") -> Dict[str, float]:
        """Normalize signal keys across classification (DINOv2/CLIP) and detection (DETR)."""
        # Confidence
        if "top1_confidence" in signals and signals["top1_confidence"] is not None:
            conf = float(signals["top1_confidence"])
        elif "top_detection_confidence" in signals and signals["top_detection_confidence"] is not None:
            conf = float(signals["top_detection_confidence"])
        elif "confidence" in signals and signals["confidence"] is not None:
            conf = float(signals["confidence"])
        else:
            conf = 0.5

        # Margin / Objectness
        if "logit_margin" in signals and signals["logit_margin"] is not None and not np.isnan(signals["logit_margin"]):
            margin = float(signals["logit_margin"])
        elif "objectness" in signals and signals["objectness"] is not None and not np.isnan(signals["objectness"]):
            margin = float(signals["objectness"])
        elif "margin" in signals and signals["margin"] is not None and not np.isnan(signals["margin"]):
            margin = float(signals["margin"])
        else:
            margin = 0.0

        # Cosine similarity
        cos_sim = signals.get("cosine_sim", 1.0)
        if cos_sim is None or np.isnan(cos_sim):
            cos_sim = 1.0
        else:
            cos_sim = float(cos_sim)

        # L2 distance
        l2 = signals.get("l2_distance", 0.0)
        if l2 is None or np.isnan(l2):
            l2 = 0.0
        else:
            l2 = float(l2)

        return {
            "confidence": conf,
            "margin": margin,
            "cosine_sim": cos_sim,
            "l2_distance": l2,
        }

    def update(self, signals: Dict[str, Any], model_type: str = "classification") -> np.ndarray:
        """
        Process a single streaming frame and return the extracted feature vector.

        Args:
            signals: Dictionary containing current frame signals
            model_type: 'classification' or 'detection'

        Returns:
            1D numpy array of shape (len(CORE_FEATURE_NAMES),)
        """
        base = self._extract_base_signals(signals, model_type=model_type)
        self.history.append(base)

        conf = base["confidence"]
        margin = base["margin"]
        cos_sim = base["cosine_sim"]
        l2 = base["l2_distance"]

        # 1st-order differences (velocity)
        if len(self.history) >= 2:
            prev = self.history[-2]
            d_conf = conf - prev["confidence"]
            d_margin = margin - prev["margin"]
            d_cos = cos_sim - prev["cosine_sim"]
            d_l2 = l2 - prev["l2_distance"]
        else:
            d_conf = 0.0
            d_margin = 0.0
            d_cos = 0.0
            d_l2 = 0.0

        # 2nd-order difference (acceleration of drift)
        if len(self.history) >= 3:
            prev = self.history[-2]
            prev2 = self.history[-3]
            prev_d_cos = prev["cosine_sim"] - prev2["cosine_sim"]
            accel_cos = d_cos - prev_d_cos
        else:
            accel_cos = 0.0

        # Rolling window statistics
        conf_history = [h["confidence"] for h in self.history]
        cos_history = [h["cosine_sim"] for h in self.history]

        roll_mean_conf = float(np.mean(conf_history))
        roll_std_conf = float(np.std(conf_history)) if len(conf_history) > 1 else 0.0
        roll_mean_cos = float(np.mean(cos_history))

        features = np.array([
            conf,
            margin,
            cos_sim,
            l2,
            d_conf,
            d_margin,
            d_cos,
            d_l2,
            accel_cos,
            roll_mean_conf,
            roll_std_conf,
            roll_mean_cos,
        ], dtype=np.float32)

        return features

    @classmethod
    def extract_batch(
        cls,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        sort_col: str = "frame",
    ) -> pd.DataFrame:
        """
        Extract dynamic kinematic features across an entire offline benchmark DataFrame.

        Args:
            df: DataFrame containing benchmark trajectory data
            group_cols: Columns defining an individual trajectory (default: ['image', 'degradation', 'model'])
            sort_col: Temporal sequence column (default: 'frame')

        Returns:
            DataFrame augmented with CORE_FEATURE_NAMES columns
        """
        if group_cols is None:
            available_groups = [c for c in ["image", "degradation", "model"] if c in df.columns]
            group_cols = available_groups if available_groups else ["image"]

        data = df.sort_values(by=group_cols + [sort_col]).copy()

        # Extract normalized base signals
        if "top1_confidence" in data.columns:
            conf_series = data["top1_confidence"].fillna(data.get("top_detection_confidence", 0.5))
        elif "top_detection_confidence" in data.columns:
            conf_series = data["top_detection_confidence"].fillna(0.5)
        else:
            conf_series = pd.Series(0.5, index=data.index)

        if "logit_margin" in data.columns:
            margin_series = data["logit_margin"].fillna(data.get("objectness", 0.0))
        elif "objectness" in data.columns:
            margin_series = data["objectness"].fillna(0.0)
        else:
            margin_series = pd.Series(0.0, index=data.index)

        cos_series = data["cosine_sim"].fillna(1.0) if "cosine_sim" in data.columns else pd.Series(1.0, index=data.index)
        l2_series = data["l2_distance"].fillna(0.0) if "l2_distance" in data.columns else pd.Series(0.0, index=data.index)

        data["confidence"] = conf_series.astype(float)
        data["margin"] = margin_series.astype(float)
        data["cosine_sim"] = cos_series.astype(float)
        data["l2_distance"] = l2_series.astype(float)

        grouped = data.groupby(group_cols, sort=False)

        # 1st order velocity differences (leave NaN for true 2nd diff)
        data["delta_confidence"] = grouped["confidence"].diff()
        data["delta_margin"] = grouped["margin"].diff()
        data["delta_cosine_sim"] = grouped["cosine_sim"].diff()
        data["delta_l2"] = grouped["l2_distance"].diff()

        # 2nd order acceleration computed on un-filled velocities
        grouped_del = data.groupby(group_cols, sort=False)
        data["accel_cosine_sim"] = grouped_del["delta_cosine_sim"].diff().fillna(0.0)

        # Now fill NaN for initial boundary frames
        data["delta_confidence"] = data["delta_confidence"].fillna(0.0)
        data["delta_margin"] = data["delta_margin"].fillna(0.0)
        data["delta_cosine_sim"] = data["delta_cosine_sim"].fillna(0.0)
        data["delta_l2"] = data["delta_l2"].fillna(0.0)

        # Rolling window stats (window=3, ddof=0 to match numpy streaming std)
        data["rolling_mean_confidence"] = grouped["confidence"].rolling(window=3, min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)
        data["rolling_std_confidence"] = grouped["confidence"].rolling(window=3, min_periods=1).std(ddof=0).fillna(0.0).reset_index(level=list(range(len(group_cols))), drop=True)
        data["rolling_mean_cosine_sim"] = grouped["cosine_sim"].rolling(window=3, min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)

        return data
