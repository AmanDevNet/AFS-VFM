"""
AFS-VFM Phase 3B — Failure Predictor
======================================
The main predictor class that assesses model health based on internal signals
and triggers warnings when failure is predicted.

Usage:
    from src.earlywarning.failure_predictor import FailurePredictor
    predictor = FailurePredictor()
    result = predictor.assess("dinov2", {"top1_confidence": 0.45, "logit_margin": 0.2})
"""

import os, json
from typing import Dict, Optional, List, Any, Union


class FailurePredictor:
    """
    Monitors internal model signals during inference and predicts failure
    based on learned thresholds from Phase 2C analysis.

    Status levels:
        OK       — All signals within normal range
        WARNING  — One or more signals below warning threshold
        CRITICAL — One or more signals below critical threshold
    """

    def __init__(self, thresholds_path: Optional[str] = None) -> None:
        if thresholds_path is None:
            thresholds_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "thresholds.json"
            )
        if not os.path.exists(thresholds_path):
            raise FileNotFoundError(
                f"Thresholds file not found: {thresholds_path}\n"
                "Run: python -m src.earlywarning.learn_thresholds"
            )
        with open(thresholds_path) as f:
            self.thresholds: Dict[str, Any] = json.load(f)

    def assess(self, model_name: str, signals_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess current model health based on signal values.

        Args:
            model_name: One of 'dinov2', 'clip', 'detr'
            signals_dict: Dict mapping signal names to current values,
                          e.g. {"top1_confidence": 0.45, "logit_margin": 0.2}

        Returns:
            dict with keys:
                status: "OK" | "WARNING" | "CRITICAL"
                failure_probability: float (0-1)
                estimated_frames_to_failure: int or None
                triggering_signals: list of signal names that triggered
                signal_details: per-signal breakdown
                recommendation: str
        """
        model = model_name.lower()
        if model not in self.thresholds:
            return {
                "status": "UNKNOWN",
                "failure_probability": 0.0,
                "estimated_frames_to_failure": None,
                "triggering_signals": [],
                "signal_details": {},
                "recommendation": f"No thresholds available for model '{model_name}'",
            }

        model_thresholds = self.thresholds[model]
        triggering_warning = []
        triggering_critical = []
        signal_details = {}
        probabilities = []

        for signal_name, value in signals_dict.items():
            if signal_name not in model_thresholds:
                continue

            thr = model_thresholds[signal_name]
            warn_thr = thr["warning_threshold"]
            crit_thr = thr["critical_threshold"]
            opt_thr = thr["optimal_threshold"]

            # Determine signal status
            if value <= crit_thr:
                sig_status = "CRITICAL"
                triggering_critical.append(signal_name)
            elif value <= warn_thr:
                sig_status = "WARNING"
                triggering_warning.append(signal_name)
            else:
                sig_status = "OK"

            # Estimate failure probability using normalized distance from optimal threshold:
            # delta = max(0, (theta_opt - v) / |theta_opt|)
            # Maps to linear probability ramp in [0.5, 1.0] when below optimal threshold
            if opt_thr != 0:
                distance_ratio = max(0, (opt_thr - value) / abs(opt_thr))
                prob = min(1.0, max(0.0, 0.5 + distance_ratio * 0.5))
            else:
                prob = 0.5 if value < 0 else 0.1

            probabilities.append(prob)
            signal_details[signal_name] = {
                "value": round(value, 6),
                "status": sig_status,
                "warning_threshold": round(warn_thr, 4),
                "critical_threshold": round(crit_thr, 4),
                "failure_probability": round(prob, 4),
            }

        # Overall status
        if triggering_critical:
            status = "CRITICAL"
            est_frames = 1
        elif triggering_warning:
            status = "WARNING"
            est_frames = 3
        else:
            status = "OK"
            est_frames = None

        # Overall probability
        failure_prob = max(probabilities) if probabilities else 0.0

        # Recommendation
        all_triggering = triggering_critical + triggering_warning
        if status == "CRITICAL":
            rec = (f"CRITICAL: Failure imminent. Signals {all_triggering} are below "
                   f"critical thresholds. Consider using fallback model or ensemble.")
        elif status == "WARNING":
            rec = (f"WARNING: Failure predicted in ~{est_frames} frames. "
                   f"Signals {all_triggering} are degrading. Monitor closely.")
        else:
            rec = "All signals within normal range. Prediction likely reliable."

        return {
            "status": status,
            "failure_probability": round(failure_prob, 4),
            "estimated_frames_to_failure": est_frames,
            "triggering_signals": all_triggering,
            "signal_details": signal_details,
            "recommendation": rec,
        }

    def assess_trajectory(
        self,
        model_name: str,
        signal_history: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Assess a full trajectory (list of signal dicts across frames).
        Returns per-frame assessments and identifies the first warning frame.
        """
        results = []
        first_warning = None
        first_critical = None

        for frame_idx, signals in enumerate(signal_history):
            assessment = self.assess(model_name, signals)
            assessment["frame"] = frame_idx + 1
            results.append(assessment)

            if assessment["status"] == "WARNING" and first_warning is None:
                first_warning = frame_idx + 1
            if assessment["status"] == "CRITICAL" and first_critical is None:
                first_critical = frame_idx + 1

        return {
            "trajectory": results,
            "first_warning_frame": first_warning,
            "first_critical_frame": first_critical,
            "total_frames": len(signal_history),
        }
