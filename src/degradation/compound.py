"""
AFS-VFM Compound Degradation Engine
===================================

Simulates realistic compound environmental and sensor failures by composing
multiple degradation transformations simultaneously.

Supports pre-defined real-world benchmark recipes as well as custom ad-hoc
chains with independent severity weighting.

Canonical Benchmark Recipes:
----------------------------
1. ``night_motion``:
   Low-light darkness + horizontal motion blur (automotive night driving).
2. ``adverse_weather``:
   Light attenuation + translucent rain/water spot occlusion + motion blur.
3. ``sensor_stress``:
   Pixelation/downscaling + motion blur + occlusion (drone/robotics bandwidth drop).
4. ``turbulent_flight``:
   3D perspective/viewpoint shear + motion blur (high-speed camera jitter).
"""

import numpy as np
from typing import Union, Sequence, Dict, Any, List

from . import transformations


COMPOUND_RECIPES: Dict[str, List[Dict[str, Any]]] = {
    "night_motion": [
        {"type": "lighting", "weight": 1.0},
        {"type": "motion_blur", "weight": 0.8},
    ],
    "adverse_weather": [
        {"type": "lighting", "weight": 0.7},
        {
            "type": "occlusion",
            "weight": 0.7,
            "params": {"mode": "irregular", "texture": "semi_transparent", "alpha": 0.75},
        },
        {"type": "motion_blur", "weight": 0.5},
    ],
    "sensor_stress": [
        {"type": "scale", "weight": 1.0},
        {"type": "motion_blur", "weight": 0.6},
        {"type": "occlusion", "weight": 0.5},
    ],
    "turbulent_flight": [
        {"type": "viewpoint", "weight": 1.0},
        {"type": "motion_blur", "weight": 0.7},
    ],
}


class CompoundDegradation:
    """Orchestrates chaining multiple degradation transformations onto an image."""

    _TRANSFORMS: dict = {
        "blur": transformations.motion_blur,
        "motion_blur": transformations.motion_blur,
        "occlusion": transformations.occlusion,
        "lighting": transformations.lighting,
        "low_light": transformations.lighting,
        "scale": transformations.scale,
        "pixelate": transformations.scale,
        "viewpoint": transformations.viewpoint,
        "shear": transformations.viewpoint,
    }

    @classmethod
    def list_recipes(cls) -> List[str]:
        """Return list of canonical compound recipe names."""
        return sorted(COMPOUND_RECIPES.keys())

    @classmethod
    def parse_components(
        cls,
        spec: Union[str, Sequence[Union[str, Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Normalize recipe names, plus-separated strings, or lists into standardized component dicts."""
        if isinstance(spec, str):
            spec_key = spec.lower().strip()
            if spec_key in COMPOUND_RECIPES:
                return [dict(c) for c in COMPOUND_RECIPES[spec_key]]
            elif "+" in spec_key:
                parts = [p.strip() for p in spec_key.split("+") if p.strip()]
                return [{"type": p, "weight": 1.0} for p in parts]
            else:
                return [{"type": spec_key, "weight": 1.0}]

        components: List[Dict[str, Any]] = []
        for item in spec:
            if isinstance(item, str):
                components.append({"type": item.lower().strip(), "weight": 1.0})
            elif isinstance(item, dict):
                comp = dict(item)
                comp["type"] = comp["type"].lower().strip()
                comp.setdefault("weight", 1.0)
                components.append(comp)
            else:
                raise ValueError(f"Invalid component specification: {item}")
        return components

    @classmethod
    def apply(
        cls,
        image: np.ndarray,
        severity: float,
        components: Union[str, Sequence[Union[str, Dict[str, Any]]]],
        **global_params
    ) -> np.ndarray:
        """Apply a compound sequence of transformations sequentially.

        Parameters
        ----------
        image : np.ndarray
            RGB uint8 array of shape (H, W, 3).
        severity : float
            Base progression severity in [0.0, 1.0].
        components : str or sequence
            Recipe name (e.g. 'night_motion'), plus-delimited string
            (e.g. 'lighting+motion_blur'), or list of component dicts.
        **global_params :
            Additional parameter overrides.

        Returns
        -------
        np.ndarray
            Compounded RGB uint8 image.
        """
        if severity <= 0.0:
            return image.copy()

        parsed_components = cls.parse_components(components)
        current = image.copy()

        for comp in parsed_components:
            comp_type = comp["type"]
            weight = float(comp.get("weight", 1.0))
            comp_sev = float(np.clip(severity * weight, 0.0, 1.0))

            transform_fn = cls._TRANSFORMS.get(comp_type)
            if transform_fn is None:
                raise ValueError(
                    f"Unknown degradation type '{comp_type}'. Supported: {sorted(cls._TRANSFORMS.keys())}"
                )

            # Merge per-component params with global params
            kwargs = dict(comp.get("params", {}))
            kwargs.update(global_params)

            if comp_type == "occlusion":
                current = transform_fn(current, comp_sev, **kwargs)
            else:
                current = transform_fn(current, comp_sev)

        return current
