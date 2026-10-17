"""
Degradation pipeline for the AFS-VFM research project.

Usage
-----
>>> from degradation import generate_degradation_sequence
>>> frames = generate_degradation_sequence("image.jpg", "blur")
>>> print(frames.shape)   # (20, H, W, 3)

The module is designed for reuse: another developer can import
``generate_degradation_sequence`` and feed its output directly into a
model-inference pipeline.
"""

import numpy as np

from . import transformations
from .utils import load_image
from .scheduler import SeverityScheduler
from .compound import CompoundDegradation, COMPOUND_RECIPES


# ── Supported degradation type aliases ──────────────────────────────────────
_TYPE_ALIASES: dict[str, str] = {
    "blur": "motion_blur",
    "motion_blur": "motion_blur",
    "occlusion": "occlusion",
    "lighting": "lighting",
    "low_light": "lighting",
    "scale": "scale",
    "pixelate": "scale",
    "viewpoint": "viewpoint",
    "shear": "viewpoint",
    "gaussian_noise": "gaussian_noise",
    "noise": "gaussian_noise",
    "fog": "fog_haze",
    "fog_haze": "fog_haze",
    "haze": "fog_haze",
    "contrast": "contrast",
    "glare": "contrast",
}


class DegradationPipeline:
    """Generates a sequence of progressively degraded image frames.

    Parameters
    ----------
    num_frames : int, optional
        Number of degradation steps in each sequence (default 20).
    schedule : str, optional
        Progression schedule name: 'linear', 'exponential', 'logarithmic',
        'step_shock', or 'oscillatory' (default 'linear').
    schedule_params : dict, optional
        Parameters passed to the SeverityScheduler (e.g. gamma, shock_frame).
    transform_params : dict, optional
        Parameters passed to the underlying transform (e.g. occlusion mode, texture).
    """

    # Mapping from canonical degradation name → implementation function
    _TRANSFORMS: dict = {
        "motion_blur": transformations.motion_blur,
        "occlusion": transformations.occlusion,
        "lighting": transformations.lighting,
        "scale": transformations.scale,
        "viewpoint": transformations.viewpoint,
        "gaussian_noise": transformations.gaussian_noise,
        "fog_haze": transformations.fog_haze,
        "contrast": transformations.contrast,
    }

    def __init__(
        self,
        num_frames: int = 20,
        schedule: str = "linear",
        schedule_params: dict = None,
        transform_params: dict = None,
    ) -> None:
        self.num_frames = num_frames
        self.schedule = schedule
        self.schedule_params = schedule_params or {}
        self.transform_params = transform_params or {}
        self.last_severities: np.ndarray = None

    # ── Thin wrapper methods (for discoverability / IDE hints) ──────────

    @staticmethod
    def motion_blur(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply horizontal motion blur. See ``transformations.motion_blur``."""
        return transformations.motion_blur(image, severity)

    @staticmethod
    def occlusion(
        image: np.ndarray,
        severity: float,
        mode: str = "irregular",
        texture: str = "semi_transparent",
        alpha: float = 0.85,
        **kwargs,
    ) -> np.ndarray:
        """Apply realistic occlusion effect. See ``transformations.occlusion``."""
        return transformations.occlusion(
            image, severity, mode=mode, texture=texture, alpha=alpha, **kwargs
        )

    @staticmethod
    def lighting(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply low-light effect. See ``transformations.lighting``."""
        return transformations.lighting(image, severity)

    @staticmethod
    def scale(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply pixelation via downscaling. See ``transformations.scale``."""
        return transformations.scale(image, severity)

    @staticmethod
    def viewpoint(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply 3D perspective warp. See ``transformations.viewpoint``."""
        return transformations.viewpoint(image, severity)

    @staticmethod
    def gaussian_noise(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply Gaussian sensor noise. See ``transformations.gaussian_noise``."""
        return transformations.gaussian_noise(image, severity)

    @staticmethod
    def fog_haze(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply atmospheric fog/haze. See ``transformations.fog_haze``."""
        return transformations.fog_haze(image, severity)

    @staticmethod
    def contrast(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply dynamic range / contrast reduction. See ``transformations.contrast``."""
        return transformations.contrast(image, severity)

    @staticmethod
    def compound(
        image: np.ndarray,
        severity: float,
        components,
        **kwargs,
    ) -> np.ndarray:
        """Apply compound degradation. See ``CompoundDegradation.apply``."""
        return CompoundDegradation.apply(image, severity, components, **kwargs)

    # ── Core orchestration ──────────────────────────────────────────────

    def generate_sequence(
        self,
        image: np.ndarray,
        degradation_type,
        schedule: str = None,
        schedule_params: dict = None,
        transform_params: dict = None,
    ) -> np.ndarray:
        """Produce a sequence of *num_frames* degraded versions of *image*.

        Parameters
        ----------
        image : np.ndarray
            Source image in RGB uint8 format, shape (H, W, 3).
        degradation_type : str or sequence
            Name of degradation (e.g. ``"blur"``), compound recipe name
            (e.g. ``"night_motion"``), plus-separated chain (e.g. ``"lighting+blur"``),
            or a list of component dicts.
        schedule : str, optional
            Progression schedule to use. If None, uses ``self.schedule``.
        schedule_params : dict, optional
            Additional schedule parameters (e.g. gamma, shock_frame).
            If None, uses ``self.schedule_params``.
        transform_params : dict, optional
            Keyword parameters forwarded to the specific transform function.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_frames, H, W, 3)`` with dtype ``uint8``.

        Raises
        ------
        ValueError
            If *degradation_type* or *schedule* is not recognised.
        """
        eff_schedule = schedule if schedule is not None else self.schedule
        eff_params = schedule_params if schedule_params is not None else self.schedule_params
        eff_transform_params = transform_params if transform_params is not None else self.transform_params

        severities = SeverityScheduler.get_schedule(
            self.num_frames,
            schedule=eff_schedule,
            **eff_params
        )
        self.last_severities = severities

        h, w = image.shape[:2]
        frames = np.empty((self.num_frames, h, w, 3), dtype=np.uint8)

        # Check if degradation_type is a compound specification
        is_compound = False
        if isinstance(degradation_type, (list, tuple)):
            is_compound = True
        elif isinstance(degradation_type, str):
            deg_lower = degradation_type.lower().strip()
            if deg_lower in COMPOUND_RECIPES or "+" in deg_lower:
                is_compound = True

        if is_compound:
            for i in range(self.num_frames):
                frames[i] = CompoundDegradation.apply(
                    image,
                    float(severities[i]),
                    degradation_type,
                    **eff_transform_params
                )
            return frames

        # Single degradation route
        canonical = _TYPE_ALIASES.get(str(degradation_type).lower())
        if canonical is None:
            supported = sorted(set(list(_TYPE_ALIASES.keys()) + list(COMPOUND_RECIPES.keys())))
            raise ValueError(
                f"Unknown degradation type '{degradation_type}'. "
                f"Supported types: {supported}"
            )

        transform_fn = self._TRANSFORMS[canonical]

        for i in range(self.num_frames):
            sev = float(severities[i])
            if canonical == "occlusion" and eff_transform_params:
                frames[i] = transform_fn(image, sev, **eff_transform_params)
            else:
                frames[i] = transform_fn(image, sev)

        return frames


# ── Module-level convenience function ───────────────────────────────────────

def generate_degradation_sequence(
    image_path: str,
    degradation_type: str,
    num_frames: int = 20,
    schedule: str = "linear",
    schedule_params: dict = None,
    transform_params: dict = None,
) -> np.ndarray:
    """Load an image and return a degradation sequence.

    This is the main public API of the module.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    degradation_type : str
        Kind of degradation (e.g. ``"blur"``).
    num_frames : int, optional
        Number of degradation steps (default 20).
    schedule : str, optional
        Progression schedule name ('linear', 'exponential', 'logarithmic',
        'step_shock', or 'oscillatory'). Default is 'linear'.
    schedule_params : dict, optional
        Keyword parameters for the scheduler.
    transform_params : dict, optional
        Keyword parameters for the underlying transformation (e.g. mode, texture).

    Returns
    -------
    np.ndarray
        Array of shape ``(num_frames, H, W, 3)`` with dtype ``uint8``.
    """
    image = load_image(image_path)
    pipeline = DegradationPipeline(
        num_frames=num_frames,
        schedule=schedule,
        schedule_params=schedule_params,
        transform_params=transform_params,
    )
    return pipeline.generate_sequence(image, degradation_type)

