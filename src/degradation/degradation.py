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
    """

    # Mapping from canonical degradation name → implementation function
    _TRANSFORMS: dict = {
        "motion_blur": transformations.motion_blur,
        "occlusion": transformations.occlusion,
        "lighting": transformations.lighting,
        "scale": transformations.scale,
        "viewpoint": transformations.viewpoint,
    }

    def __init__(
        self,
        num_frames: int = 20,
        schedule: str = "linear",
        schedule_params: dict = None,
    ) -> None:
        self.num_frames = num_frames
        self.schedule = schedule
        self.schedule_params = schedule_params or {}
        self.last_severities: np.ndarray = None

    # ── Thin wrapper methods (for discoverability / IDE hints) ──────────

    @staticmethod
    def motion_blur(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply horizontal motion blur. See ``transformations.motion_blur``."""
        return transformations.motion_blur(image, severity)

    @staticmethod
    def occlusion(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply random black box occlusion. See ``transformations.occlusion``."""
        return transformations.occlusion(image, severity)

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

    # ── Core orchestration ──────────────────────────────────────────────

    def generate_sequence(
        self,
        image: np.ndarray,
        degradation_type: str,
        schedule: str = None,
        schedule_params: dict = None,
    ) -> np.ndarray:
        """Produce a sequence of *num_frames* degraded versions of *image*.

        Parameters
        ----------
        image : np.ndarray
            Source image in RGB uint8 format, shape (H, W, 3).
        degradation_type : str
            Name of the degradation to apply. Accepted values include
            ``"blur"``, ``"motion_blur"``, ``"occlusion"``, etc.
        schedule : str, optional
            Progression schedule to use. If None, uses ``self.schedule``.
        schedule_params : dict, optional
            Additional schedule parameters (e.g. gamma, shock_frame).
            If None, uses ``self.schedule_params``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_frames, H, W, 3)`` with dtype ``uint8``.

        Raises
        ------
        ValueError
            If *degradation_type* or *schedule* is not recognised.
        """
        canonical = _TYPE_ALIASES.get(degradation_type.lower())
        if canonical is None:
            supported = sorted(set(_TYPE_ALIASES.keys()))
            raise ValueError(
                f"Unknown degradation type '{degradation_type}'. "
                f"Supported types: {supported}"
            )

        eff_schedule = schedule if schedule is not None else self.schedule
        eff_params = schedule_params if schedule_params is not None else self.schedule_params

        severities = SeverityScheduler.get_schedule(
            self.num_frames,
            schedule=eff_schedule,
            **eff_params
        )
        self.last_severities = severities

        transform_fn = self._TRANSFORMS[canonical]
        h, w = image.shape[:2]
        frames = np.empty((self.num_frames, h, w, 3), dtype=np.uint8)

        for i in range(self.num_frames):
            frames[i] = transform_fn(image, float(severities[i]))

        return frames


# ── Module-level convenience function ───────────────────────────────────────

def generate_degradation_sequence(
    image_path: str,
    degradation_type: str,
    num_frames: int = 20,
    schedule: str = "linear",
    schedule_params: dict = None,
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
    )
    return pipeline.generate_sequence(image, degradation_type)

