"""
AFS-VFM Non-Linear Severity Schedulers
======================================

Implements realistic, non-linear degradation progression schedules to model
real-world physical sensor degradation, sudden environmental disruptions,
and fluctuating atmospheric conditions.

Schedules:
----------
1. ``linear``:
   Standard baseline progression: s(t) = t / (N - 1).
2. ``exponential``:
   Accelerating failure regime modeling thermal throttling, frost accumulation,
   or progressive condensation build-up where failure accelerates non-linearly:
   s(t) = (exp(gamma * t / (N-1)) - 1) / (exp(gamma) - 1).
3. ``logarithmic``:
   Front-loaded abrupt onset that gradually saturates (e.g., sudden dawn/dusk glare):
   s(t) = ln(1 + beta * t / (N-1)) / ln(1 + beta).
4. ``step_shock``:
   Catastrophic instant disruption modeling sudden lens occlusion (mud/debris splatter),
   sensor impact, or instant optical path blockage:
   s(t) = pre_shock for t < shock_frame, post_shock for t >= shock_frame.
5. ``oscillatory``:
   Fluctuating degradation modeling windshield wiper cycles, street light flicker,
   or periodic shadows superimposed on background progression:
   s(t) = clip(base(t) + amplitude * sin(2*pi*f*t/(N-1) + phase), 0.0, 1.0).
"""

import numpy as np
from typing import Union, Optional, Sequence


class SeverityScheduler:
    """Computes progression curves for degradation severity across a frame sequence."""

    SUPPORTED_SCHEDULES = (
        "linear",
        "exponential",
        "logarithmic",
        "step_shock",
        "oscillatory",
    )

    @classmethod
    def get_schedule(
        cls,
        num_frames: int,
        schedule: str = "linear",
        **kwargs
    ) -> np.ndarray:
        """Compute a 1D array of severity values in [0.0, 1.0] for *num_frames*.

        Parameters
        ----------
        num_frames : int
            Number of frames in the sequence (must be >= 1).
        schedule : str, optional
            Schedule name: 'linear', 'exponential', 'logarithmic',
            'step_shock', or 'oscillatory'. Default is 'linear'.
        **kwargs :
            Additional parameter overrides for the chosen schedule:
            - exponential: ``gamma`` (float, default 3.0)
            - logarithmic: ``beta`` (float, default 10.0)
            - step_shock: ``shock_frame`` (int), ``pre_shock`` (float), ``post_shock`` (float)
            - oscillatory: ``amplitude`` (float), ``frequency`` (float), ``phase`` (float), ``base`` (str)

        Returns
        -------
        np.ndarray
            1D array of float64 values with shape ``(num_frames,)``,
            bounded strictly in [0.0, 1.0].
        """
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")

        if num_frames == 1:
            return np.array([0.0], dtype=np.float64)

        schedule_key = schedule.lower().strip()

        if schedule_key in ("linear", "lin"):
            return cls.linear(num_frames)
        elif schedule_key in ("exponential", "exp"):
            gamma = float(kwargs.get("gamma", 3.0))
            return cls.exponential(num_frames, gamma=gamma)
        elif schedule_key in ("logarithmic", "log"):
            beta = float(kwargs.get("beta", 10.0))
            return cls.logarithmic(num_frames, beta=beta)
        elif schedule_key in ("step_shock", "step", "shock"):
            shock_frame = kwargs.get("shock_frame", None)
            pre_shock = float(kwargs.get("pre_shock", 0.0))
            post_shock = float(kwargs.get("post_shock", 0.85))
            return cls.step_shock(
                num_frames,
                shock_frame=shock_frame,
                pre_shock=pre_shock,
                post_shock=post_shock,
            )
        elif schedule_key in ("oscillatory", "sinusoidal", "osc"):
            amplitude = float(kwargs.get("amplitude", 0.2))
            frequency = float(kwargs.get("frequency", 2.0))
            phase = float(kwargs.get("phase", 0.0))
            base = kwargs.get("base", "linear")
            return cls.oscillatory(
                num_frames,
                amplitude=amplitude,
                frequency=frequency,
                phase=phase,
                base=base,
            )
        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. Supported schedules: {cls.SUPPORTED_SCHEDULES}"
            )

    @staticmethod
    def linear(num_frames: int) -> np.ndarray:
        """Linear ramp from 0.0 to 1.0."""
        if num_frames == 1:
            return np.array([0.0], dtype=np.float64)
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float64)

    @staticmethod
    def exponential(num_frames: int, gamma: float = 3.0) -> np.ndarray:
        """Accelerating failure curve.

        s(t) = (exp(gamma * t) - 1) / (exp(gamma) - 1)
        where t in [0, 1].

        As gamma -> 0, curve approaches linear.
        For gamma > 0 (e.g. 3.0), curve stays mild initially and accelerates rapidly.
        """
        if num_frames == 1:
            return np.array([0.0], dtype=np.float64)
        if abs(gamma) < 1e-6:
            return np.linspace(0.0, 1.0, num_frames, dtype=np.float64)

        t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
        curve = (np.exp(gamma * t) - 1.0) / (np.exp(gamma) - 1.0)
        return np.clip(curve, 0.0, 1.0)

    @staticmethod
    def logarithmic(num_frames: int, beta: float = 10.0) -> np.ndarray:
        """Front-loaded / rapid onset curve with gradual saturation.

        s(t) = ln(1 + beta * t) / ln(1 + beta)
        where t in [0, 1].
        """
        if num_frames == 1:
            return np.array([0.0], dtype=np.float64)
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")

        t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
        curve = np.log1p(beta * t) / np.log1p(beta)
        return np.clip(curve, 0.0, 1.0)

    @staticmethod
    def step_shock(
        num_frames: int,
        shock_frame: Optional[int] = None,
        pre_shock: float = 0.0,
        post_shock: float = 0.85,
    ) -> np.ndarray:
        """Step function simulating sudden instantaneous catastrophe/occlusion.

        Parameters
        ----------
        num_frames : int
            Total number of frames.
        shock_frame : int, optional
            0-indexed frame where shock occurs. Defaults to num_frames // 2.
        pre_shock : float
            Severity prior to shock frame (default 0.0).
        post_shock : float
            Severity at and after shock frame (default 0.85).
        """
        if num_frames == 1:
            return np.array([post_shock], dtype=np.float64)
        if shock_frame is None:
            shock_frame = num_frames // 2

        curve = np.full(num_frames, pre_shock, dtype=np.float64)
        curve[shock_frame:] = post_shock
        return np.clip(curve, 0.0, 1.0)

    @staticmethod
    def oscillatory(
        num_frames: int,
        amplitude: float = 0.2,
        frequency: float = 2.0,
        phase: float = 0.0,
        base: Union[str, Sequence[float]] = "linear",
    ) -> np.ndarray:
        """Oscillatory progression with sinusoidal perturbations.

        Parameters
        ----------
        num_frames : int
            Total number of frames.
        amplitude : float
            Amplitude of oscillation (default 0.2).
        frequency : float
            Number of oscillation cycles across the sequence (default 2.0).
        phase : float
            Phase offset in radians (default 0.0).
        base : str or sequence of floats
            Base progression: 'linear', 'constant', or custom array.
        """
        if num_frames == 1:
            return np.array([0.0], dtype=np.float64)

        t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)

        if isinstance(base, str):
            if base == "linear":
                base_curve = t
            elif base == "constant":
                base_curve = np.full(num_frames, 0.5, dtype=np.float64)
            else:
                base_curve = t
        else:
            base_curve = np.asarray(base, dtype=np.float64)
            if len(base_curve) != num_frames:
                raise ValueError(
                    f"base array length ({len(base_curve)}) does not match num_frames ({num_frames})"
                )

        oscillation = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
        return np.clip(base_curve + oscillation, 0.0, 1.0)
