"""
AFS-VFM Gradient-Based Adversarial Perturbation Engine
======================================================

Implements active adversarial stress testing for Vision Foundation Models (VFMs).
Unlike passive spatial/photometric filters, adversarial perturbations are
deliberate mathematical perturbations that maximize latent feature disruption
within an imperceptible L-infinity epsilon budget:

    max_{||delta||_inf <= eps(s)}  Loss(f(x + delta), f(x))

Algorithms:
-----------
1. ``FGSM`` (Fast Gradient Sign Method): Single-step gradient attack.
2. ``PGD`` (Projected Gradient Descent): Multi-step iterative feature disruption.

Progression:
------------
Integrates with ``SeverityScheduler`` so that perturbation budget:
    eps(t) = s(t) * max_eps
scales smoothly across a frame sequence, enabling direct evaluation of
early-warning anticipatory signals against active adversarial evasion.
"""

from typing import Union, Optional, Tuple, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .scheduler import SeverityScheduler


class AdversarialAttacker:
    """Generates gradient-based adversarial perturbations targeting VFM feature spaces."""

    DEFAULT_MAX_EPS: float = 16.0 / 255.0  # ~0.0627 in [0, 1] scale

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    # ── Internal Feature Extraction ─────────────────────────────────────

    def _extract_feature_tensor(self, model, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract a 2D feature embedding tensor (batch_size, feature_dim) with autograd enabled."""
        # Detect model architecture type
        class_name = model.__class__.__name__.lower()

        if "clip" in class_name:
            if hasattr(model, "get_image_features"):
                feat = model.get_image_features(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)
                feat = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
        elif "detr" in class_name:
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
                feat = outputs.encoder_last_hidden_state.mean(dim=1)
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                feat = outputs.last_hidden_state.mean(dim=1)
            else:
                feat = outputs.logits.mean(dim=1)
        else:
            # DINOv2 / ViT
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                # Last layer CLS embedding
                feat = outputs.hidden_states[-1][:, 0, :]
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                feat = outputs.last_hidden_state[:, 0, :]
            else:
                feat = outputs.logits

        # Flatten spatial dims if needed
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return feat

    # ── Image Tensor Preparation ────────────────────────────────────────

    def _image_to_tensor(self, image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Convert input PIL or NumPy RGB image to (1, 3, H, W) float tensor in [0.0, 1.0]."""
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image, dtype=np.uint8)

        orig_shape = (arr.shape[0], arr.shape[1])
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(self.device), orig_shape

    def _tensor_to_image(self, tensor: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert (1, 3, H, W) float tensor in [0.0, 1.0] back to (H, W, 3) uint8 NumPy array."""
        arr = tensor.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)

        if target_shape is not None and (arr.shape[0], arr.shape[1]) != target_shape:
            pil = Image.fromarray(arr).resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            arr = np.array(pil)
        return arr

    # ── Single Frame Attack Methods ─────────────────────────────────────

    def fgsm_feature_attack(
        self,
        model,
        image: Union[np.ndarray, Image.Image],
        severity: float,
        max_eps: float = DEFAULT_MAX_EPS,
    ) -> np.ndarray:
        """Fast Gradient Sign Method (FGSM) targeting latent feature disruption.

        Maximizes: 1.0 - cos_sim(f(x), f(x + delta)) with single gradient step.
        """
        if severity <= 0.0:
            return np.asarray(image) if isinstance(image, np.ndarray) else np.array(image)

        eps = float(severity * max_eps)
        x_orig, orig_shape = self._image_to_tensor(image)

        # Freeze model parameters to conserve memory
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Clean reference feature embedding
        with torch.no_grad():
            clean_feat = self._extract_feature_tensor(model, x_orig).detach()
            clean_feat_norm = F.normalize(clean_feat, p=2, dim=-1)

        # Initialize perturbation
        delta = torch.zeros_like(x_orig, requires_grad=True)
        adv_x = torch.clamp(x_orig + delta, 0.0, 1.0)
        pert_feat = self._extract_feature_tensor(model, adv_x)
        pert_feat_norm = F.normalize(pert_feat, p=2, dim=-1)

        # Loss: maximize distance (minimize cosine similarity)
        cos_sim = (clean_feat_norm * pert_feat_norm).sum(dim=-1).mean()
        loss = 1.0 - cos_sim
        loss.backward()

        # FGSM step
        grad = delta.grad.data
        delta = eps * torch.sign(grad)
        perturbed_x = torch.clamp(x_orig + delta, 0.0, 1.0)

        return self._tensor_to_image(perturbed_x, orig_shape)

    def pgd_feature_attack(
        self,
        model,
        image: Union[np.ndarray, Image.Image],
        severity: float,
        steps: int = 7,
        max_eps: float = DEFAULT_MAX_EPS,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Projected Gradient Descent (PGD) multi-step feature disruption attack.

        Iteratively ascends the feature drift loss and projects back into
        the L-infinity epsilon ball: ||delta||_inf <= eps(s).
        """
        if severity <= 0.0:
            return np.asarray(image) if isinstance(image, np.ndarray) else np.array(image)

        eps = float(severity * max_eps)
        step_alpha = alpha if alpha is not None else (eps / max(1, steps // 2))

        x_orig, orig_shape = self._image_to_tensor(image)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # Compute clean target feature
        with torch.no_grad():
            clean_feat = self._extract_feature_tensor(model, x_orig).detach()
            clean_feat_norm = F.normalize(clean_feat, p=2, dim=-1)

        # Uniform random initialization within epsilon ball
        delta = torch.empty_like(x_orig).uniform_(-eps, eps)
        delta.data = torch.clamp(x_orig + delta.data, 0.0, 1.0) - x_orig

        for _ in range(steps):
            delta.requires_grad = True
            adv_x = x_orig + delta
            pert_feat = self._extract_feature_tensor(model, adv_x)
            pert_feat_norm = F.normalize(pert_feat, p=2, dim=-1)

            # Maximize distance between clean and perturbed representations
            loss = 1.0 - (clean_feat_norm * pert_feat_norm).sum(dim=-1).mean()
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.data
                delta.data = delta.data + step_alpha * torch.sign(grad)
                # Projection into L_inf epsilon ball
                delta.data = torch.clamp(delta.data, -eps, eps)
                # Ensure valid pixel range
                delta.data = torch.clamp(x_orig + delta.data, 0.0, 1.0) - x_orig

        perturbed_x = torch.clamp(x_orig + delta.data, 0.0, 1.0)
        return self._tensor_to_image(perturbed_x, orig_shape)

    # ── Sequence Generation Orchestration ───────────────────────────────

    def generate_adversarial_sequence(
        self,
        model,
        image: Union[np.ndarray, Image.Image],
        num_frames: int = 20,
        attack_type: str = "pgd",
        schedule: str = "linear",
        schedule_params: Optional[dict] = None,
        max_eps: float = DEFAULT_MAX_EPS,
        steps: int = 7,
    ) -> np.ndarray:
        """Generate a sequence of progressively perturbed adversarial frames.

        Parameters
        ----------
        model :
            Vision Foundation Model under test (DINOv2, CLIP, DETR).
        image :
            Base clean input image.
        num_frames : int
            Number of steps in the sequence (default 20).
        attack_type : str
            'pgd' (iterative, default) or 'fgsm' (single-step).
        schedule : str
            Severity progression ('linear', 'exponential', 'step_shock', etc.).
        schedule_params : dict, optional
            Parameters passed to SeverityScheduler.
        max_eps : float
            Maximum perturbation budget at severity 1.0 (default 16/255).
        steps : int
            Number of iterations per frame for PGD (default 7).

        Returns
        -------
        np.ndarray
            Array of shape (num_frames, H, W, 3) with dtype uint8.
        """
        sched_params = schedule_params or {}
        severities = SeverityScheduler.get_schedule(num_frames, schedule=schedule, **sched_params)

        if isinstance(image, Image.Image):
            base_arr = np.array(image.convert("RGB"))
        else:
            base_arr = np.asarray(image, dtype=np.uint8)

        h, w = base_arr.shape[:2]
        frames = np.empty((num_frames, h, w, 3), dtype=np.uint8)

        attack_fn = self.pgd_feature_attack if attack_type.lower() == "pgd" else self.fgsm_feature_attack

        for i in range(num_frames):
            sev = float(severities[i])
            if sev <= 0.0:
                frames[i] = base_arr.copy()
            else:
                if attack_type.lower() == "pgd":
                    frames[i] = self.pgd_feature_attack(model, base_arr, severity=sev, steps=steps, max_eps=max_eps)
                else:
                    frames[i] = self.fgsm_feature_attack(model, base_arr, severity=sev, max_eps=max_eps)

        return frames
