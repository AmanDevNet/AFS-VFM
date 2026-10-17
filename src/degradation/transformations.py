"""
Deterministic image degradation transformations for AFS-VFM.

Each transformation accepts an RGB uint8 image and a severity value in
[0, 1], and returns a degraded RGB uint8 image of the *same* dimensions.

All transformations are fully deterministic — no randomness is used.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Motion Blur
# ---------------------------------------------------------------------------

def motion_blur(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply horizontal motion blur with increasing kernel length.

    Kernel length scales linearly from 1 px (severity 0, no blur) to
    51 px (severity 1).  At severity 0 the image is returned untouched.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1] controlling blur intensity.

    Returns:
        Blurred image with the same shape and dtype.
    """
    # At severity 0 → kernel_size 1 (identity), severity 1 → 51
    kernel_size = int(1 + severity * 50)  # 1 → 51
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, kernel_size)

    # kernel_size 1 means no blur at all — return original
    if kernel_size <= 1:
        return image.copy()

    # Horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size

    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


# ---------------------------------------------------------------------------
# 2. Occlusion
# ---------------------------------------------------------------------------

def occlusion(
    image: np.ndarray,
    severity: float,
    mode: str = "irregular",
    texture: str = "semi_transparent",
    alpha: float = 0.85,
    **kwargs,
) -> np.ndarray:
    """Apply realistic occlusion artifacts to an image.

    Supports both organic irregular polygon splatters (simulating mud, debris,
    water spots, or natural silhouettes) and classic rectangular boxes.
    Supports realistic photometric textures: semi-transparent dirty-lens smudges,
    sensor noise, or solid opaque fills.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 array of shape (H, W, 3).
    severity : float
        Float in [0, 1] controlling coverage and intensity. At 0, returns untouched image.
    mode : str, optional
        Occlusion geometry:
        - "irregular" / "splatter" / "blob": Organic randomized polygon splatters (default).
        - "boxes" / "rectangular": Axis-aligned rectangular patches (legacy mode).
    texture : str, optional
        Photometric appearance of occluded regions:
        - "semi_transparent" / "smudge": Translucent smudge with edge softening (default).
        - "noise": Textured Gaussian noise simulating sensor/surface grain.
        - "black" / "solid": Opaque zero-fill.
    alpha : float, optional
        Opacity of occlusion when texture is semi-transparent, in [0.0, 1.0]. Default 0.85.

    Returns
    -------
    np.ndarray
        Occluded RGB image with same shape and uint8 dtype.
    """
    if severity <= 0.0:
        return image.copy()

    h, w = image.shape[:2]
    # Deterministic pseudo-random generator tied strictly to severity
    seed = int(severity * 10000) + 42
    rs = np.random.RandomState(seed)

    mask = np.zeros((h, w), dtype=np.float32)
    mode_key = mode.lower().strip()

    if mode_key in ("boxes", "box", "rectangular"):
        # Legacy rectangular patches
        max_boxes = int(severity * 50)
        max_box_h = int(severity * (h * 0.3))
        max_box_w = int(severity * (w * 0.3))
        num_boxes = rs.randint(1, max(2, max_boxes + 1))

        for _ in range(num_boxes):
            box_h = rs.randint(10, max(11, max_box_h))
            box_w = rs.randint(10, max(11, max_box_w))
            y = rs.randint(0, max(1, h - box_h))
            x = rs.randint(0, max(1, w - box_w))
            mask[y:y + box_h, x:x + box_w] = 1.0
    else:
        # Default: Organic irregular splatters / blobs
        max_blobs = int(1 + severity * 25)
        num_blobs = rs.randint(1, max(2, max_blobs + 1))
        max_radius = int(max(15, min(h, w) * 0.25 * severity))

        for _ in range(num_blobs):
            cx = rs.randint(0, w)
            cy = rs.randint(0, h)
            base_r = rs.randint(8, max(9, max_radius))

            # Sample 8 to 14 irregular radial points
            num_vertices = rs.randint(8, 15)
            angles = np.linspace(0, 2.0 * np.pi, num_vertices, endpoint=False)
            angles += rs.uniform(-0.15, 0.15, size=num_vertices)

            radii = base_r * rs.uniform(0.5, 1.4, size=num_vertices)
            pts_x = np.clip(cx + radii * np.cos(angles), 0, w - 1).astype(np.int32)
            pts_y = np.clip(cy + radii * np.sin(angles), 0, h - 1).astype(np.int32)
            pts = np.stack([pts_x, pts_y], axis=1)

            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [pts], 255)
            mask = np.maximum(mask, poly_mask.astype(np.float32) / 255.0)

        # Soften boundaries for realistic organic edge dispersion
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

    mask = np.clip(mask, 0.0, 1.0)
    tex_key = texture.lower().strip()
    out = image.astype(np.float32)

    if tex_key in ("black", "solid", "opaque"):
        out[mask > 0.5] = 0.0
    elif tex_key in ("noise", "grain"):
        noise = rs.normal(100.0, 45.0, size=image.shape)
        noise = np.clip(noise, 0.0, 255.0)
        eff_mask = (mask * alpha)[:, :, None]
        out = (1.0 - eff_mask) * out + eff_mask * noise
    else:
        # "semi_transparent" / "smudge"
        blurred_bg = cv2.GaussianBlur(image, (21, 21), 0).astype(np.float32) * 0.25
        eff_mask = (mask * alpha)[:, :, None]
        out = (1.0 - eff_mask) * out + eff_mask * blurred_bg

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Lighting (Low-Light)
# ---------------------------------------------------------------------------

def lighting(image: np.ndarray, severity: float) -> np.ndarray:
    """Progressively plunge the image into darkness.
    
    At severity 0, normal brightness. At severity 1, extremely dark.
    """
    if severity <= 0.0:
        return image.copy()
        
    # Decrease brightness (beta: 0 to -150)
    # Decrease contrast slightly (alpha: 1.0 to 0.5)
    alpha = max(0.5, 1.0 - (severity * 0.5))
    beta = int(-150 * severity)
    
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return out


# ---------------------------------------------------------------------------
# 4. Scale (Pixelation / Compression)
# ---------------------------------------------------------------------------

def scale(image: np.ndarray, severity: float) -> np.ndarray:
    """Crush spatial resolution and repeatedly upsample it back.
    
    At severity 0, original resolution. At severity 1, heavily pixelated.
    """
    if severity <= 0.0:
        return image.copy()
        
    h, w = image.shape[:2]
    
    # Scale down factor: 1.0 (no scale) to ~0.02 (massive scale down)
    scale_factor = max(0.02, 1.0 - (severity * 0.98))
    
    small_w = max(1, int(w * scale_factor))
    small_h = max(1, int(h * scale_factor))
    
    # Downsample
    temp = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    # Upsample back to original size (using nearest neighbor for blocky artifacts)
    out = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


# ---------------------------------------------------------------------------
# 5. Viewpoint (Perspective Shear)
# ---------------------------------------------------------------------------

def viewpoint(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply an extreme perspective warp (shearing the image back).
    """
    if severity <= 0.0:
        return image.copy()
        
    h, w = image.shape[:2]
    
    # Source points: the 4 corners
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Destination points: pinch the top corners inwards dynamically
    pinch_x = int(severity * (w * 0.4))
    pinch_y = int(severity * (h * 0.2))
    
    pts2 = np.float32([
        [pinch_x, pinch_y],          # Top-left gets pushed right and down
        [w - pinch_x, pinch_y],      # Top-right gets pushed left and down
        [0, h],                      # Bottom-left stays
        [w, h]                       # Bottom-right stays
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    out = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return out


# ---------------------------------------------------------------------------
# 6. Gaussian Noise (Sensor Electronics Noise)
# ---------------------------------------------------------------------------

def gaussian_noise(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply additive Gaussian sensor noise modeling digital camera ISO grain.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1] controlling noise standard deviation.

    Returns:
        Noisy RGB uint8 image of the same shape.
    """
    if severity <= 0.0:
        return image.copy()
    seed = int(severity * 10000) + 77
    rs = np.random.RandomState(seed)
    sigma = severity * 60.0
    noise = rs.normal(0.0, sigma, size=image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 7. Fog / Atmospheric Haze (Koschmieder Optical Scattering)
# ---------------------------------------------------------------------------

def fog_haze(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply atmospheric scattering/haze based on Koschmieder's optical model.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1] controlling atmospheric scattering density.

    Returns:
        Hazy/foggy RGB uint8 image of the same shape.
    """
    if severity <= 0.0:
        return image.copy()
    h, w = image.shape[:2]
    # Atmospheric light vector (soft bright white/gray veil)
    atmospheric_light = 240.0
    # Depth-correlated optical transmission: vertical distance gradient
    y_coords = np.linspace(0.4, 1.0, h, dtype=np.float32)[:, None, None]
    beta = severity * 1.8
    transmission = np.exp(-beta * y_coords)
    transmission = np.clip(transmission, 0.05, 1.0)
    hazy = image.astype(np.float32) * transmission + atmospheric_light * (1.0 - transmission)
    return np.clip(hazy, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 8. Contrast (Dynamic Range Compression / Glare)
# ---------------------------------------------------------------------------

def contrast(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply dynamic range / contrast attenuation (modeling optical glare or sensor wash-out).

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1] controlling contrast reduction.

    Returns:
        Low-contrast RGB uint8 image of the same shape.
    """
    if severity <= 0.0:
        return image.copy()
    mean_val = float(np.mean(image))
    factor = max(0.05, 1.0 - (severity * 0.90))
    out = mean_val + factor * (image.astype(np.float32) - mean_val)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)

