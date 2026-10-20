from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

DEFAULT_FOV_DEG = 60.0


@dataclass
class Prediction:
    points: np.ndarray  # (H, W, 3) float32, camera frame (OpenCV: +X right, +Y down, +Z forward)
    depth: np.ndarray  # (H, W) float32
    confidence: np.ndarray | None  # (H, W) float32 in [0, 1]
    mask: np.ndarray  # (H, W) bool, valid pixels
    normals: np.ndarray | None  # (H, W, 3) float32, camera frame, unit length
    intrinsics: np.ndarray  # (3, 3) float32
    scale_kind: Literal["metric", "relative", "affine-invariant"]
    meta: dict
    raw: dict[str, np.ndarray] = field(default_factory=dict)  # untouched model outputs, e.g. disparity


class Predictor(Protocol):
    name: str
    version: str  # part of the cache key; change it when weights or preprocessing change
    scale_kind: str

    def predict(self, image: np.ndarray) -> Prediction: ...


def default_intrinsics(h: int, w: int, fov_deg: float = DEFAULT_FOV_DEG) -> np.ndarray:
    """Pinhole intrinsics from a horizontal FOV. Pixel centers sit at integer + 0.5."""
    f = (w / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1).astype(np.float32)
