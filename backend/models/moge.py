import time

import numpy as np
import torch
from moge.model.v3 import MoGeModel

from ..device import DEVICE
from .base import Prediction


def refine_steps(device: str) -> int:
    # The sparse volumetric refinement runs Triton kernels (flex-gemm), which need CUDA.
    return 3 if device == "cuda" else 0


class MoGe:
    """MoGe-3, the newest release: metric pointmap, intrinsics, mask and normals in one pass."""

    name = "moge"
    scale_kind = "metric"
    repo = "Ruicheng/moge-3-vitl"
    version = f"{repo}@refine{refine_steps(DEVICE)}"

    def __init__(self, device: str):
        self.device = device
        self.model = MoGeModel.from_pretrained(self.repo).to(device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Prediction:
        h, w = image.shape[:2]
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float().div(255)
        t0 = time.perf_counter()
        steps = refine_steps(self.device)
        out = self.model.infer(x, refine_steps=steps)
        elapsed = time.perf_counter() - t0
        get = lambda k: out[k].float().cpu().numpy() if k in out and out[k] is not None else None

        K = get("intrinsics").astype(np.float32)  # normalized: fx/W, cx/W, fy/H, cy/H
        K[0] *= w
        K[1] *= h
        mask = get("mask").astype(bool)
        points = np.nan_to_num(get("points"), nan=0, posinf=0, neginf=0).astype(np.float32)
        depth = np.nan_to_num(get("depth"), nan=0, posinf=0, neginf=0).astype(np.float32)
        normals = get("normal")
        if normals is not None:
            normals = normals.astype(np.float32)
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
        return Prediction(
            points=points,
            depth=depth,
            confidence=None,
            mask=mask & np.isfinite(depth) & (depth > 0),
            normals=normals,
            intrinsics=K,
            scale_kind=self.scale_kind,
            meta={
                "model": self.repo,
                "license": "MIT (code and weights per repo/model card; DINOv2 parts Apache-2.0)",
                "inference_s": round(elapsed, 4),
                "device": self.device,
                "input_resolution": [w, h],
                "refine_steps": steps,
                "intrinsics_source": "predicted (MoGe camera estimate)",
                "depth_source": "predicted metric pointmap z",
                "normals_source": "MoGe normal head" if normals is not None else "none",
            },
        )
