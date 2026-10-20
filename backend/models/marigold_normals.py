import time

import numpy as np
import torch
from diffusers import MarigoldNormalsPipeline
from PIL import Image

from .base import DEFAULT_FOV_DEG, Prediction, default_intrinsics, unproject

STEPS, ENSEMBLE = 4, 3  # ensemble > 2 is what makes the pipeline return an uncertainty map


class MarigoldNormals:
    """Dedicated surface-normal estimator. It has no depth head: points are a fronto-parallel plane
    at depth 1 so the run still fits the schema; the normals are the payload."""

    name = "marigold_normals"
    scale_kind = "relative"
    repo = "prs-eth/marigold-normals-v1-1"
    version = f"{repo}@steps{STEPS}-ens{ENSEMBLE}"

    def __init__(self, device: str):
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = MarigoldNormalsPipeline.from_pretrained(self.repo, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Prediction:
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        out = self.pipe(
            Image.fromarray(image),
            num_inference_steps=STEPS,
            ensemble_size=ENSEMBLE,
            output_uncertainty=True,
            match_input_resolution=True,
            output_type="np",
            generator=torch.Generator(device=self.device).manual_seed(0),
        )
        elapsed = time.perf_counter() - t0
        n_raw = out.prediction[0].astype(np.float32)  # (H, W, 3): X right, Y up, Z toward viewer
        n = n_raw * np.array([1, -1, -1], np.float32)  # -> OpenCV camera frame
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        unc = out.uncertainty[0].astype(np.float32).reshape(h, w)

        depth = np.ones((h, w), np.float32)
        K = default_intrinsics(h, w)
        return Prediction(
            points=unproject(depth, K),
            depth=depth,
            confidence=np.clip(1 - unc, 0, 1).astype(np.float32),
            mask=np.ones((h, w), bool),
            normals=n,
            intrinsics=K,
            scale_kind=self.scale_kind,
            meta={
                "model": self.repo,
                "license": "CreativeML Open RAIL++-M (weights); diffusers: Apache-2.0",
                "inference_s": round(elapsed, 4),
                "device": self.device,
                "input_resolution": [w, h],
                "steps": STEPS,
                "ensemble_size": ENSEMBLE,
                "intrinsics_source": f"assumed {DEFAULT_FOV_DEG:g} deg horizontal FOV",
                "depth_source": "none (no depth head); constant plane at depth 1",
                "normals_source": "Marigold normals, converted from (right, up, toward viewer) to OpenCV",
                "confidence_source": "1 - ensemble uncertainty",
            },
            raw={"normals_marigold_frame": n_raw, "uncertainty": unc},
        )
