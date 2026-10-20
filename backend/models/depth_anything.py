import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .base import DEFAULT_FOV_DEG, Prediction, default_intrinsics, unproject

# Small is Apache-2.0; Base and Large are CC-BY-NC-4.0.
LICENSES = {"Small": "Apache-2.0", "Base": "CC-BY-NC-4.0", "Large": "CC-BY-NC-4.0"}


class DepthAnythingV2:
    name = "depth_anything"
    scale_kind = "affine-invariant"
    size = os.environ.get("DEPTH_ANYTHING_SIZE", "Small")  # Small | Base | Large
    repo = f"depth-anything/Depth-Anything-V2-{size}-hf"
    version = repo

    def __init__(self, device: str):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.repo).to(device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Prediction:
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs).predicted_depth  # (1, h', w') relative disparity
        disp = F.interpolate(out[:, None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
        disp = disp.float().cpu().numpy()
        elapsed = time.perf_counter() - t0

        # The model predicts affine-invariant disparity. 1/disp is only depth up to an unknown
        # shift in disparity; pixels with ~zero disparity (sky, far background) are masked out.
        mask = disp > 1e-3 * max(float(disp.max()), 1e-6)
        depth = np.where(mask, 1.0 / np.maximum(disp, 1e-6), 0.0).astype(np.float32)
        K = default_intrinsics(h, w)
        return Prediction(
            points=unproject(depth, K),
            depth=depth,
            confidence=None,
            mask=mask,
            normals=None,
            intrinsics=K,
            scale_kind=self.scale_kind,
            meta={
                "model": self.repo,
                "revision": getattr(self.model.config, "_commit_hash", None),
                "license": LICENSES.get(self.size),
                "inference_s": round(elapsed, 4),
                "device": self.device,
                "input_resolution": [w, h],
                "model_resolution": list(inputs["pixel_values"].shape[-1:-3:-1]),
                "intrinsics_source": f"assumed {DEFAULT_FOV_DEG:g} deg horizontal FOV",
                "depth_source": "1 / predicted disparity",
            },
            raw={"disparity": disp.astype(np.float32)},
        )
