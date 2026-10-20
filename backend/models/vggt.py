import time

import numpy as np
import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT as _VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .base import Prediction, unproject

MODEL_RES = 518  # VGGT's training width; height is rounded to a multiple of the 14 px patch


class VGGT:
    """VGGT on a single image (sequence length 1), using the depth head plus the camera head."""

    name = "vggt"
    scale_kind = "relative"
    repo = "facebook/VGGT-1B"
    version = repo

    def __init__(self, device: str):
        self.device = device
        self.model = _VGGT.from_pretrained(self.repo).to(device).eval()
        cuda = device == "cuda"
        self.dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if cuda else torch.float32

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Prediction:
        h, w = image.shape[:2]
        mh = max(14, round(MODEL_RES * h / w / 14) * 14)
        x = torch.from_numpy(image).permute(2, 0, 1).float().div(255)[None].to(self.device)
        # Plain resize (no crop/pad) so every model pixel maps back to an input pixel.
        x = F.interpolate(x, size=(mh, MODEL_RES), mode="bilinear", align_corners=False, antialias=True)
        t0 = time.perf_counter()
        with torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.model(x)  # adds the batch dim itself
        elapsed = time.perf_counter() - t0
        _, K = pose_encoding_to_extri_intri(out["pose_enc"].float(), (mh, MODEL_RES))

        depth_m = out["depth"][0, 0, ..., 0].float()[None, None]
        conf_m = out["depth_conf"][0, 0].float()[None, None]
        depth = F.interpolate(depth_m, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        conf_raw = F.interpolate(conf_m, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        K = K[0, 0].cpu().numpy().astype(np.float32)
        K[0] *= w / MODEL_RES  # model-res pixels -> input-res pixels
        K[1] *= h / mh
        mask = depth > 0
        return Prediction(
            points=unproject(depth.astype(np.float32), K),
            depth=depth.astype(np.float32),
            confidence=(1 - 1 / np.maximum(conf_raw, 1)).astype(np.float32),
            mask=mask,
            normals=None,
            intrinsics=K,
            scale_kind=self.scale_kind,
            meta={
                "model": self.repo,
                "license": "VGGT-1B weights: non-commercial (facebook/VGGT-1B model card); code: see repo",
                "inference_s": round(elapsed, 4),
                "device": self.device,
                "dtype": str(self.dtype),
                "input_resolution": [w, h],
                "model_resolution": [MODEL_RES, mh],
                "intrinsics_source": "camera head (pose_enc), rescaled to input resolution",
                "depth_source": "depth head, single-image sequence",
                "confidence_source": "1 - 1/depth_conf (raw depth_conf >= 1 kept in raw_depth_conf)",
            },
            raw={"depth_conf": conf_raw.astype(np.float32)},
        )
