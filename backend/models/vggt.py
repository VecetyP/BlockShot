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
        return self._prediction(depth, conf_raw, mask, K, elapsed, w, h, mh)

    @torch.inference_mode()
    def predict_frames(self, frames: list[np.ndarray]) -> dict:
        """Several frames of one scene in a single pass, so poses and depth share one frame of
        reference: the first frame's camera. All frames must have the same size. Returns, at frame
        resolution: extrinsics (S, 3, 4) camera-from-world, intrinsics (S, 3, 3), depth (S, H, W)
        and depth_conf (S, H, W) (>= 1)."""
        h, w = frames[0].shape[:2]
        mh = max(14, round(MODEL_RES * h / w / 14) * 14)
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255).to(self.device)
        x = F.interpolate(x, size=(mh, MODEL_RES), mode="bilinear", align_corners=False, antialias=True)
        with torch.autocast(self.device, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.model(x)
        E, K = pose_encoding_to_extri_intri(out["pose_enc"].float(), (mh, MODEL_RES))
        size = dict(size=(h, w), mode="bilinear", align_corners=False)
        depth = F.interpolate(out["depth"][0, ..., 0].float()[:, None], **size)[:, 0]
        conf = F.interpolate(out["depth_conf"][0].float()[:, None], **size)[:, 0]
        K = K[0].cpu().numpy().astype(np.float64)
        K[:, 0] *= w / MODEL_RES  # model-res pixels -> frame pixels
        K[:, 1] *= h / mh
        return {
            "extrinsics": E[0].cpu().numpy().astype(np.float64),
            "intrinsics": K,
            "depth": depth.cpu().numpy().astype(np.float32),
            "depth_conf": conf.cpu().numpy().astype(np.float32),
        }

    def _prediction(self, depth, conf_raw, mask, K, elapsed, w, h, mh) -> Prediction:
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
