"""Run one model over the bake-off windows and score it.

    # baseline, in the app's venv (uses the app's own VGGT code):
    uv run python tools/bakeoff/run.py vggt "<clip.mp4>"
    # contenders, in the bake-off venv:
    tools/bakeoff/.venv/Scripts/python.exe tools/bakeoff/run.py pi3x "<clip.mp4>"
    tools/bakeoff/.venv/Scripts/python.exe tools/bakeoff/run.py mapanything "<clip.mp4>"

Each window is scored as is and with the player's arms blanked; results go to out/<model>.json and
renders to out/renders/. Nothing in the app or runs/ is touched.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import common

MIN_FREE_GB = 9.0  # leave the GPU to the user's BlockShot server if it holds the memory


def _resized(frames, size):
    return [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]


def _fit_intrinsics(local: np.ndarray) -> np.ndarray:
    """Pinhole K from a per-camera point map (H, W, 3): u = fx x/z + cx, v = fy y/z + cy."""
    h, w = local.shape[:2]
    v, u = np.mgrid[0:h, 0:w] + 0.5
    z = local[..., 2]
    ok = z > 1e-6
    a, b = local[..., 0][ok] / z[ok], local[..., 1][ok] / z[ok]
    fx, cx = np.linalg.lstsq(np.c_[a, np.ones_like(a)], u[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[b, np.ones_like(b)], v[ok], rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


class VGGT:
    """The app's pipeline: backend.video.vggt_poses (VGGT-1B, frames at 518 on the longer side)."""

    def __init__(self):
        sys.path.insert(0, str(common.HERE.parent.parent))
        from backend import video
        from backend.models import registry
        self.video, self.registry = video, registry

    def __call__(self, frames, paths=None):
        vg, _ = self.video.vggt_poses(frames)
        self.registry.free_cache()
        return {"depth": vg["depth"], "K": vg["K"], "c2w": vg["c2w"], "conf": vg["conf"], "rgb": np.stack(frames)}


class Pi3X:
    """Pi3X (yyfz233/Pi3X): permutation-equivariant, no reference frame, built with dynamic scenes in mind."""

    size = (518, 294)  # 640x360 at 518 wide, height a multiple of the 14 px patch

    def __init__(self):
        from pi3.models.pi3x import Pi3X as Model
        self.model = Model.from_pretrained("yyfz233/Pi3X").to("cuda").eval()
        # float16, not bfloat16: Pi3's attention forces FlashAttention for bfloat16, which Windows
        # PyTorch builds don't ship ("No available kernel"); float16 uses the standard kernels.
        self.dtype = torch.float16

    @torch.inference_mode()
    def __call__(self, frames, paths=None):
        small = _resized(frames, self.size)
        x = torch.from_numpy(np.stack(small)).permute(0, 3, 1, 2).float().div(255)[None].to("cuda")
        with torch.autocast("cuda", dtype=self.dtype):
            r = self.model(imgs=x)
        local = r["local_points"][0].float().cpu().numpy()
        out = {
            "depth": local[..., 2].astype(np.float32),
            "K": np.stack([_fit_intrinsics(p) for p in local]),
            "c2w": r["camera_poses"][0].float().cpu().numpy().astype(np.float64),
            "conf": torch.sigmoid(r["conf"][0, ..., 0].float()).cpu().numpy(),
            "rgb": np.stack(small),
        }
        torch.cuda.empty_cache()
        return out


class MapAnything:
    """MapAnything (facebook/map-anything-apache, Apache 2.0), images only."""

    def __init__(self):
        from mapanything.models import MapAnything as Model
        from mapanything.utils.image import load_images
        self.model = Model.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
        self.load_images = load_images

    def __call__(self, frames, paths=None):
        if paths is None:  # e.g. blanked frames: MapAnything loads from files
            d = common.OUT / "tmp_mapanything"
            d.mkdir(parents=True, exist_ok=True)
            paths = [d / f"{i:03d}.png" for i in range(len(frames))]
            for p, f in zip(paths, frames):
                cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        views = self.load_images([str(p) for p in paths])
        preds = self.model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                 apply_mask=False, mask_edges=False)
        g = lambda k: np.stack([p[k][0].float().cpu().numpy() for p in preds])
        out = {
            "depth": g("depth_z")[..., 0].astype(np.float32),
            "K": g("intrinsics").astype(np.float64),
            "c2w": g("camera_poses").astype(np.float64),
            "conf": g("conf"),
            "rgb": (np.clip(g("img_no_norm"), 0, 1) * 255).astype(np.uint8),
        }
        torch.cuda.empty_cache()
        return out


MODELS = {"vggt": VGGT, "pi3x": Pi3X, "mapanything": MapAnything}


def main(model_name: str, clip: str, only: str | None = None) -> None:
    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < MIN_FREE_GB:
        sys.exit(f"Only {free:.1f} GB of GPU memory is free; stop the BlockShot server (close its windows) and run again.")
    model = MODELS[model_name]()
    results = {}
    (common.OUT / "renders").mkdir(parents=True, exist_ok=True)
    for name, t0, t1, n in common.WINDOWS:
        if only and only not in name:
            continue
        frames, paths = common.load_window(clip, name, t0, t1, n)
        entry = {}
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            pred = model(frames, paths)
            entry["seconds"] = round(time.perf_counter() - t, 1)
            entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            entry["as_is"] = common.score(pred)
            cv2.imwrite(str(common.OUT / "renders" / f"{model_name}_{common.slug(name)}.png"),
                        cv2.cvtColor(common.render(pred, pred["rgb"][0]), cv2.COLOR_RGB2BGR))
            arms = common.near_mask(pred["depth"])
            entry["arms_share"] = round(float(arms.mean()), 3)
            entry["arms_blanked"] = common.score(model(common.blank(frames, arms)), arms)
        except torch.cuda.OutOfMemoryError:
            entry["error"] = "out of GPU memory"
            torch.cuda.empty_cache()
        results[name] = entry
        print(name, entry, flush=True)
        common.save_result(model_name + (f"_{common.slug(only)}" if only else ""), results)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
