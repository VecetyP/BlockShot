import dataclasses
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .analysis.discontinuity import discontinuity
from .models import registry
from .models.base import Prediction

RUNS = Path(__file__).resolve().parent.parent / "runs"


def decode_image(data: bytes) -> np.ndarray:
    """RGB uint8. Transparent pixels are composited onto white (illusion drawings often ship with alpha)."""
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


def run_id(image_sha: str, model: str, version: str) -> str:
    return f"{image_sha[:16]}_{model}_{hashlib.sha256(version.encode()).hexdigest()[:8]}"


def predict_cached(data: bytes, model: str, force: bool = False, source: str | None = None) -> tuple[str, dict, bool]:
    """Returns (run_id, meta, cached)."""
    sha = hashlib.sha256(data).hexdigest()
    version = registry.cls(model).version
    rid = run_id(sha, model, version)
    d = RUNS / rid
    if not force and (d / "meta.json").exists():
        return rid, load_meta(rid), True
    image = decode_image(data)
    pred = registry.get(model).predict(image)
    meta = save(d, rid, image, pred, model=model, version=version, image_sha256=sha, source=source)
    return rid, meta, False


def save(d: Path, rid: str, image: np.ndarray, pred: Prediction, **info) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in dataclasses.asdict(pred).items() if isinstance(v, np.ndarray)}
    arrays |= {f"raw_{k}": v for k, v in pred.raw.items()}
    np.savez_compressed(d / "prediction.npz", **arrays)
    Image.fromarray(image).save(d / "input.png")
    write_ply(d / "points.ply", pred.points[pred.mask], image[pred.mask])
    meta = {
        "id": rid,
        **info,
        "scale_kind": pred.scale_kind,
        "height": image.shape[0],
        "width": image.shape[1],
        "intrinsics": pred.intrinsics.tolist(),
        "has_confidence": pred.confidence is not None,
        "has_normals": pred.normals is not None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **pred.meta,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def load_meta(rid: str) -> dict:
    return json.loads((RUNS / rid / "meta.json").read_text())


def load_arrays(rid: str) -> np.lib.npyio.NpzFile:
    return np.load(RUNS / rid / "prediction.npz")


def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    metas = [json.loads(p.read_text()) for p in RUNS.glob("*/meta.json")]
    return sorted(metas, key=lambda m: m["created"], reverse=True)


MAX_POINTS = 500_000


def map_array(rid: str, kind: str) -> np.ndarray | None:
    a = load_arrays(rid)
    if kind == "discontinuity":
        return discontinuity(a["points"], a["mask"])
    return a[kind] if kind in a else None


def points_bin(rid: str, stride: int | None = None) -> bytes:
    """Little-endian: uint32 N, uint32 has_confidence, float32 xyz[N*3], float32 discontinuity[N],
    float32 confidence[N] (only if has_confidence), uint8 rgb[N*3].
    Floats come first so every block stays 4-byte aligned. Pixel-grid subsampling keeps structure."""
    a = load_arrays(rid)
    pts, mask = a["points"], a["mask"]
    rgb = np.asarray(Image.open(RUNS / rid / "input.png").convert("RGB"))
    min_stride = max(1, int(np.ceil(np.sqrt(mask.sum() / MAX_POINTS))))
    stride = max(stride or 1, min_stride)
    sl = (slice(None, None, stride), slice(None, None, stride))
    m = mask[sl]
    blocks = [pts[sl][m].astype("<f4"), discontinuity(pts, mask)[sl][m].astype("<f4")]
    if "confidence" in a:
        blocks.append(a["confidence"][sl][m].astype("<f4"))
    blocks.append(rgb[sl][m].astype(np.uint8))
    header = np.array([m.sum(), "confidence" in a], "<u4")
    return header.tobytes() + b"".join(x.tobytes() for x in blocks)
