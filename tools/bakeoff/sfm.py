import sys, math, shutil, time
from pathlib import Path
import cv2, numpy as np, pycolmap
clip, out, fps, vfov = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
img, msk = out / "images", out / "masks"
shutil.rmtree(out, ignore_errors=True); img.mkdir(parents=True); msk.mkdir()
cap = cv2.VideoCapture(clip); src = cap.get(cv2.CAP_PROP_FPS); step = int(round(src / fps)); i = k = 0
W, H = 960, 540
while True:
    if not cap.grab(): break
    if i % step == 0:
        ok, f = cap.retrieve(); f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img / f"{k:04d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        m = np.full((H, W), 255, np.uint8); m[:26] = 0  # Roblox top bar
        cv2.imwrite(str(msk / f"{k:04d}.jpg.png"), m); k += 1
    i += 1
f = (H / 2) / math.tan(math.radians(vfov / 2))
print("frames", k, "focal", round(f, 1), flush=True)
db = out / "db.db"
ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = f"{f},{f},{W/2},{H/2}"; ro.mask_path = str(msk)
t = time.time()
pycolmap.extract_features(db, img, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
po = pycolmap.SequentialPairingOptions(); po.overlap = 12
pycolmap.match_sequential(db, pairing_options=po)
print("features+matching", round(time.time() - t), "s", flush=True)
for refine in (False, True):
    o = pycolmap.IncrementalPipelineOptions(); o.ba_refine_focal_length = refine; o.ba_refine_principal_point = False
    t = time.time()
    recs = pycolmap.incremental_mapping(db, img, out / f"sparse_refine{int(refine)}", o)
    for idx, r in recs.items():
        cam = next(iter(r.cameras.values()))
        print(f"refine={refine} model {idx}: {r.num_reg_images()}/{k} images, {r.num_points3D()} points, reproj {r.compute_mean_reprojection_error():.2f}px, focal {cam.params[0]:.1f} -> vFOV {math.degrees(2*math.atan(H/2/cam.params[0])):.1f}, {round(time.time()-t)} s", flush=True)
