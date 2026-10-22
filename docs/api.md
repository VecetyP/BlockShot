# API

[← README](../README.md)

The backend is a FastAPI app on :8000 (`backend/app.py`); the web app reaches it through Vite's
`/api` proxy. Anything slow runs as a background job: the route returns a job id, and
`GET /jobs/{id}` reports its stage, progress and result.

| Route | |
|---|---|
| `GET /health` | `{stale}`: whether the code on disk changed after the server started |
| `GET /models` · `POST /models/unload?name=` | models, load status, device · free GPU memory |
| `GET /jobs/{id}` | a background job's stage, progress and result |
| `POST /predict` | multipart `image`, `models` (repeatable), `force`, optional `truth` → job |
| `POST /video` | multipart `video`, `fps`, `max_frames`, `max_side`, `hud`, `geometry`, optional `fov_deg` and `truth` → job |
| `GET /runs` · `GET /runs/{id}` · `DELETE /runs/{id}` | run metadata · delete a run |
| `POST /runs/{id}/recompute` | redo a run with the current version → job with its (possibly new) id |
| `GET /runs/{id}/score` | a playground run's scores against its ground truth |
| `GET /runs/{id}/frames/{i}.jpg` | a scene's sampled frames |
| `POST /runs/{id}/minecraft/build` | build the blocks → job with the summary |
| `GET /runs/{id}/minecraft.json` | build summary: size, block count, tiles, FOV, pitch, paste offset |
| `GET /runs/{id}/minecraft.schem` | Sponge v2 schematic (builds up to one tile) |
| `POST /runs/{id}/minecraft/tiles` | write the tiles zip → job with its URL |
| `GET /runs/{id}/minecraft.bin` | block preview for the viewer |
| `GET /runs/{id}/points.bin?stride=` | point cloud for the viewer (≤ 500k points) |
| `GET /runs/{id}/maps/{kind}.png` | depth, confidence, normals, discontinuity, curl, residual |
| `GET /runs/{id}/analysis` · `GET /runs/{id}/field/{curl,residual}.bin` | integrability, reprojection, tears · per-point fields |
| `POST /runs/{id}/offaxis` | ±5/15/30° renders → job |

Minecraft routes take `mode` (`auto` · `meters` · `fit`), `value` and, for scenes, `frame`.
