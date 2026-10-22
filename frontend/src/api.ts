export const API = "/api";

export type ModelInfo = { name: string; loaded: boolean; device: string; scale_kind: string; version: string };

export type RunMeta = {
  id: string;
  model: string;
  source: string | null;
  scale_kind: string;
  width: number;
  height: number;
  intrinsics: number[][];
  has_confidence: boolean;
  created: string;
  kind?: "scene"; // a video scene: points fused from many frames, in frame 0's camera coordinates
  frames?: number;
  cameras?: { c2w: number[][]; K: number[][] }[]; // per frame, OpenCV axes
  [k: string]: unknown;
};

export const isScene = (r: RunMeta) => r.kind === "scene";

export type Cloud = {
  n: number;
  xyz: Float32Array; // camera frame, OpenCV axes (+X right, +Y down, +Z forward)
  discontinuity: Float32Array;
  confidence: Float32Array | null;
  rgb: Uint8Array;
};

const json = async <T,>(r: Response): Promise<T> => {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};

export const getModels = () => fetch(`${API}/models`).then(json<ModelInfo[]>);
export const getRuns = () => fetch(`${API}/runs`).then(json<RunMeta[]>);
export const getRun = (id: string) => fetch(`${API}/runs/${id}`).then(json<RunMeta>);
export const deleteRun = (id: string) => fetch(`${API}/runs/${id}`, { method: "DELETE" }).then(json<unknown>);
/** Redoes a run with the current code; resolves with its id (scenes get a new one). */
export const recomputeRun = (id: string, onProgress?: OnProgress) =>
  startJob(`${API}/runs/${id}/recompute`).then((job) => waitJob<string>(job, onProgress));

/** A background job on the server (backend/jobs.py). total 0 = no measurable progress in this stage. */
export type JobState = {
  id: string;
  kind: string;
  state: "running" | "done" | "error";
  stage: string;
  done: number;
  total: number;
  message: string;
  result: unknown;
  error: string | null;
  elapsed: number;
};
export type OnProgress = (job: JobState) => void;

const startJob = (url: string, init?: RequestInit) =>
  fetch(url, { method: "POST", ...init }).then(json<{ job_id: string }>).then((j) => j.job_id);

/** Polls a job until it finishes; resolves with its result or rejects with its error. Aborting
 * `signal` stops polling and cancels the job on the server (e.g. a build nobody waits for). */
export async function waitJob<T>(jobId: string, onProgress?: OnProgress, signal?: AbortSignal): Promise<T> {
  const cancel = () => void fetch(`${API}/jobs/${jobId}/cancel`, { method: "POST" }).catch(() => {});
  signal?.addEventListener("abort", cancel);
  let failures = 0;
  try {
    for (;;) {
      signal?.throwIfAborted();
      let job: JobState;
      try {
        job = await fetch(`${API}/jobs/${jobId}`, { signal }).then(json<JobState>);
        failures = 0;
      } catch (e) {
        if (signal?.aborted || ++failures > 5) throw e; // ride out a hiccup in the dev proxy
        await new Promise((r) => setTimeout(r, 1000));
        continue;
      }
      onProgress?.(job);
      if (job.state === "done") return job.result as T;
      if (job.state === "error") throw new Error(job.error ?? "job failed");
      await new Promise((r) => setTimeout(r, 400));
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
}

export const predict = (file: File, models: string[], onProgress?: OnProgress, force = false) => {
  const fd = new FormData();
  fd.append("image", file);
  models.forEach((m) => fd.append("models", m));
  fd.append("force", String(force));
  return startJob(`${API}/predict`, { body: fd }).then((id) =>
    waitJob<{ id: string; cached: boolean; meta: RunMeta }[]>(id, onProgress),
  );
};

// Layout documented in backend/store.py:points_bin.
export async function getCloud(id: string): Promise<Cloud> {
  const buf = await (await fetch(`${API}/runs/${id}/points.bin`)).arrayBuffer();
  const [n, hasConf] = new Uint32Array(buf, 0, 2);
  let off = 8;
  const take = (len: number) => {
    const a = new Float32Array(buf, off, len);
    off += len * 4;
    return a;
  };
  const xyz = take(n * 3);
  const discontinuity = take(n);
  const confidence = hasConf ? take(n) : null;
  return { n, xyz, discontinuity, confidence, rgb: new Uint8Array(buf, off, n * 3) };
}

export type VideoSettings = { fps: number; max_frames: number; hud: boolean; geometry: "moge" | "vggt" };

/** Reconstructs one scene from a video (cached by video + settings). */
export const processVideo = (file: File, s: VideoSettings, onProgress?: OnProgress) => {
  const fd = new FormData();
  fd.append("video", file);
  Object.entries(s).forEach(([k, v]) => fd.append(k, String(v)));
  return startJob(`${API}/video`, { body: fd }).then((id) => waitJob<{ id: string; cached: boolean; meta: RunMeta }>(id, onProgress));
};

export const frameUrl = (id: string, i: number) => `${API}/runs/${id}/frames/${i}.jpg`;
export const mapUrl = (id: string, kind: string) => `${API}/runs/${id}/maps/${kind}.png`;
export const inputUrl = (id: string) => `${API}/runs/${id}/input.png`;

export type Tear = { u: number; v: number; score: number; xyz: [number, number, number] };
export type Analysis = {
  reprojection_px: number | null;
  integrability: Partial<Record<"native" | "derived", { residual_fraction: number; edges: number }>>;
  tears: Tear[];
};

// Panels and strips both ask for the same run's analysis; share one request.
const analyses = new Map<string, Promise<Analysis>>();
export const getAnalysis = (id: string) => {
  if (!analyses.has(id)) analyses.set(id, fetch(`${API}/runs/${id}/analysis`).then(json<Analysis>));
  return analyses.get(id)!;
};

export const getField = async (id: string, kind: string) => {
  const r = await fetch(`${API}/runs/${id}/field/${kind}.bin`);
  return r.ok ? new Float32Array(await r.arrayBuffer()) : null;
};

export const renderOffAxis = (id: string, onProgress?: OnProgress) =>
  startJob(`${API}/runs/${id}/offaxis`)
    .then((job) => waitJob<string[]>(job, onProgress))
    .then((u) => u.map((x) => API + x));

/** How blocks are sized. auto: about `value` thousand blocks; meters: one block = value m; fit: longest side. */
export type BuildMode = "auto" | "meters" | "fit";
export type BuildSettings = { mode: BuildMode; value: number; above?: boolean }; // above: a top-down picture (images only)
export type MinecraftInfo = {
  dims: [number, number, number]; // width (x), height (y), length (z) in blocks
  blocks: number;
  block_size: number;
  block_size_unit: "m" | "units"; // metres for metric models
  budget_limited: boolean; // blocks were made bigger to stay under the block budget
  too_tall: boolean; // taller than Minecraft's 384-block build height
  single_file: boolean; // small enough for one .schem
  tiles: number;
  fov_deg: number; // vertical FOV that makes the build line up with the image
  pitch_deg: number; // in-game pitch to look along the camera
  yaw_deg: number; // in-game yaw (0 = south); scene frames can face any way
  offset: [number, number, number];
  preview_lod: number; // the 3D preview merges this many blocks per side
  data_version: number;
  top_blocks: [string, number][];
};
// `frame`: for scenes, the frame whose camera is the standing spot.
const q = (b: BuildSettings, frame = 0) => `mode=${b.mode}&value=${b.value}&frame=${frame}&above=${!!b.above}`;
/** axes: row-major 3x3, columns = Minecraft east, up, south in cloud coordinates; origin: the eye. */
export type BlockPreview = { n: number; size: number; lo: Float32Array; axes: Float32Array; origin: Float32Array; ijk: Int16Array; rgb: Uint8Array };

/** Builds the blocks in a background job (cached on the server); resolves with the summary. */
export const buildMinecraft = (id: string, b: BuildSettings, frame: number, onProgress?: OnProgress, signal?: AbortSignal) =>
  startJob(`${API}/runs/${id}/minecraft/build?${q(b, frame)}`, { signal }).then((job) => waitJob<MinecraftInfo>(job, onProgress, signal));

/** Writes the build as a zip of tiles; resolves with its download URL. */
export const exportTiles = (id: string, b: BuildSettings, frame: number, onProgress?: OnProgress) =>
  startJob(`${API}/runs/${id}/minecraft/tiles?${q(b, frame)}`).then((job) => waitJob<string>(job, onProgress)).then((u) => API + u);

// Layout documented in backend/store.py:minecraft_preview_bin.
export async function getBlocks(id: string, b: BuildSettings, frame: number, signal?: AbortSignal): Promise<BlockPreview> {
  await buildMinecraft(id, b, frame, undefined, signal); // build as a cancellable job; the preview then comes from the cache
  const r = await fetch(`${API}/runs/${id}/minecraft.bin?${q(b, frame)}`, { signal });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const buf = await r.arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  const f = new Float32Array(buf, 4, 16);
  const head = 4 + 16 * 4;
  return {
    n, size: f[0], lo: f.slice(1, 4), axes: f.slice(4, 13), origin: f.slice(13, 16),
    ijk: new Int16Array(buf, head, n * 3), rgb: new Uint8Array(buf, head + n * 6, n * 3),
  };
}

export const schemUrl = (id: string, b: BuildSettings, frame: number) => `${API}/runs/${id}/minecraft.schem?${q(b, frame)}`;
