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
  telephone?: { root: string; step: number; yaw: number }; // set on telephone-game generations
  [k: string]: unknown;
};

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

export const predict = (file: File, models: string[], force = false) => {
  const fd = new FormData();
  fd.append("image", file);
  models.forEach((m) => fd.append("models", m));
  fd.append("force", String(force));
  return fetch(`${API}/predict`, { method: "POST", body: fd }).then(
    json<{ id: string; cached: boolean; meta: RunMeta }[]>,
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

export const renderOffAxis = (id: string) =>
  fetch(`${API}/runs/${id}/offaxis`, { method: "POST" }).then(json<string[]>).then((u) => u.map((x) => API + x));

export type TelephoneStep = { step: number; id: string; turned: number; meta: RunMeta };

/** Render the cloud `yaw` degrees to the side, feed it back into the same model, `steps` times. */
export const telephone = (id: string, steps: number, yaw: number) =>
  fetch(`${API}/runs/${id}/telephone?steps=${steps}&yaw=${yaw}`, { method: "POST" }).then(json<TelephoneStep[]>);

export type MinecraftInfo = {
  dims: [number, number, number]; // width (x), height (y), length (z) in blocks
  blocks: number;
  fov_deg: number; // vertical FOV that makes the build line up with the image
  offset: [number, number, number];
  data_version: number;
  top_blocks: [string, number][];
};
export type BlockPreview = { n: number; size: number; lo: Float32Array; ijk: Int16Array; rgb: Uint8Array };

export const getMinecraftInfo = (id: string, size: number) =>
  fetch(`${API}/runs/${id}/minecraft.json?size=${size}`).then(json<MinecraftInfo>);

// Layout documented in backend/store.py:minecraft_preview_bin.
export async function getBlocks(id: string, size: number): Promise<BlockPreview> {
  const r = await fetch(`${API}/runs/${id}/minecraft.bin?size=${size}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const buf = await r.arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  const f = new Float32Array(buf, 4, 4);
  return { n, size: f[0], lo: f.slice(1), ijk: new Int16Array(buf, 20, n * 3), rgb: new Uint8Array(buf, 20 + n * 6, n * 3) };
}

export const schemUrl = (id: string, size: number) => `${API}/runs/${id}/minecraft.schem?size=${size}`;
