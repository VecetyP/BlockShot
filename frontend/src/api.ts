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

export type FillOrder = "slope" | "curvature";
export type FillSettings = { on: boolean; reach: number; order: FillOrder; tint: boolean }; // reach: % of image width
export type FillCloud = { n: number; xyz: Float32Array; rgb: Uint8Array };

// Layout documented in backend/store.py:fill_bin.
export async function getFill(id: string, reach: number, order: FillOrder): Promise<FillCloud> {
  const r = await fetch(`${API}/runs/${id}/fill.bin?reach=${reach}&order=${order}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const buf = await r.arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  return { n, xyz: new Float32Array(buf, 4, n * 3), rgb: new Uint8Array(buf, 4 + n * 12, n * 3) };
}

export type OffAxisView = {
  yaw: number;
  gaps: string; // render with gaps marked
  filled?: string; // same view with the fill added
  gap_px: number;
  other_hidden_px: number;
  covered?: number; // share of gap pixels the fill covers
};

export const renderOffAxis = (id: string, fill?: { reach: number; order: FillOrder }) =>
  fetch(`${API}/runs/${id}/offaxis${fill ? `?reach=${fill.reach}&order=${fill.order}` : ""}`, { method: "POST" })
    .then(json<OffAxisView[]>)
    .then((views) =>
      views.map((v) => ({ ...v, gaps: API + v.gaps, filled: v.filled && API + v.filled })),
    );
