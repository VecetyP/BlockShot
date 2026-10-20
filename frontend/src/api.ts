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
