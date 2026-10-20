import type { Cloud } from "./api";

export const COLOR_MODES = ["rgb", "depth", "confidence", "discontinuity", "curl", "residual"] as const;
export const FIELD_MODES: readonly ColorMode[] = ["curl", "residual"]; // fetched per run on demand
export type ColorMode = (typeof COLOR_MODES)[number];

// Polynomial fit of Google's Turbo colormap, x in [0, 1].
function turbo(x: number, out: Float32Array, i: number) {
  out[i] = 0.1357 + x * (4.6154 + x * (-42.6603 + x * (132.1311 + x * (-152.9424 + x * 59.2864))));
  out[i + 1] = 0.0914 + x * (2.1942 + x * (4.843 + x * (-14.185 + x * (4.2773 + x * 2.8296))));
  out[i + 2] = 0.1067 + x * (12.6419 + x * (-60.582 + x * (110.3628 + x * (-89.9031 + x * 27.3482))));
}

function percentile(v: Float32Array, p: number) {
  const s = Float32Array.from(v).sort();
  return s[Math.min(s.length - 1, Math.floor(p * s.length))] ?? 0;
}

/** Display-only normalization; the cloud's raw values are never modified. */
export function colorize(cloud: Cloud, mode: ColorMode, field?: Float32Array | null): Float32Array {
  const out = new Float32Array(cloud.n * 3);
  if (mode === "curl" && field) {
    // Signed, relative to the run's RMS gradient, on a fixed ±1 range: green is zero curl, red/blue the two senses.
    for (let i = 0; i < cloud.n; i++) turbo(Math.min(1, Math.max(0, (field[i] + 1) / 2)), out, 3 * i);
    return out;
  }
  if (mode === "rgb" || (mode === "confidence" && !cloud.confidence) || ((mode === "curl" || mode === "residual") && !field)) {
    for (let i = 0; i < out.length; i++) out[i] = cloud.rgb[i] / 255;
    return out;
  }
  let v: Float32Array;
  if (mode === "depth") {
    v = new Float32Array(cloud.n);
    for (let i = 0; i < cloud.n; i++) v[i] = cloud.xyz[3 * i + 2];
  } else v = mode === "confidence" ? cloud.confidence! : mode === "residual" ? field! : cloud.discontinuity;
  const lo = mode === "confidence" || mode === "residual" ? 0 : percentile(v, 0.01);
  const hi = mode === "confidence" || mode === "residual" ? 1 : percentile(v, 0.99); // residual: fixed, relative units
  const span = Math.max(hi - lo, 1e-12);
  for (let i = 0; i < cloud.n; i++) turbo(Math.min(1, Math.max(0, (v[i] - lo) / span)), out, 3 * i);
  return out;
}
