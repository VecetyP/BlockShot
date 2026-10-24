import { useEffect, useRef, useState } from "react";
import * as api from "./api";

/** Worlds: video scenes placed into one map over several recording sessions (backend/world.py).
 * A new scene arrives at the spot whose roofs match best; drag it on the map, turn or resize it,
 * then confirm. Only confirmed scenes go into the export. */

// world = scale-adjusted cell turned by quarter turns (+ yaw), plus offset (as backend/world.py rot / place)
const turn = (t: number, yaw: number, x: number, z: number): [number, number] => {
  const a = ((90 * t + yaw) * Math.PI) / 180;
  const c = Math.round(Math.cos(a) * 1e12) / 1e12, s = Math.round(Math.sin(a) * 1e12) / 1e12;
  return [c * x + s * z, -s * x + c * z];
};
const HUES = [210, 30, 120, 280, 0, 170, 60, 320];

function place(s: api.WorldScene, m: api.WorldMap[number] | undefined) {
  if (!m) return [];
  const k = s.scale / m.scale;
  return m.cells.map(([x, z, h]) => {
    const [wx, wz] = turn(s.turns, s.yaw ?? 0, x * k, z * k);
    return [wx + s.offset[0], wz + s.offset[2], h * k + s.offset[1]] as const;
  });
}

export default function World({ scenes }: { scenes: api.RunMeta[] }) {
  const [names, setNames] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [draftName, setDraftName] = useState("");
  const [world, setWorld] = useState<api.World | null>(null);
  const [map, setMap] = useState<api.WorldMap>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [pick, setPick] = useState("");
  const [shape, setShape] = useState<"voxels" | "boxes">("boxes");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zip, setZip] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const view = useRef({ x0: 0, z0: 0, k: 1 }); // world block -> canvas pixel
  const drag = useRef<{ x: number; z: number; off: [number, number, number] } | null>(null);

  useEffect(() => void api.getWorlds().then(setNames).catch((e) => setError(String(e))), []);

  const task = async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };
  const progress = (j: api.JobState) => setBusy(j.message || j.stage);
  const reloadMap = (n: string) => api.getWorldMap(n, progress).then(setMap);

  const open = (n: string) =>
    task("opening", async () => {
      const w = await api.getWorld(n);
      setName(n);
      setWorld(w);
      setSel(null);
      setZip(null);
      setMap(w.sessions.length ? await api.getWorldMap(n, progress) : []);
    });

  const add = () =>
    task("adding scene", async () => {
      const w = await api.addWorldScene(name, pick, progress);
      setWorld(w);
      setSel(w.sessions.length - 1);
      await reloadMap(name);
      setNames((ns) => (ns.includes(name) ? ns : [...ns, name].sort()));
    });

  const addVideo = (file: File) =>
    task("cutting the recording", async () => {
      const w = await api.addWorldVideo(name, file, progress);
      setWorld(w);
      setSel(null);
      await reloadMap(name);
      setNames((ns) => (ns.includes(name) ? ns : [...ns, name].sort()));
    });

  const change = (i: number, c: Partial<api.WorldScene>, save = true) => {
    setWorld((w) => w && { ...w, sessions: w.sessions.map((s, j) => (j === i ? { ...s, ...c } : s)) });
    if (save) api.updateWorldScene(name, i, c).then(setWorld).catch((e) => setError(String(e)));
  };

  // Turn or resize about the scene's middle on the map, so it stays where it was put.
  const reshape = (i: number, turns: number, scale: number) => {
    const s = world!.sessions[i];
    const pts = place(s, map[i]);
    if (!pts.length) return change(i, { turns, scale });
    const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
    const mz = pts.reduce((a, p) => a + p[1], 0) / pts.length;
    const moved = place({ ...s, turns, scale, offset: [0, s.offset[1], 0] }, map[i]);
    const nx = moved.reduce((a, p) => a + p[0], 0) / moved.length;
    const nz = moved.reduce((a, p) => a + p[1], 0) / moved.length;
    change(i, { turns, scale, offset: [mx - nx, s.offset[1], mz - nz] });
  };

  useEffect(() => {
    const c = canvas.current;
    if (!c || !world) return;
    const ctx = c.getContext("2d")!;
    const W = (c.width = c.clientWidth * devicePixelRatio);
    const H = (c.height = c.clientHeight * devicePixelRatio);
    ctx.clearRect(0, 0, W, H);
    const all = world.sessions.map((s, i) => place(s, map[i]));
    const flat = all.flat();
    if (!flat.length) return;
    if (!drag.current) {
      const xs = flat.map((p) => p[0]), zs = flat.map((p) => p[1]);
      const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
      const k = 0.9 * Math.min(W / Math.max(1, x1 - x0), H / Math.max(1, z1 - z0));
      view.current = { k, x0: (x0 + x1) / 2 - W / 2 / k, z0: (z0 + z1) / 2 - H / 2 / k };
    }
    const { x0, z0, k } = view.current;
    const hs = flat.map((p) => p[2]);
    const [h0, h1] = [Math.min(...hs), Math.max(...hs)];
    const order = world.sessions.map((_, i) => i).filter((i) => i !== sel).concat(sel ?? []);
    for (const i of order) {
      const s = world.sessions[i];
      const size = Math.max(1, (map[i]?.cell ?? 1) * (s.scale / (map[i]?.scale ?? s.scale)) * k);
      ctx.globalAlpha = s.confirmed ? 0.9 : 0.6;
      for (const [x, z, h] of all[i]) {
        const light = 25 + (55 * (h - h0)) / Math.max(1e-6, h1 - h0); // higher = lighter
        ctx.fillStyle = `hsl(${HUES[i % HUES.length]} ${i === sel ? 80 : 35}% ${light}%)`;
        ctx.fillRect((x - x0) * k, (z - z0) * k, size, size);
      }
    }
    ctx.globalAlpha = 1;
  }, [world, map, sel]);

  const toWorld = (e: React.PointerEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    const { x0, z0, k } = view.current;
    return [x0 + ((e.clientX - r.left) * devicePixelRatio) / k, z0 + ((e.clientY - r.top) * devicePixelRatio) / k];
  };

  const s = sel !== null ? world?.sessions[sel] : undefined;
  const sceneName = (id: string) => scenes.find((r) => r.id === id)?.source ?? id;
  return (
    <div className="world">
      <div className="toolbar">
        <select value={name} onChange={(e) => open(e.target.value)}>
          <option value="" disabled>Open a world…</option>
          {names.map((n) => <option key={n}>{n}</option>)}
        </select>
        <input placeholder="new world name" value={draftName} onChange={(e) => setDraftName(e.target.value.replace(/[^A-Za-z0-9_-]/g, ""))} />
        <button disabled={!draftName || !!busy} onClick={() => open(draftName)}>Create</button>
        {world && (
          <>
            <select value={pick} onChange={(e) => setPick(e.target.value)}>
              <option value="" disabled>Video scene to add…</option>
              {scenes.map((r) => <option key={r.id} value={r.id}>{r.source ?? r.id}</option>)}
            </select>
            <button className="primary" disabled={!pick || !!busy} onClick={add}>Add scene</button>
            <label className="button" title="Cuts a long recording into 2.5 s pieces at 8 fps; each piece is placed after the one before. Pieces whose frames disagree are left unconfirmed.">
              Add recording…
              <input type="file" accept="video/*" className="visually-hidden" disabled={!!busy}
                onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) addVideo(f); }} />
            </label>
          </>
        )}
        {busy && <span className="hint">{busy}…</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {world && (
        <div className="world-body">
          <canvas
            ref={canvas}
            className="world-map"
            onPointerDown={(e) => {
              if (!s || sel === null) return;
              const [x, z] = toWorld(e);
              drag.current = { x, z, off: s.offset };
              (e.target as Element).setPointerCapture(e.pointerId);
            }}
            onPointerMove={(e) => {
              if (!drag.current || sel === null) return;
              const [x, z] = toWorld(e);
              const o = drag.current.off;
              change(sel, { offset: [Math.round(o[0] + x - drag.current.x), o[1], Math.round(o[2] + z - drag.current.z)] }, false);
            }}
            onPointerUp={() => {
              if (drag.current && sel !== null && s) change(sel, { offset: s.offset });
              drag.current = null;
            }}
          />
          <aside className="world-side">
            <ol className="world-scenes">
              {world.sessions.map((x, i) => (
                <li key={i} className={i === sel ? "active" : ""}>
                  <button onClick={() => setSel(i)}>
                    <span><span className="swatch" style={{ background: `hsl(${HUES[i % HUES.length]} 70% 50%)` }} />{sceneName(x.scene)}</span>
                    <small>
                      {x.confirmed ? "placed" : x.ghosts !== undefined ? "unsure: its frames disagree" : `proposed${x.score === null ? " (no match: place it by hand)" : ` · match ${x.score.toFixed(2)}`}`}
                      {x.ghosts !== undefined && ` · ${Math.round(100 * x.ghosts)}% dropped`}
                    </small>
                  </button>
                </li>
              ))}
            </ol>
            {s && sel !== null && (
              <div className="world-edit">
                <p className="note">Drag on the map to move it.{s.eye_scale ? "" : " Its size was guessed (no floor seen under the camera)."}</p>
                <div className="row">
                  <button onClick={() => reshape(sel, (s.turns + 1) % 4, s.scale)}>⟲ 90°</button>
                  <button onClick={() => reshape(sel, (s.turns + 3) % 4, s.scale)}>⟳ 90°</button>
                  <button onClick={() => reshape(sel, s.turns, s.scale / 1.03)}>Smaller</button>
                  <button onClick={() => reshape(sel, s.turns, s.scale * 1.03)}>Bigger</button>
                </div>
                <label className="slider">
                  <span>Height</span>
                  <button onClick={() => change(sel, { offset: [s.offset[0], s.offset[1] - 1, s.offset[2]] })}>−1</button>
                  <span className="val">{Math.round(s.offset[1]) || 0}</span>
                  <button onClick={() => change(sel, { offset: [s.offset[0], s.offset[1] + 1, s.offset[2]] })}>+1</button>
                </label>
                <div className="row">
                  {!s.confirmed && <button className="primary" onClick={() => change(sel, { confirmed: true })}>Confirm</button>}
                  <button className="quiet" onClick={() => task("removing", async () => {
                    setWorld(await api.removeWorldScene(name, sel));
                    setSel(null);
                    await reloadMap(name);
                  })}>Remove</button>
                </div>
              </div>
            )}
            <div className="world-export">
              <label className="slider">
                <span>Shape</span>
                <select value={shape} onChange={(e) => setShape(e.target.value as "voxels" | "boxes")}>
                  <option value="voxels">As seen</option>
                  <option value="boxes">Boxes (buildings)</option>
                </select>
              </label>
              <button disabled={!!busy || !world.sessions.some((x) => x.confirmed)} onClick={() => task("exporting", async () => setZip(await api.exportWorld(name, shape, progress)))}>
                Export tiles
              </button>
              {zip && <a className="button primary" href={zip} download>Download world (.zip)</a>}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
