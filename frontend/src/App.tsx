import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "./api";
import { COLOR_MODES, type ColorMode } from "./colors";
import Viewer, { OrbitSync } from "./Viewer";

const MAX_PANELS = 4;
const modelLabel = (r: api.RunMeta) => String(r.model).split("/").pop();

function Panel(props: { id: string; mode: ColorMode; pointSize: number; snapKey: number; sync?: OrbitSync; titled: boolean }) {
  const [meta, setMeta] = useState<api.RunMeta | null>(null);
  const [cloud, setCloud] = useState<api.Cloud | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localSnap, setLocalSnap] = useState(0);

  useEffect(() => {
    setCloud(null);
    setError(null);
    Promise.all([api.getRun(props.id), api.getCloud(props.id)])
      .then(([m, c]) => { setMeta(m); setCloud(c); })
      .catch((e) => setError(String(e)));
  }, [props.id]);

  return (
    <div className="panel">
      {props.titled && meta && (
        <div className="panel-head">
          <span>{modelLabel(meta)} <small>{meta.scale_kind}</small></span>
          <button onClick={() => setLocalSnap((k) => k + 1)}>Snap</button>
        </div>
      )}
      {cloud && meta ? (
        <Viewer
          cloud={cloud}
          meta={meta}
          mode={props.mode === "confidence" && !cloud.confidence ? "rgb" : props.mode}
          pointSize={props.pointSize}
          snapKey={props.snapKey + localSnap}
          sync={props.sync}
        />
      ) : (
        <p className="empty">{error ?? "Loading…"}</p>
      )}
    </div>
  );
}

function Strip({ run }: { run: api.RunMeta }) {
  // Normal-only models store a placeholder plane; its depth and discontinuity maps are meaningless.
  const hasDepth = !String(run.depth_source ?? "").startsWith("none");
  const maps = [
    ...(hasDepth ? ["depth"] : []),
    ...(run.has_confidence ? ["confidence"] : []),
    ...(run.has_normals ? ["normals"] : []),
    ...(hasDepth ? ["discontinuity"] : []),
  ];
  return (
    <div className="strip">
      <figure><img src={api.inputUrl(run.id)} alt="input" /><figcaption>{modelLabel(run)} · input</figcaption></figure>
      {maps.map((k) => (
        <figure key={k}><img src={api.mapUrl(run.id, k)} alt={k} /><figcaption>{k}</figcaption></figure>
      ))}
    </div>
  );
}

export default function App() {
  const [models, setModels] = useState<api.ModelInfo[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [runs, setRuns] = useState<api.RunMeta[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState<ColorMode>("rgb");
  const [pointSize, setPointSize] = useState(2);
  const [snapKey, setSnapKey] = useState(0);
  // A fresh sync object per selection, so panels from an old grid don't keep listening.
  const sync = useMemo(() => new OrbitSync(), [selected.join()]);

  const refreshRuns = useCallback(() => api.getRuns().then(setRuns).catch((e) => setError(String(e))), []);

  useEffect(() => {
    api.getModels().then((m) => {
      setModels(m);
      setPicked(m.slice(0, 1).map((x) => x.name));
    }).catch((e) => setError(`backend unreachable: ${e}`));
    refreshRuns();
  }, [refreshRuns]);

  const run = async () => {
    if (!file || !picked.length) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.predict(file, picked);
      await refreshRuns();
      setSelected(out.slice(0, MAX_PANELS).map((r) => r.id));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const toggle = (id: string, on: boolean) =>
    setSelected((s) => (on ? [...s.filter((x) => x !== id), id].slice(-MAX_PANELS) : s.filter((x) => x !== id)));
  const selectedRuns = selected.map((id) => runs.find((r) => r.id === id)).filter((r): r is api.RunMeta => !!r);
  const anyConfidence = selectedRuns.some((r) => r.has_confidence);

  return (
    <div className="app">
      <aside className="sidebar">
        <section
          className={`drop ${file ? "has-file" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); setFile(e.dataTransfer.files[0] ?? null); }}
        >
          <label>
            <input type="file" accept="image/*" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            {file ? file.name : "Drop an image or click to choose"}
          </label>
        </section>
        <fieldset className="models">
          <legend>Models</legend>
          {models.map((m) => (
            <label key={m.name}>
              <input
                type="checkbox"
                checked={picked.includes(m.name)}
                onChange={(e) => setPicked((p) => (e.target.checked ? [...p, m.name] : p.filter((x) => x !== m.name)))}
              />
              {m.name} <small>{m.scale_kind}{m.loaded ? " · loaded" : ""}</small>
            </label>
          ))}
        </fieldset>
        <button className="primary" disabled={!file || !picked.length || busy} onClick={run}>
          {busy ? "Running…" : "Run"}
        </button>
        {error && <p className="error">{error}</p>}

        <h2>History <small>· tick up to {MAX_PANELS} to compare</small></h2>
        <ul className="history">
          {runs.map((r) => (
            <li key={r.id} className={selected.includes(r.id) ? "active" : ""}>
              <input
                type="checkbox"
                aria-label={`compare ${r.source ?? r.id}`}
                checked={selected.includes(r.id)}
                onChange={(e) => toggle(r.id, e.target.checked)}
              />
              <button onClick={() => setSelected([r.id])}>
                <img src={api.inputUrl(r.id)} alt="" loading="lazy" />
                <span>
                  {r.source ?? r.id}
                  <small>{modelLabel(r)} · {r.created.replace("T", " ")}</small>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main>
        <div className="toolbar">
          <div className="modes" role="radiogroup" aria-label="Color mode">
            {COLOR_MODES.map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={mode === m}
                className={mode === m ? "active" : ""}
                disabled={m === "confidence" && !anyConfidence}
                onClick={() => setMode(m)}
              >
                {m}
              </button>
            ))}
          </div>
          <label className="size">
            Point size
            <input type="range" min={1} max={8} step={0.5} value={pointSize} onChange={(e) => setPointSize(+e.target.value)} />
            {pointSize}px
          </label>
          <button onClick={() => setSnapKey((k) => k + 1)} disabled={!selected.length}>Snap to input camera</button>
        </div>

        <div className={`stage grid-${selected.length}`}>
          {selected.length ? (
            selected.map((id) => (
              <Panel
                key={id}
                id={id}
                mode={mode}
                pointSize={pointSize}
                snapKey={snapKey}
                sync={selected.length > 1 ? sync : undefined}
                titled={selected.length > 1}
              />
            ))
          ) : (
            <p className="empty">Pick a run from the history or run a new image.</p>
          )}
        </div>

        {selectedRuns.length > 0 && (
          <div className="strips">
            {selectedRuns.map((r) => <Strip key={r.id} run={r} />)}
          </div>
        )}
      </main>
    </div>
  );
}
