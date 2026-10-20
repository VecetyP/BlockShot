import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import { COLOR_MODES, type ColorMode } from "./colors";
import Viewer from "./Viewer";

export default function App() {
  const [models, setModels] = useState<api.ModelInfo[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [runs, setRuns] = useState<api.RunMeta[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [runId, setRunId] = useState<string | null>(null);
  const [meta, setMeta] = useState<api.RunMeta | null>(null);
  const [cloud, setCloud] = useState<api.Cloud | null>(null);
  const [mode, setMode] = useState<ColorMode>("rgb");
  const [pointSize, setPointSize] = useState(2);
  const [snapKey, setSnapKey] = useState(0);

  const refreshRuns = useCallback(() => api.getRuns().then(setRuns).catch((e) => setError(String(e))), []);

  useEffect(() => {
    api.getModels().then((m) => {
      setModels(m);
      setPicked(m.slice(0, 1).map((x) => x.name));
    }).catch((e) => setError(`backend unreachable: ${e}`));
    refreshRuns();
  }, [refreshRuns]);

  useEffect(() => {
    if (!runId) return;
    setCloud(null);
    Promise.all([api.getRun(runId), api.getCloud(runId)])
      .then(([m, c]) => { setMeta(m); setCloud(c); })
      .catch((e) => setError(String(e)));
  }, [runId]);

  const run = async () => {
    if (!file || !picked.length) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.predict(file, picked);
      await refreshRuns();
      setRunId(out[0].id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const maps = meta ? ["depth", ...(meta.has_confidence ? ["confidence"] : []), "discontinuity"] : [];

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

        <h2>History</h2>
        <ul className="history">
          {runs.map((r) => (
            <li key={r.id}>
              <button className={r.id === runId ? "active" : ""} onClick={() => setRunId(r.id)}>
                <img src={api.inputUrl(r.id)} alt="" loading="lazy" />
                <span>
                  {r.source ?? r.id}
                  <small>{r.model.split("/").pop()} · {r.created.replace("T", " ")}</small>
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
                disabled={m === "confidence" && !cloud?.confidence}
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
          <button onClick={() => setSnapKey((k) => k + 1)} disabled={!cloud}>Snap to input camera</button>
          {cloud && <span className="count">{cloud.n.toLocaleString()} points</span>}
        </div>

        <div className="stage">
          {cloud && meta ? (
            <Viewer cloud={cloud} meta={meta} mode={mode} pointSize={pointSize} snapKey={snapKey} />
          ) : (
            <p className="empty">{runId ? "Loading…" : "Pick a run from the history or run a new image."}</p>
          )}
        </div>

        {meta && (
          <div className="strip">
            <figure><img src={api.inputUrl(meta.id)} alt="input" /><figcaption>input</figcaption></figure>
            {maps.map((k) => (
              <figure key={k}><img src={api.mapUrl(meta.id, k)} alt={k} /><figcaption>{k}</figcaption></figure>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
