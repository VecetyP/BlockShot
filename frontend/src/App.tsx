import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "./api";
import { COLOR_MODES, FIELD_MODES, type ColorMode } from "./colors";
import Viewer, { OrbitSync } from "./Viewer";

const MAX_PANELS = 4;
const modelLabel = (r: api.RunMeta) => String(r.model).split("/").pop();

const pct = (x?: number) => (x === undefined ? "–" : `${(100 * x).toFixed(1)}%`);

function Panel(props: {
  id: string;
  mode: ColorMode;
  pointSize: number;
  snapKey: number;
  sync?: OrbitSync;
  titled: boolean;
  showTears: boolean;
}) {
  const [meta, setMeta] = useState<api.RunMeta | null>(null);
  const [cloud, setCloud] = useState<api.Cloud | null>(null);
  const [analysis, setAnalysis] = useState<api.Analysis | null>(null);
  const [field, setField] = useState<{ kind: string; values: Float32Array | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localSnap, setLocalSnap] = useState(0);

  useEffect(() => {
    setCloud(null);
    setError(null);
    setAnalysis(null);
    setField(null);
    Promise.all([api.getRun(props.id), api.getCloud(props.id)])
      .then(([m, c]) => { setMeta(m); setCloud(c); })
      .catch((e) => setError(String(e)));
    api.getAnalysis(props.id).then(setAnalysis).catch(() => {});
  }, [props.id]);

  const wantsField = FIELD_MODES.includes(props.mode);
  useEffect(() => {
    if (!wantsField || field?.kind === props.mode) return;
    api.getField(props.id, props.mode).then((values) => setField({ kind: props.mode, values }));
  }, [props.id, props.mode, wantsField, field?.kind]);

  const unavailable =
    (props.mode === "confidence" && !cloud?.confidence) || (wantsField && field?.kind === props.mode && !field.values);
  const shownMode = unavailable || (wantsField && field?.kind !== props.mode) ? "rgb" : props.mode;
  const integ = analysis?.integrability;

  return (
    <div className="panel">
      {meta && (
        <div className="panel-head">
          <span>
            {props.titled && <>{modelLabel(meta)} <small>{meta.scale_kind}</small> </>}
            {analysis && (
              <small className="stats">
                {[
                  analysis.reprojection_px !== null && `reproj ${analysis.reprojection_px.toFixed(3)} px`,
                  integ?.native && `native residual ${pct(integ.native.residual_fraction)}`,
                  integ?.derived && `depth residual ${pct(integ.derived.residual_fraction)}`,
                ].filter(Boolean).join(" · ")}
              </small>
            )}
            {unavailable && <small className="warn"> · no {props.mode} for this model</small>}
          </span>
          {props.titled && <button onClick={() => setLocalSnap((k) => k + 1)}>Snap</button>}
        </div>
      )}
      {cloud && meta ? (
        <Viewer
          cloud={cloud}
          meta={meta}
          mode={shownMode}
          field={field?.kind === props.mode ? field.values : null}
          tears={props.showTears ? analysis?.tears : undefined}
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

function Strip({ run, showTears }: { run: api.RunMeta; showTears: boolean }) {
  const [tears, setTears] = useState<api.Tear[]>([]);
  const [offAxis, setOffAxis] = useState<string[] | null>(null);
  const [rendering, setRendering] = useState(false);
  useEffect(() => {
    setOffAxis(null);
    api.getAnalysis(run.id).then((a) => setTears(a.tears)).catch(() => setTears([]));
  }, [run.id]);

  // Normal-only models store a placeholder plane; its depth and discontinuity maps are meaningless.
  const hasDepth = !String(run.depth_source ?? "").startsWith("none");
  const maps = [
    ...(hasDepth ? ["depth"] : []),
    ...(run.has_confidence ? ["confidence"] : []),
    ...(run.has_normals ? ["normals"] : []),
    ...(hasDepth ? ["discontinuity"] : []),
    ...(run.has_normals || hasDepth ? ["curl", "residual"] : []),
  ];
  const renderOffAxis = () => {
    setRendering(true);
    api.renderOffAxis(run.id).then(setOffAxis).finally(() => setRendering(false));
  };
  return (
    <div className="strip">
      <figure className="with-tears">
        <div className="marked">
          <img src={api.inputUrl(run.id)} alt="input" />
          {showTears &&
            tears.map((t, i) => (
              <span key={i} className="tear" style={{ left: `${(100 * (t.u + 0.5)) / run.width}%`, top: `${(100 * (t.v + 0.5)) / run.height}%` }}>
                {i + 1}
              </span>
            ))}
        </div>
        <figcaption>{modelLabel(run)} · input</figcaption>
      </figure>
      {maps.map((k) => (
        <figure key={k}><img src={api.mapUrl(run.id, k)} alt={k} /><figcaption>{k}</figcaption></figure>
      ))}
      {showTears && tears.length > 0 && (
        <ol className="tear-list" aria-label="Top tears by pixel">
          {tears.map((t, i) => <li key={i}>({t.u}, {t.v}) <small>{t.score.toPrecision(3)}</small></li>)}
        </ol>
      )}
      {hasDepth && (
        offAxis ? (
          offAxis.map((u) => (
            <figure key={u}><img src={u} alt="" /><figcaption>{u.match(/yaw_([+-]\d+)/)?.[1]}°</figcaption></figure>
          ))
        ) : (
          <button className="offaxis" onClick={renderOffAxis} disabled={rendering}>
            {rendering ? "Rendering…" : "Off-axis renders"}
          </button>
        )
      )}
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
  const [showTears, setShowTears] = useState(false);
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
          <label className="size">
            <input type="checkbox" checked={showTears} onChange={(e) => setShowTears(e.target.checked)} />
            Tears
          </label>
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
                showTears={showTears}
              />
            ))
          ) : (
            <p className="empty">Pick a run from the history or run a new image.</p>
          )}
        </div>

        {selectedRuns.length > 0 && (
          <div className="strips">
            {selectedRuns.map((r) => <Strip key={r.id} run={r} showTears={showTears} />)}
          </div>
        )}
      </main>
    </div>
  );
}
