import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as api from "./api";
import { COLOR_MODES, FIELD_MODES, type ColorMode } from "./colors";
import Viewer, { OrbitSync, type NavSettings } from "./Viewer";

const MAX_PANELS = 4;
const RECOMMENDED_MODEL = "moge"; // the best-looking builds in practice; listed first and picked by default
const modelLabel = (r: api.RunMeta) => String(r.model).split("/").pop();

const DEFAULT_NAV: NavSettings = { move: 0.5, rotate: 1, zoom: 1, pan: 1 };
const NAV_KEY = "nav-settings";

// A per-viewer convenience: storage can be unavailable (private window, blocked site data).
function loadNav(): NavSettings {
  try {
    return { ...DEFAULT_NAV, ...JSON.parse(localStorage.getItem(NAV_KEY) ?? "{}") };
  } catch {
    return DEFAULT_NAV;
  }
}

function Slider(props: { label: string; min: number; max: number; step: number; value: number; unit?: string; onChange: (v: number) => void }) {
  return (
    <label className="slider">
      <span>{props.label}</span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(+e.target.value)}
      />
      <span className="val">{props.value}{props.unit ?? "×"}</span>
    </label>
  );
}

const pct = (x?: number) => (x === undefined ? "–" : `${(100 * x).toFixed(1)}%`);

function Panel(props: {
  id: string;
  mode: ColorMode;
  pointSize: number;
  snapKey: number;
  sync?: OrbitSync;
  nav: NavSettings;
  titled: boolean;
  showTears: boolean;
  blockSize?: number; // when set, show the Minecraft build of this many blocks instead of the points
}) {
  const [meta, setMeta] = useState<api.RunMeta | null>(null);
  const [cloud, setCloud] = useState<api.Cloud | null>(null);
  const [analysis, setAnalysis] = useState<api.Analysis | null>(null);
  const [field, setField] = useState<{ kind: string; values: Float32Array | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localSnap, setLocalSnap] = useState(0);
  const [blocks, setBlocks] = useState<{ key: string; data: api.BlockPreview | null } | null>(null);

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
  const buildable = !!meta && !String(meta.depth_source ?? "").startsWith("none");
  const blocksKey = `${props.id}:${props.blockSize}`;
  useEffect(() => {
    if (!props.blockSize || !buildable || blocks?.key === blocksKey) return;
    api.getBlocks(props.id, props.blockSize)
      .then((data) => setBlocks({ key: blocksKey, data }))
      .catch(() => setBlocks({ key: blocksKey, data: null }));
  }, [props.blockSize, buildable, blocksKey, blocks?.key, props.id]);
  const shownBlocks = props.blockSize && blocks?.key === blocksKey ? blocks.data : null;

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
          nav={props.nav}
          blocks={shownBlocks}
        />
      ) : (
        <p className="empty">{error ?? "Loading…"}</p>
      )}
    </div>
  );
}

type LightboxItem = { src: string; caption: string };

const Icon = ({ d }: { d: string }) => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={d} />
  </svg>
);

/** Enlarged view of one strip image. ←/→ step through the strip, Esc or a click outside closes. */
function Lightbox({ items, index, onIndex, onClose }: { items: LightboxItem[]; index: number; onIndex: (i: number) => void; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();
  }, []);
  const step = useCallback((by: number) => onIndex((index + by + items.length) % items.length), [index, items.length, onIndex]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") step(1);
      if (e.key === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [step]);
  const item = items[index];
  return (
    <dialog
      ref={ref}
      className="lightbox"
      aria-label={item.caption}
      onClose={onClose}
      onClick={(e) => e.target === e.currentTarget && ref.current?.close()}
    >
      <figure>
        <img src={item.src} alt={item.caption} />
        <figcaption>
          <span>{item.caption}</span>
          {items.length > 1 && <span className="counter">{index + 1} / {items.length}</span>}
        </figcaption>
      </figure>
      {items.length > 1 && (
        <>
          <button className="icon nav-prev" aria-label="Previous image" onClick={() => step(-1)}><Icon d="M15 18l-6-6 6-6" /></button>
          <button className="icon nav-next" aria-label="Next image" onClick={() => step(1)}><Icon d="M9 18l6-6-6-6" /></button>
        </>
      )}
      <button className="icon close" aria-label="Close" onClick={() => ref.current?.close()}><Icon d="M18 6L6 18M6 6l12 12" /></button>
    </dialog>
  );
}

/** Download a run's Minecraft build, with what to do in game so it lines up with the image. */
function MinecraftCard({ run, size }: { run: api.RunMeta; size: number }) {
  const [info, setInfo] = useState<api.MinecraftInfo | null>(null);
  useEffect(() => {
    setInfo(null);
    api.getMinecraftInfo(run.id, size).then(setInfo).catch(() => setInfo(null));
  }, [run.id, size]);
  if (!info) return <div className="mc-card"><small>Building…</small></div>;
  const [w, h, l] = info.dims;
  const fov = Math.round(info.fov_deg);
  return (
    <div className="mc-card">
      <strong>Minecraft build</strong>
      <small>{w} × {h} × {l} blocks · {info.blocks.toLocaleString()} placed</small>
      <a className="button primary" href={api.schemUrl(run.id, size)} download>Download .schem</a>
      <details>
        <summary>How to line it up in game</summary>
        <ol>
          <li>Put the file in <code>.minecraft/config/worldedit/schematics/</code> (WorldEdit mod) or the server's <code>plugins/WorldEdit/schematics/</code>.</li>
          <li>Stand where the photo was taken from and face south: <code>/tp @s ~ ~ ~ 0 0</code></li>
          <li><code>//schem load &lt;file name&gt;</code>, then <code>//paste</code>. The photo's camera lands at your eyes.</li>
          <li>Options → FOV: <strong>{fov}</strong>{fov < 30 || fov > 110 ? " (outside Minecraft's 30–110 range; use the closest)" : ""}, FOV Effects off, F1 to hide the HUD.</li>
        </ol>
      </details>
    </div>
  );
}

function Strip({ run, showTears, blockSize }: { run: api.RunMeta; showTears: boolean; blockSize: number }) {
  const [tears, setTears] = useState<api.Tear[]>([]);
  const [offAxis, setOffAxis] = useState<string[] | null>(null);
  const [rendering, setRendering] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
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
  const label = modelLabel(run);
  const items: LightboxItem[] = [
    { src: api.inputUrl(run.id), caption: `${label} · input` },
    ...maps.map((k) => ({ src: api.mapUrl(run.id, k), caption: `${label} · ${k}` })),
    ...(offAxis ?? []).map((u) => ({ src: u, caption: `${label} · ${u.match(/yaw_([+-]\d+)/)?.[1]}°` })),
  ];
  const tile = (i: number, caption: string, extra?: ReactNode) => (
    <figure key={items[i].src} className="tile">
      <button className="thumb" onClick={() => setOpen(i)} aria-label={`Enlarge ${items[i].caption}`}>
        <img src={items[i].src} alt="" />
        {extra}
      </button>
      <figcaption>{caption}</figcaption>
    </figure>
  );
  return (
    <div className="strip">
      {tile(0, `${label} · input`,
        showTears &&
          tears.map((t, i) => (
            <span key={i} className="tear" style={{ left: `${(100 * (t.u + 0.5)) / run.width}%`, top: `${(100 * (t.v + 0.5)) / run.height}%` }}>
              {i + 1}
            </span>
          )),
      )}
      {maps.map((k, i) => tile(1 + i, k))}
      {showTears && tears.length > 0 && (
        <ol className="tear-list" aria-label="Top tears by pixel">
          {tears.map((t, i) => <li key={i}>({t.u}, {t.v}) <small>{t.score.toPrecision(3)}</small></li>)}
        </ol>
      )}
      {hasDepth && (
        offAxis ? (
          offAxis.map((u, i) => tile(1 + maps.length + i, `${u.match(/yaw_([+-]\d+)/)?.[1]}°`))
        ) : (
          <button className="offaxis" onClick={renderOffAxis} disabled={rendering}>
            {rendering ? "Rendering…" : "Off-axis renders"}
          </button>
        )
      )}
      {hasDepth && <MinecraftCard run={run} size={blockSize} />}
      {open !== null && <Lightbox items={items} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />}
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
  const [mcSize, setMcSize] = useState(128);
  const [showBlocks, setShowBlocks] = useState(false);
  const [nav, setNavState] = useState<NavSettings>(loadNav);
  const setNav = (patch: Partial<NavSettings>) =>
    setNavState((n) => {
      const next = { ...n, ...patch };
      try {
        localStorage.setItem(NAV_KEY, JSON.stringify(next));
      } catch {}
      return next;
    });
  // A fresh sync object per selection, so panels from an old grid don't keep listening.
  const sync = useMemo(() => new OrbitSync(), [selected.join()]);

  const refreshRuns = useCallback(() => api.getRuns().then(setRuns).catch((e) => setError(String(e))), []);

  useEffect(() => {
    api.getModels().then((m) => {
      const sorted = [...m].sort((a, b) => Number(b.name === RECOMMENDED_MODEL) - Number(a.name === RECOMMENDED_MODEL));
      setModels(sorted);
      setPicked(sorted.slice(0, 1).map((x) => x.name));
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
        <h1 className="brand">BlockShot</h1>
        <section
          className={`drop ${file ? "has-file" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); setFile(e.dataTransfer.files[0] ?? null); }}
        >
          <label>
            <input type="file" accept="image/*" className="visually-hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <Icon d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
            <span>{file ? file.name : "Drop an image, or click to choose"}</span>
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
              <span className="model-name">
                {m.name}
                {m.name === RECOMMENDED_MODEL && <span className="badge">Recommended</span>}
              </span>
              <small>{m.scale_kind}{m.loaded ? " · loaded" : ""}</small>
            </label>
          ))}
        </fieldset>
        <button className="primary" disabled={!file || !picked.length || busy} onClick={run}>
          {busy ? "Running…" : "Run"}
        </button>
        {error && <p className="error">{error}</p>}

        <header className="history-head">
          <h2>History</h2>
          <small>Tick up to {MAX_PANELS} to compare</small>
        </header>
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
                <img src={api.inputUrl(r.id)} alt="" loading="lazy" decoding="async" />
                <span>
                  <span className="name" title={r.source ?? r.id}>{r.source ?? r.id}</span>
                  <small>{modelLabel(r)} · {r.created.replace("T", " ")}</small>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main>
        <div className="toolbar">
          <div className="segmented" role="radiogroup" aria-label="Color mode">
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
          <span className="divider" aria-hidden="true" />
          <Slider label="Point size" min={1} max={8} step={0.5} value={pointSize} unit="px" onChange={setPointSize} />
          <span className="divider" aria-hidden="true" />
          <div className="group">
            <button onClick={() => setSnapKey((k) => k + 1)} disabled={!selected.length}>Snap to input camera</button>
            <label className="check">
              <input type="checkbox" checked={showTears} onChange={(e) => setShowTears(e.target.checked)} />
              Tears
            </label>
          </div>
        </div>
        <div className="toolbar nav" aria-label="Navigation">
          <span className="hint" title="Click or hover a viewer first. In a comparison grid the panel under the mouse leads.">
            <kbd>WASD</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up · <kbd>Shift</kbd> faster
          </span>
          <div className="group sliders">
            <Slider label="Move" min={0.05} max={3} step={0.05} value={nav.move} onChange={(move) => setNav({ move })} />
            <Slider label="Orbit" min={0.1} max={3} step={0.1} value={nav.rotate} onChange={(rotate) => setNav({ rotate })} />
            <Slider label="Zoom" min={0.1} max={3} step={0.1} value={nav.zoom} onChange={(zoom) => setNav({ zoom })} />
            <Slider label="Pan" min={0.1} max={3} step={0.1} value={nav.pan} onChange={(pan) => setNav({ pan })} />
            <button className="quiet" onClick={() => setNav(DEFAULT_NAV)}>Reset</button>
          </div>
        </div>

        <div className={`stage grid-${selected.length}`}>
          {selected.length ? (
            selected.map((id) => (
              <Panel
                key={id}
                id={id}
                blockSize={showBlocks ? mcSize : undefined}
                mode={mode}
                pointSize={pointSize}
                snapKey={snapKey}
                sync={selected.length > 1 ? sync : undefined}
                nav={nav}
                titled={selected.length > 1}
                showTears={showTears}
              />
            ))
          ) : (
            <p className="empty">Pick a run from History, or drop in an image and press Run.</p>
          )}
        </div>

        {selectedRuns.length > 0 && (
          <div className="strips">
            <div className="strip-bar" aria-label="Minecraft build">
              <strong>Minecraft build</strong>
              <span className="hint">looks like the image from one spot: the original camera</span>
              <Slider label="Size" min={32} max={256} step={16} value={mcSize} unit=" blocks" onChange={setMcSize} />
              <label className="check">
                <input type="checkbox" checked={showBlocks} onChange={(e) => setShowBlocks(e.target.checked)} />
                Show as blocks
              </label>
            </div>
            {selectedRuns.map((r) => <Strip key={r.id} run={r} showTears={showTears} blockSize={mcSize} />)}
          </div>
        )}
      </main>
    </div>
  );
}
