import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as api from "./api";
import { COLOR_MODES, FIELD_MODES, type ColorMode } from "./colors";
import Viewer, { OrbitSync, type NavSettings } from "./Viewer";

const MAX_PANELS = 4;
const RECOMMENDED_MODEL = "moge"; // the best-looking builds in practice; listed first and picked by default
const modelLabel = (r: api.RunMeta) => String(r.model).split("/").pop();

const DEFAULT_NAV: NavSettings = { move: 0.5, rotate: 1, zoom: 1 };
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
  build?: api.BuildSettings; // when set, show the Minecraft build instead of the points
  frame: number; // scenes: the frame whose camera Snap goes to and the build is seen from
  onFrame: (i: number) => void;
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
      .then(([m, c]) => {
        setMeta(m);
        setCloud(c);
        if (!api.isScene(m)) api.getAnalysis(props.id).then(setAnalysis).catch(() => {});
      })
      .catch((e) => setError(String(e)));
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
  const blocksKey = `${props.id}:${props.build?.mode}:${props.build?.value}:${props.frame}`;
  useEffect(() => {
    if (!props.build || !buildable || blocks?.key === blocksKey) return;
    const abort = new AbortController();
    api.getBlocks(props.id, props.build, props.frame, abort.signal)
      .then((data) => setBlocks({ key: blocksKey, data }))
      .catch(() => !abort.signal.aborted && setBlocks({ key: blocksKey, data: null }));
    return () => abort.abort();
  }, [props.build, buildable, blocksKey, blocks?.key, props.id, props.frame]);
  const shownBlocks = props.build && blocks?.key === blocksKey ? blocks.data : null;

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
      {meta && api.isScene(meta) && (
        <div className="panel-head">
          <Slider label="View from frame" min={0} max={(meta.frames ?? 1) - 1} step={1} value={props.frame} unit={` / ${(meta.frames ?? 1) - 1}`} onChange={props.onFrame} />
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
          frame={props.frame}
        />
      ) : (
        <p className="empty">{error ?? "Loading…"}</p>
      )}
    </div>
  );
}

type LightboxItem = { src: string; caption: string };

/** Progress of a background job: stage, percent when measurable, elapsed time. */
function ProgressBar({ job }: { job: api.JobState }) {
  const pct = job.total > 0 ? Math.min(100, (100 * job.done) / job.total) : null;
  return (
    <div className="progress" role="progressbar" aria-label={job.stage} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? undefined}>
      <div className="progress-track">
        <div className="progress-fill" style={{ transform: `scaleX(${(pct ?? 100) / 100})` }} data-indeterminate={pct === null || undefined} />
      </div>
      <div className="progress-text">
        <span>{job.stage}{job.message ? ` · ${job.message}` : ""}</span>
        <span>{pct !== null && `${Math.round(pct)}% · `}{Math.round(job.elapsed)}s</span>
      </div>
    </div>
  );
}

const startingJob = (kind: string): api.JobState => ({
  id: "", kind, state: "running", stage: "starting", done: 0, total: 0, message: "", result: null, error: null, elapsed: 0,
});

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
function MinecraftCard({ run, settings, frame }: { run: api.RunMeta; settings: api.BuildSettings; frame: number }) {
  const [info, setInfo] = useState<api.MinecraftInfo | null>(null);
  const [building, setBuilding] = useState<api.JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState<api.JobState | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);
  useEffect(() => {
    const abort = new AbortController();
    const live = () => !abort.signal.aborted;
    setInfo(null);
    setError(null);
    setZipUrl(null);
    setBuilding(startingJob("minecraft"));
    api.buildMinecraft(run.id, settings, frame, (j) => live() && setBuilding(j), abort.signal)
      .then((i) => live() && setInfo(i))
      .catch((e) => live() && setError(String(e)))
      .finally(() => live() && setBuilding(null));
    return () => abort.abort();
  }, [run.id, settings.mode, settings.value, frame]);
  const exportZip = () => {
    setExporting(startingJob("tiles"));
    api.exportTiles(run.id, settings, frame, setExporting)
      .then(setZipUrl)
      .catch((e) => setError(String(e)))
      .finally(() => setExporting(null));
  };
  if (building) return <div className="mc-card"><strong>Minecraft build</strong><ProgressBar job={building} /></div>;
  if (error || !info) return <div className="mc-card"><strong>Minecraft build</strong><small className="error">{error ?? "No build"}</small></div>;
  const [w, h, l] = info.dims;
  const fov = Math.round(info.fov_deg);
  const pitch = Math.round(info.pitch_deg);
  const yaw = Math.round(info.yaw_deg);
  const size = info.block_size_unit === "m" ? `${info.block_size < 1 ? `${Math.round(info.block_size * 100)} cm` : `${info.block_size.toFixed(2)} m`}` : `${info.block_size.toPrecision(2)} units`;
  return (
    <div className="mc-card">
      <strong>Minecraft build</strong>
      <small>{w} × {h} × {l} blocks · {info.blocks.toLocaleString()} placed · 1 block = {size}</small>
      {info.budget_limited && <small className="warn">Blocks were enlarged to stay under the block budget.</small>}
      {info.too_tall && <small className="warn">Taller than Minecraft's 384-block build height: the top or bottom will be cut off.</small>}
      {info.preview_lod > 1 && <small>3D preview merges {info.preview_lod}³ blocks per cube; downloads are full detail.</small>}
      {info.tiles <= 1 && info.single_file ? (
        <a className="button primary" href={api.schemUrl(run.id, settings, frame)} download>Download .schem</a>
      ) : exporting ? (
        <ProgressBar job={exporting} />
      ) : zipUrl ? (
        <a className="button primary" href={zipUrl} download>Download {info.tiles} tiles (.zip)</a>
      ) : (
        <button className="primary" onClick={exportZip}>Prepare {info.tiles} tiles (.zip)</button>
      )}
      <details>
        <summary>How to line it up in game (1.20+)</summary>
        <ol>
          <li>Put the {info.tiles > 1 ? "tiles" : "file"} in <code>.minecraft/config/worldedit/schematics/</code> (WorldEdit mod) or the server's <code>plugins/WorldEdit/schematics/</code>.</li>
          <li>Stand where {api.isScene(run) ? `frame ${frame} was filmed` : "the picture was taken"} from, looking the same way: <code>/tp @s ~ ~ ~ {yaw} {pitch}</code></li>
          <li>
            <code>//schem load &lt;name&gt;</code>, then <code>//paste</code>
            {info.tiles > 1 ? " for every tile, without moving in between (the zip has the list)." : "."} The picture's camera lands at your eyes.
          </li>
          <li>Options → FOV: <strong>{fov}</strong>{fov < 30 || fov > 110 ? " (outside Minecraft's 30–110 range; use the closest)" : ""}, FOV Effects off, F1 to hide the HUD.</li>
        </ol>
      </details>
    </div>
  );
}

const AUTO_STEPS = [50, 100, 200, 300, 500, 750, 1000, 2000, 4000]; // thousand blocks

/** Mode select plus the one slider that mode needs. */
function BuildControls({ value, onChange }: { value: api.BuildSettings; onChange: (b: api.BuildSettings) => void }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  // Builds take seconds, so only commit when the slider is released.
  const commit = () => onChange(draft);
  const slider = (label: string, min: number, max: number, step: number, shown: string, v: number, set: (x: number) => void) => (
    <label className="slider">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => set(+e.target.value)} onPointerUp={commit} onKeyUp={commit} />
      <span className="val">{shown}</span>
    </label>
  );
  const idx = Math.max(0, AUTO_STEPS.indexOf(draft.value));
  return (
    <>
      <label className="slider">
        <span>Block size</span>
        <select
          value={value.mode}
          onChange={(e) => {
            const mode = e.target.value as api.BuildMode;
            onChange({ mode, value: { auto: 300, meters: 0.25, fit: 128 }[mode] });
          }}
        >
          <option value="auto">Auto (by block count)</option>
          <option value="meters">Real size</option>
          <option value="fit">Fit longest side</option>
        </select>
      </label>
      {draft.mode === "auto" && slider("About", 0, AUTO_STEPS.length - 1, 1, `${AUTO_STEPS[idx] >= 1000 ? `${AUTO_STEPS[idx] / 1000}M` : `${AUTO_STEPS[idx]}k`} blocks`, idx, (i) => setDraft({ mode: "auto", value: AUTO_STEPS[i] }))}
      {draft.mode === "meters" && slider("1 block =", 0.05, 2, 0.05, `${draft.value.toFixed(2)} m`, draft.value, (v) => setDraft({ mode: "meters", value: v }))}
      {draft.mode === "fit" && slider("Longest side", 32, 1024, 32, `${draft.value} blocks`, draft.value, (v) => setDraft({ mode: "fit", value: v }))}
    </>
  );
}

/** A video scene's sampled frames: click one to view the scene (and the build) from it. */
function SceneStrip({ run, build, frame, onFrame }: { run: api.RunMeta; build: api.BuildSettings; frame: number; onFrame: (i: number) => void }) {
  const [open, setOpen] = useState<number | null>(null);
  const items: LightboxItem[] = Array.from({ length: run.frames ?? 0 }, (_, i) => ({
    src: api.frameUrl(run.id, i),
    caption: `${run.source ?? "video"} · frame ${i}`,
  }));
  return (
    <div className="strip">
      <MinecraftCard run={run} settings={build} frame={frame} />
      {items.map((it, i) => (
        <figure key={it.src} className={`tile ${i === frame ? "current" : ""}`}>
          <button className="thumb" onClick={() => onFrame(i)} onDoubleClick={() => setOpen(i)} aria-label={`View from frame ${i}`} title="Click: view from here · double-click: enlarge">
            <img src={it.src} alt="" loading="lazy" />
          </button>
          <figcaption>{i}</figcaption>
        </figure>
      ))}
      {open !== null && <Lightbox items={items} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />}
    </div>
  );
}

function Strip({ run, showTears, build }: { run: api.RunMeta; showTears: boolean; build: api.BuildSettings }) {
  const [tears, setTears] = useState<api.Tear[]>([]);
  const [offAxis, setOffAxis] = useState<string[] | null>(null);
  const [rendering, setRendering] = useState<api.JobState | null>(null);
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
    setRendering(startingJob("offaxis"));
    api.renderOffAxis(run.id, setRendering).then(setOffAxis).finally(() => setRendering(null));
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
      {hasDepth && <MinecraftCard run={run} settings={build} frame={0} />}
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
        ) : rendering ? (
          <div className="offaxis"><ProgressBar job={rendering} /></div>
        ) : (
          <button className="offaxis" onClick={renderOffAxis}>Off-axis renders</button>
        )
      )}
      {open !== null && <Lightbox items={items} index={open} onIndex={setOpen} onClose={() => setOpen(null)} />}
    </div>
  );
}

export default function App() {
  const [models, setModels] = useState<api.ModelInfo[]>([]);
  const [picked, setPicked] = useState<string[]>([]);
  const [runs, setRuns] = useState<api.RunMeta[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState<api.JobState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string[]>([]);
  const [mode, setMode] = useState<ColorMode>("rgb");
  const [pointSize, setPointSize] = useState(2);
  const [snapKey, setSnapKey] = useState(0);
  const [showTears, setShowTears] = useState(false);
  const [build, setBuild] = useState<api.BuildSettings>({ mode: "auto", value: 300 });
  const [showBlocks, setShowBlocks] = useState(false);
  const [frames, setFrames] = useState<Record<string, number>>({}); // scene id -> frame viewed from
  const frameOf = (id: string) => frames[id] ?? 0;
  const setFrame = (id: string) => (i: number) => setFrames((f) => ({ ...f, [id]: i }));
  const [video, setVideo] = useState<api.VideoSettings>({ fps: 2, max_frames: 60, hud: true, geometry: "moge" });
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

  const isVideo = !!file && (file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi)$/i.test(file.name));
  const run = async () => {
    if (!file || (!isVideo && !picked.length)) return;
    setRunning(startingJob(isVideo ? "video" : "predict"));
    setError(null);
    try {
      if (isVideo) {
        const scene = await api.processVideo(file, video, setRunning);
        await refreshRuns();
        setSelected([scene.id]);
        return;
      }
      const out = await api.predict(file, picked, setRunning);
      await refreshRuns();
      setSelected(out.slice(0, MAX_PANELS).map((r) => r.id));
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(null);
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
            <input type="file" accept="image/*,video/*" className="visually-hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <Icon d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
            <span>{file ? file.name : "Drop an image or a video, or click to choose"}</span>
          </label>
        </section>
        {isVideo ? (
          <fieldset className="models video-opts">
            <legend>Video → scene</legend>
            <Slider label="Frames/s" min={0.5} max={6} step={0.5} value={video.fps} unit="" onChange={(fps) => setVideo((v) => ({ ...v, fps }))} />
            <Slider label="Max frames" min={8} max={200} step={4} value={video.max_frames} unit="" onChange={(max_frames) => setVideo((v) => ({ ...v, max_frames }))} />
            <label className="slider">
              <span>Geometry</span>
              <select value={video.geometry} onChange={(e) => setVideo((v) => ({ ...v, geometry: e.target.value as api.VideoSettings["geometry"] }))}>
                <option value="moge">MoGe depth (best, metres)</option>
                <option value="vggt">VGGT depth (faster)</option>
              </select>
            </label>
            <label>
              <input type="checkbox" checked={video.hud} onChange={(e) => setVideo((v) => ({ ...v, hud: e.target.checked }))} />
              <span>Ignore a static game HUD</span>
            </label>
            <small>Move slowly and keep things in view between frames. Each frame takes about a second.</small>
          </fieldset>
        ) : (
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
        )}
        {running ? (
          <ProgressBar job={running} />
        ) : (
          <button className="primary" disabled={!file || (!isVideo && !picked.length)} onClick={run}>Run</button>
        )}
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
                  <small>
                    {api.isScene(r) ? <span className="badge">video · {r.frames} frames</span> : modelLabel(r)} · {r.created.replace("T", " ")}
                  </small>
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
            Drag to look · <kbd>WASD</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up · scroll forward/back · <kbd>Shift</kbd> faster
          </span>
          <div className="group sliders">
            <Slider label="Move" min={0.05} max={3} step={0.05} value={nav.move} onChange={(move) => setNav({ move })} />
            <Slider label="Look" min={0.1} max={3} step={0.1} value={nav.rotate} onChange={(rotate) => setNav({ rotate })} />
            <Slider label="Scroll" min={0.1} max={3} step={0.1} value={nav.zoom} onChange={(zoom) => setNav({ zoom })} />
            <button className="quiet" onClick={() => setNav(DEFAULT_NAV)}>Reset</button>
          </div>
        </div>

        <div className={`stage grid-${selected.length}`}>
          {selected.length ? (
            selected.map((id) => (
              <Panel
                key={id}
                id={id}
                build={showBlocks ? build : undefined}
                mode={mode}
                pointSize={pointSize}
                snapKey={snapKey}
                sync={selected.length > 1 ? sync : undefined}
                nav={nav}
                titled={selected.length > 1}
                showTears={showTears}
                frame={frameOf(id)}
                onFrame={setFrame(id)}
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
              <span className="hint">looks like the image from one spot</span>
              <BuildControls value={build} onChange={setBuild} />
              <label className="check">
                <input type="checkbox" checked={showBlocks} onChange={(e) => setShowBlocks(e.target.checked)} />
                Show as blocks
              </label>
            </div>
            {selectedRuns.map((r) =>
              api.isScene(r) ? (
                <SceneStrip key={r.id} run={r} build={build} frame={frameOf(r.id)} onFrame={setFrame(r.id)} />
              ) : (
                <Strip key={r.id} run={r} showTears={showTears} build={build} />
              ),
            )}
          </div>
        )}
      </main>
    </div>
  );
}
