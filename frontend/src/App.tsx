import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as api from "./api";
import { COLOR_MODES, FIELD_MODES, type ColorMode } from "./colors";
import Playground, {
  MATERIALS, PRESETS, SHAPES, adoptIds, shuffled,
  type AutopilotKind, type Material, type Nuisances, type PgScene, type Shape, type Tool, type Truth,
} from "./Playground";
import Viewer, { OrbitSync, type NavSettings } from "./Viewer";

const MAX_PANELS = 4;
const RECOMMENDED_MODEL = "moge"; // the best-looking builds in practice; listed first and picked by default
const modelLabel = (r: api.RunMeta) => String(r.model).split("/").pop();
/** Short model name from the run id (`<sha>_<model>_<version>`): "moge", "depth anything", ... */
const shortModel = (r: api.RunMeta) => r.id.split("_").slice(1, -1).join(" ");
const runName = (r: api.RunMeta) => r.source ?? r.id;
/** Long file names differ at the end (camera timestamps), so shorten them in the middle. */
const shortName = (name: string, max = 24) => (name.length <= max ? name : `${name.slice(0, max - 13)}…${name.slice(-12)}`);
const when = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const runDetail = (r: api.RunMeta) => (api.isScene(r) ? `video · ${r.frames} frames` : shortModel(r)); // playground runs are named playground_…

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
  const stats = analysis
    ? [
        analysis.reprojection_px !== null && `reproj ${analysis.reprojection_px.toFixed(3)} px`,
        integ?.native && `native residual ${pct(integ.native.residual_fraction)}`,
        integ?.derived && `depth residual ${pct(integ.derived.residual_fraction)}`,
      ].filter(Boolean).join(" · ")
    : "";
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
          {(props.titled || stats || unavailable) && (
            <div className="chip info">
              {props.titled && (
                <span className="chip-title">
                  <strong title={runName(meta)}>{api.isScene(meta) ? shortName(runName(meta), 32) : modelLabel(meta)}</strong>
                  <small>{meta.scale_kind}</small>
                </span>
              )}
              {stats && <small className="stats">{stats}</small>}
              {unavailable && <small className="warn">No {props.mode} for this model</small>}
            </div>
          )}
          {api.isScene(meta) && (
            <div className="chip">
              <Slider label="View from frame" min={0} max={(meta.frames ?? 1) - 1} step={1} value={props.frame} unit={` / ${(meta.frames ?? 1) - 1}`} onChange={props.onFrame} />
            </div>
          )}
          {props.titled && <button className="chip snap" onClick={() => setLocalSnap((k) => k + 1)}>Snap</button>}
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

/** A settings menu under a button; closes on Escape or a click outside it. */
function Popover({ label, children }: { label: string; children: ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const away = (e: Event) => {
      const d = ref.current;
      if (d?.open && !d.contains(e.target as Node)) d.open = false;
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape" && ref.current?.open) ref.current.open = false;
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", esc);
    };
  }, []);
  return (
    <details className="popover" ref={ref}>
      <summary>
        <Icon d="M4 6h9m4 0h3M4 12h3m4 0h9M4 18h11m4 0h1M15 4v4M9 10v4M17 16v4" />
        {label}
      </summary>
      <div className="popover-body">{children}</div>
    </details>
  );
}

type LightboxItem = { src: string; caption: string };

/** Progress of a background job: stage, percent when measurable, elapsed time. */
function ProgressBar({ job }: { job: api.JobState }) {
  // A one-step stage ("0 of 1", e.g. one VGGT pass) has nothing to count: show it as working, not stuck at 0%.
  const pct = job.total > 1 ? Math.min(100, (100 * job.done) / job.total) : null;
  return (
    <div className="progress" role="progressbar" aria-label={job.stage} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? undefined}>
      <div className="progress-track">
        <div className="progress-fill" style={{ transform: `scaleX(${(pct ?? 100) / 100})` }} data-indeterminate={pct === null || undefined} />
      </div>
      <div className="progress-text">
        <span title={`${job.stage}${job.message ? ` · ${job.message}` : ""}`}>{job.stage}{job.message ? ` · ${job.message}` : ""}</span>
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
  }, [run.id, run.above, settings.mode, settings.value, frame]);
  const exportZip = () => {
    setExporting(startingJob("tiles"));
    api.exportTiles(run.id, settings, frame, setExporting)
      .then(setZipUrl)
      .catch((e) => setError(String(e)))
      .finally(() => setExporting(null));
  };
  const head = (
    <div className="mc-head">
      <strong title={runName(run)}>{shortName(runName(run), 32)}</strong>
      <small>{api.isScene(run) ? `frame ${frame}` : shortModel(run)}</small>
    </div>
  );
  if (building) return <div className="mc-card">{head}<ProgressBar job={building} /></div>;
  if (error || !info) return <div className="mc-card">{head}<small className="error">{error ?? "No build"}</small></div>;
  const [w, h, l] = info.dims;
  const fov = Math.round(info.fov_deg);
  const pitch = Math.round(info.pitch_deg);
  const yaw = Math.round(info.yaw_deg);
  const size = info.block_size_unit === "m" ? `${info.block_size < 1 ? `${Math.round(info.block_size * 100)} cm` : `${info.block_size.toFixed(2)} m`}` : `${info.block_size.toPrecision(2)} units`;
  return (
    <div className="mc-card">
      {head}
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

function VideoOptions({ value, onChange, legend }: { value: api.VideoSettings; onChange: (v: api.VideoSettings) => void; legend: string }) {
  return (
    <fieldset className="models video-opts">
      <legend>{legend}</legend>
      <Slider label="Frames/s" min={0.5} max={6} step={0.5} value={value.fps} unit="" onChange={(fps) => onChange({ ...value, fps })} />
      <Slider label="Max frames" min={8} max={200} step={4} value={value.max_frames} unit="" onChange={(max_frames) => onChange({ ...value, max_frames })} />
      <label className="slider">
        <span>Geometry</span>
        <select value={value.geometry} onChange={(e) => onChange({ ...value, geometry: e.target.value as api.VideoSettings["geometry"] })}>
          <option value="vggt">VGGT (recommended)</option>
          <option value="moge">MoGe (sharper, can ghost)</option>
        </select>
      </label>
      <label className="option">
        <input type="checkbox" checked={value.hud} onChange={(e) => onChange({ ...value, hud: e.target.checked })} />
        <span>Ignore a static game HUD</span>
      </label>
    </fieldset>
  );
}

function ModelPicker({ models, picked, onPicked, legend }: { models: api.ModelInfo[]; picked: string[]; onPicked: (p: string[]) => void; legend: string }) {
  return (
    <fieldset className="models">
      <legend>{legend}</legend>
      {models.map((m) => (
        <label key={m.name}>
          <input
            type="checkbox"
            checked={picked.includes(m.name)}
            onChange={(e) => onPicked(e.target.checked ? [...picked, m.name] : picked.filter((x) => x !== m.name))}
          />
          <span className="model-name">
            {m.name}
            {m.name === RECOMMENDED_MODEL && <span className="badge">Recommended</span>}
          </span>
          <small>{m.scale_kind}{m.loaded ? " · loaded" : ""}</small>
        </label>
      ))}
    </fieldset>
  );
}

const fmtMetric = (m: api.ScoreMetric) =>
  m.unit === "rel" ? `${(m.value * 100).toFixed(1)}%`
  : m.unit === "m" ? (m.value < 1 ? `${(m.value * 100).toFixed(1)} cm` : `${m.value.toFixed(2)} m`)
  : m.unit === "°" ? `${m.value.toFixed(1)}°`
  : m.unit === "×" ? `${m.value.toFixed(2)}×`
  : `${Math.round(m.value)} ${m.unit}`;

/** A playground run's scores against its ground truth. */
function ScoreList({ id, compact }: { id: string; compact?: boolean }) {
  const [score, setScore] = useState<api.Score | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setScore(null);
    setError(null);
    api.getScore(id).then(setScore).catch((e) => setError(String(e).replace(/^Error: \d+ /, "")));
  }, [id]);
  if (error) return <small className="error">{error}</small>;
  if (!score) return <small>Scoring against the ground truth…</small>;
  const metrics = compact ? score.metrics.filter((m) => m.verdict !== null) : score.metrics;
  return (
    <ul className="scores">
      {metrics.map((m) => (
        <li key={m.key} title={m.note}>
          <span className={`dot ${m.verdict ?? "info"}`} aria-hidden="true" />
          <span>{m.label}</span>
          <strong>{fmtMetric(m)}</strong>
        </li>
      ))}
    </ul>
  );
}

function ScoreCard({ run }: { run: api.RunMeta }) {
  return (
    <div className="mc-card score-card">
      <div className="mc-head">
        <strong>Ground truth</strong>
        <small>playground · {run.playground_scene ?? "scene"}</small>
      </div>
      <ScoreList id={run.id} />
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
      {draft.mode === "auto" && slider("About", 0, AUTO_STEPS.length - 1, 1, `${AUTO_STEPS[idx] >= 1000 ? `${AUTO_STEPS[idx] / 1000}M` : `${AUTO_STEPS[idx]}k`} blocks`, idx, (i) => setDraft({ ...draft, value: AUTO_STEPS[i] }))}
      {draft.mode === "meters" && slider("1 block =", 0.05, 2, 0.05, `${draft.value.toFixed(2)} m`, draft.value, (v) => setDraft({ ...draft, value: v }))}
      {draft.mode === "fit" && slider("Longest side", 32, 1024, 32, `${draft.value} blocks`, draft.value, (v) => setDraft({ ...draft, value: v }))}
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
      {run.playground && <ScoreCard run={run} />}
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
      {run.playground && <ScoreCard run={run} />}
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

const PG_KEY = "playground-scene";
const SHAPE_KEYS: Record<Shape, string> = { box: "1", pyramid: "2", sphere: "3", cylinder: "4" };

function PlaygroundPanel(p: {
  scene: PgScene; onScene: (s: PgScene) => void; tool: Tool; onTool: (t: Tool) => void;
  nuisances: Nuisances; onNuisances: (n: Nuisances) => void; onAutopilot: (k: AutopilotKind) => void; recordingBusy: boolean;
  video: api.VideoSettings; onVideo: (v: api.VideoSettings) => void; models: api.ModelInfo[]; picked: string[]; onPicked: (x: string[]) => void;
}) {
  const nuisance = (key: keyof Nuisances, label: string, note: string) => (
    <label className="option">
      <input type="checkbox" checked={p.nuisances[key]} onChange={(e) => p.onNuisances({ ...p.nuisances, [key]: e.target.checked })} />
      <span>{label}<small>{note}</small></span>
    </label>
  );
  return (
    <div className="pg-panel">
      <fieldset className="models">
        <legend>Scene</legend>
        <label className="slider pg-row">
          <span>Preset</span>
          <select value={Object.keys(PRESETS).includes(p.scene.name) ? p.scene.name : ""} onChange={(e) => e.target.value && p.onScene(PRESETS[e.target.value]())}>
            {!Object.keys(PRESETS).includes(p.scene.name) && <option value="">{p.scene.name}</option>}
            {Object.keys(PRESETS).map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <div className="pg-buttons">
          <button onClick={() => p.onScene(shuffled(Math.floor(Math.random() * 10000)))}>Shuffle</button>
          <button onClick={() => p.onScene(PRESETS[Object.keys(PRESETS).includes(p.scene.name) ? p.scene.name : "Courtyard"]())}>Reset</button>
        </div>
        <small>{p.scene.objects.length} objects</small>
      </fieldset>

      <fieldset className="models">
        <legend>Build</legend>
        <div className="segmented pg-shapes" role="radiogroup" aria-label="Shape">
          {SHAPES.map((s) => (
            <button key={s} role="radio" aria-checked={p.tool.shape === s} className={p.tool.shape === s ? "active" : ""} onClick={() => p.onTool({ ...p.tool, shape: s })} title={`Key ${SHAPE_KEYS[s]}`}>
              {s}
            </button>
          ))}
        </div>
        <Slider label="Size" min={0.5} max={12} step={0.5} value={p.tool.size} unit=" m" onChange={(size) => p.onTool({ ...p.tool, size })} />
        <label className="slider pg-row">
          <span>Material</span>
          <select value={p.tool.material} onChange={(e) => p.onTool({ ...p.tool, material: e.target.value as Material })}>
            {MATERIALS.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <small className="keys-list">
          <kbd>F</kbd> place · <kbd>X</kbd> delete · <kbd>1</kbd>–<kbd>4</kbd> shape · <kbd>[</kbd><kbd>]</kbd> size · <kbd>T</kbd> material · <kbd>Y</kbd> rotate ({Math.round((p.tool.rot * 180) / Math.PI)}°)
        </small>
      </fieldset>

      <fieldset className="models">
        <legend>Record</legend>
        <small className="keys-list"><kbd>R</kbd> start / stop recording · <kbd>P</kbd> snapshot</small>
        <div className="pg-buttons">
          <button disabled={p.recordingBusy} onClick={() => p.onAutopilot("loop")} title="Walk a circle around the scene, facing its middle">Autopilot: loop</button>
          <button disabled={p.recordingBusy} onClick={() => p.onAutopilot("turn")} title="Turn on the spot">Turn</button>
          <button disabled={p.recordingBusy} onClick={() => p.onAutopilot("zigzag")} title="Walk forward with quick turns">Zig-zag</button>
        </div>
      </fieldset>

      <fieldset className="models">
        <legend>Game nuisances (recorded)</legend>
        {nuisance("hud", "HUD", "Panels and a changing health bar")}
        {nuisance("arms", "Arms", "Your character's arms, swinging as you walk")}
        {nuisance("bob", "Head bob", "The camera bobs while walking")}
        {nuisance("fast", "Fast turns", "Mouse look three times as sensitive")}
      </fieldset>

      <VideoOptions value={p.video} onChange={p.onVideo} legend="Recordings go to" />
      <ModelPicker models={p.models} picked={p.picked} onPicked={p.onPicked} legend="Snapshots go to" />
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
  const [above, setAbove] = useState(false); // the next picture is a top-down map
  const [reloads, setReloads] = useState(0); // bumped after a recompute: open panels load the new data
  const [stale, setStale] = useState(false); // the backend runs older code than is on disk
  useEffect(() => {
    // A server without /health (404) predates it, so it's older too.
    const check = () => api.getHealth().then((h) => setStale(h.stale)).catch((e) => setStale(String(e).startsWith("Error: 404")));
    check();
    const t = window.setInterval(check, 30_000);
    return () => window.clearInterval(t);
  }, []);
  const [video, setVideo] = useState<api.VideoSettings>({ fps: 2, max_frames: 60, hud: true, geometry: "vggt" });
  // Playground (see Playground.tsx): its scene is kept in this browser.
  const [appMode, setAppMode] = useState<"upload" | "playground">("upload");
  const [pgScene, setPgSceneState] = useState<PgScene>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(PG_KEY) ?? "null");
      if (saved?.objects) return adoptIds(saved);
    } catch {}
    return PRESETS.Courtyard();
  });
  const setPgScene = (s: PgScene) => {
    setPgSceneState(s);
    try {
      localStorage.setItem(PG_KEY, JSON.stringify(s));
    } catch {}
  };
  const [tool, setTool] = useState<Tool>({ shape: "box", size: 1.5, material: "brick", rot: 0 });
  const [nuisances, setNuisances] = useState<Nuisances>({ hud: false, arms: false, bob: false, fast: false });
  const [autopilot, setAutopilot] = useState<{ kind: AutopilotKind; n: number } | null>(null);
  const [pgVideo, setPgVideo] = useState<api.VideoSettings>({ fps: 4, max_frames: 60, hud: true, geometry: "vggt" });
  const [pgResults, setPgResults] = useState<{ id: string; label: string }[]>([]);
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
      const out = await api.predict(file, picked, setRunning, false, above);
      await refreshRuns();
      setSelected(out.slice(0, MAX_PANELS).map((r) => r.id));
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(null);
    }
  };

  const stamp = () => new Date().toTimeString().slice(0, 8).replace(/:/g, "");
  const sendClip = async (blob: Blob, truth: Truth) => {
    setError(null);
    setRunning(startingJob("video"));
    try {
      const name = `playground_${truth.scene.replace(/\W+/g, "-")}_${stamp()}.webm`;
      const scene = await api.processVideo(new File([blob], name, { type: "video/webm" }), pgVideo, setRunning, truth);
      await refreshRuns();
      setPgResults([{ id: scene.id, label: `Recording · ${truth.frames.length ? truth.frames[truth.frames.length - 1].t.toFixed(1) : 0} s` }]);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(null);
    }
  };
  const sendSnapshot = async (png: Blob, truth: Truth) => {
    if (!picked.length) return setError("Tick at least one model under “Snapshots go to”.");
    setError(null);
    setRunning(startingJob("predict"));
    try {
      const name = `playground_${truth.scene.replace(/\W+/g, "-")}_${stamp()}.png`;
      const out = await api.predict(new File([png], name, { type: "image/png" }), picked, setRunning, false, false, truth);
      await refreshRuns();
      setPgResults(out.map((r) => ({ id: r.id, label: `Snapshot · ${shortModel(r.meta)}` })));
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(null);
    }
  };
  const openRun = (id: string) => {
    setAppMode("upload");
    setSelected([id]);
  };

  const remove = async (r: api.RunMeta) => {
    if (!window.confirm(`Delete "${r.source ?? r.id}" (${api.isScene(r) ? "video scene" : modelLabel(r)})? This removes it from disk.`)) return;
    try {
      await api.deleteRun(r.id);
      setSelected((s) => s.filter((x) => x !== r.id));
      await refreshRuns();
    } catch (e) {
      setError(String(e));
    }
  };

  const recompute = async () => {
    const targets = outdatedRuns;
    const scenes = targets.filter(api.isScene).length;
    if (!window.confirm(`Redo ${targets.length} run${targets.length > 1 ? "s" : ""} made by an older version of BlockShot?` +
      (scenes ? ` Video scenes are redone from their stored frames (about a minute each).` : ""))) return;
    setError(null);
    const done: string[] = [];
    try {
      for (const [k, r] of targets.entries()) {
        setRunning(startingJob("recompute"));
        done.push(await api.recomputeRun(r.id, (j) => setRunning({ ...j, stage: `${k + 1}/${targets.length} · ${r.source ?? r.id} · ${j.stage}` })));
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(null);
      await refreshRuns();
      setReloads((k) => k + 1);
      if (done.length) setSelected(done);
    }
  };

  const toggle = (id: string, on: boolean) =>
    setSelected((s) => (on ? [...s.filter((x) => x !== id), id].slice(-MAX_PANELS) : s.filter((x) => x !== id)));
  const selectedRuns = selected.map((id) => runs.find((r) => r.id === id)).filter((r): r is api.RunMeta => !!r);
  // Only runs made by an older version change when recomputed; the rest would come back identical.
  const outdatedRuns = selectedRuns.filter((r) => r.outdated);
  const anyConfidence = selectedRuns.some((r) => r.has_confidence);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="brand">BlockShot</h1>
        <div className="segmented mode-switch" role="tablist" aria-label="Mode">
          {(["upload", "playground"] as const).map((m) => (
            <button key={m} role="tab" aria-selected={appMode === m} className={appMode === m ? "active" : ""} onClick={() => setAppMode(m)}>
              {m === "upload" ? "Upload" : "Playground"}
            </button>
          ))}
        </div>
        {appMode === "playground" ? (
          <PlaygroundPanel
            scene={pgScene} onScene={setPgScene} tool={tool} onTool={setTool} nuisances={nuisances} onNuisances={setNuisances}
            onAutopilot={(kind) => setAutopilot((a) => ({ kind, n: (a?.n ?? 0) + 1 }))} recordingBusy={!!running}
            video={pgVideo} onVideo={setPgVideo} models={models} picked={picked} onPicked={setPicked}
          />
        ) : (
        <>
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
          <>
            <VideoOptions value={video} onChange={setVideo} legend="Video → scene" />
            <small className="note">Move slowly and keep things in view between frames. Each frame takes about a second.</small>
          </>
        ) : (
          <ModelPicker models={models} picked={picked} onPicked={setPicked} legend="Models" />
        )}
        {!isVideo && (
          <label className="option" title="Angled and tilted shots are levelled automatically without it. Applies to the picture's runs, already made ones included.">
            <input type="checkbox" checked={above} onChange={(e) => setAbove(e.target.checked)} />
            <span>
              Top-down map
              <small>The picture looks straight down at the ground</small>
            </span>
          </label>
        )}
        {running ? (
          <ProgressBar job={running} />
        ) : (
          <button className="primary" disabled={!file || (!isVideo && !picked.length)} onClick={run}>Run</button>
        )}
        </>
        )}
        {appMode === "playground" && running && <ProgressBar job={running} />}
        {error && <p className="error">{error}</p>}
        {appMode === "playground" && !running && pgResults.map((r) => (
          <div key={r.id} className="pg-result">
            <div className="pg-result-head">
              <strong>{r.label}</strong>
              <button className="quiet" onClick={() => openRun(r.id)}>Open</button>
            </div>
            <ScoreList id={r.id} compact />
          </div>
        ))}

        <header className="history-head">
          <h2>History</h2>
          {outdatedRuns.length > 0 && !running ? (
            <button className="quiet" onClick={recompute} title="Redo the ticked runs that an older version of BlockShot made">
              Recompute {outdatedRuns.length > 1 ? `${outdatedRuns.length} older runs` : "older run"}
            </button>
          ) : (
            <small>Tick up to {MAX_PANELS} to compare</small>
          )}
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
                  <span className="name" title={runName(r)}>{shortName(runName(r))}</span>
                  <small title={`${r.outdated ? "Made by an older version: tick it and press Recompute to redo it. " : ""}${runDetail(r)} · ${r.created.replace("T", " ")}`}>
                    {r.outdated ? <span className="outdated">older version · </span> : null}
                    {runDetail(r)}
                    {!r.outdated && ` · ${when(r.created)}`/* outdated rows show the tag instead; the date is in the tooltip */}
                  </small>
                </span>
              </button>
              <button className="icon delete" aria-label={`Delete ${r.source ?? r.id}`} title="Delete" onClick={() => remove(r)}>
                <Icon d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6" />
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className={appMode === "playground" ? "pg-mode" : undefined}>
        {stale && (
          <p className="banner" role="status">
            BlockShot was updated while it was running. Close both BlockShot windows and run <code>run.bat</code> again to use the new version.
          </p>
        )}
        {appMode === "playground" ? (
          <Playground
            scene={pgScene} onScene={setPgScene} tool={tool} onTool={setTool} nuisances={nuisances}
            autopilot={autopilot} onAutopilotDone={() => {}} onClip={sendClip} onSnapshot={sendSnapshot}
          />
        ) : (
        <>
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
          <button onClick={() => setSnapKey((k) => k + 1)} disabled={!selected.length}>Snap to input camera</button>
          <label className="check">
            <input type="checkbox" checked={showTears} onChange={(e) => setShowTears(e.target.checked)} />
            Tears
          </label>
          <span className="hint nav-hint" title="Click or hover a viewer first. In a comparison grid the panel under the mouse leads.">
            Drag to look · <kbd>WASD</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up · scroll to move
          </span>
          <Popover label="View">
            <Slider label="Point size" min={1} max={8} step={0.5} value={pointSize} unit="px" onChange={setPointSize} />
            <Slider label="Move speed" min={0.05} max={3} step={0.05} value={nav.move} onChange={(move) => setNav({ move })} />
            <Slider label="Look speed" min={0.1} max={3} step={0.1} value={nav.rotate} onChange={(rotate) => setNav({ rotate })} />
            <Slider label="Scroll step" min={0.1} max={3} step={0.1} value={nav.zoom} onChange={(zoom) => setNav({ zoom })} />
            <button className="quiet" onClick={() => setNav(DEFAULT_NAV)}>Reset speeds</button>
            <p className="keys">
              Drag to look around · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up ·
              scroll forward/back · hold <kbd>Shift</kbd> to go faster. Click or hover a view first.
            </p>
          </Popover>
        </div>

        <div className={`stage grid-${selected.length}`}>
          {selected.length ? (
            selected.map((id) => (
              <Panel
                key={`${id}:${runs.find((r) => r.id === id)?.above ?? false}:${reloads}`} // reload when its "up" or data changes
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
        </>
        )}
      </main>
    </div>
  );
}
