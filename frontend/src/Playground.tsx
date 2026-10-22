import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

/**
 * A small first-person world for controlled tests. Everything is drawn here, so a recording (R) or
 * snapshot (P) carries its exact ground truth: the camera of every frame and every surface. The
 * backend scores runs against it (backend/playground.py, which uses the same shape definitions).
 */

export type Shape = "box" | "pyramid" | "sphere" | "cylinder";
export const SHAPES: Shape[] = ["box", "pyramid", "sphere", "cylinder"];
export const MATERIALS = ["brick", "planks", "checker", "grass", "concrete", "stripes"] as const;
export type Material = (typeof MATERIALS)[number];
export type PgObject = { id: number; shape: Shape; center: [number, number, number]; size: [number, number, number]; rot: number; material: Material };
export type PgScene = { name: string; objects: PgObject[]; spawn: [number, number, number] }; // spawn: x, z, yaw
export type Tool = { shape: Shape; size: number; material: Material; rot: number };
export type Nuisances = { hud: boolean; arms: boolean; bob: boolean; fast: boolean };
export type AutopilotKind = "loop" | "turn" | "zigzag";
export type Truth = {
  version: 1; scene: string; width: number; height: number; fov_deg: number;
  frames: { t: number; p: number[]; q: number[] }[];
  objects: { shape: Shape; center: number[]; size: number[]; rot: number }[];
  ground: { y: number; half: number }; nuisances: Nuisances;
};

const GROUND_HALF = 60;
const EYE = 1.6;
const RADIUS = 0.35; // the player, as a vertical cylinder
const HEIGHT = 1.8;
const STEP = 0.35; // ledges this high are walked onto
const WALK = 4.5, SPRINT = 8, JUMP = 7, GRAVITY = 20;
const FOV = 70;

// --- scenes ------------------------------------------------------------------------------------------

function rng(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let nextId = 1;
/** Call with a scene restored from storage, so new objects never reuse its ids. */
export const adoptIds = (s: PgScene) => { nextId = Math.max(nextId, ...s.objects.map((o) => o.id + 1)); return s; };
const ob = (shape: Shape, x: number, y: number, z: number, w: number, h: number, d: number, material: Material, rot = 0): PgObject => {
  const size: [number, number, number] = shape === "sphere" ? [w, w, w] : shape === "cylinder" ? [w, h, w] : [w, h, d];
  return { id: nextId++, shape, center: [x, y + size[1] / 2, z], size, rot, material }; // y: the object's bottom
};

export const PRESETS: Record<string, () => PgScene> = {
  Room: () => ({
    name: "Room",
    spawn: [0, 3, 0],
    objects: [
      ob("box", 0, 0, -5, 12, 3, 0.3, "brick"), ob("box", 0, 0, 5, 12, 3, 0.3, "brick"),
      ob("box", -6, 0, 0, 0.3, 3, 10, "concrete"), ob("box", 6, 0, 0, 0.3, 3, 10, "concrete"),
      ob("box", -3.5, 0, -3.5, 2.4, 0.9, 1.2, "planks"), ob("box", 3.8, 0, -3.8, 1.2, 2, 0.6, "planks"),
      ob("cylinder", 2, 0, 1, 0.8, 1.1, 0.8, "stripes"), ob("sphere", -2, 0, 2, 0.9, 0.9, 0.9, "checker"),
      ob("pyramid", 4, 0, 3, 1.2, 1.4, 1.2, "grass"), ob("box", -4.5, 0, 3.6, 1, 1, 1, "checker", Math.PI / 4),
    ],
  }),
  Courtyard: () => ({
    name: "Courtyard",
    spawn: [0, 14, 0],
    objects: [
      ob("box", 0, 0, -10, 20, 1.2, 0.5, "brick"), ob("box", -10, 0, 0, 0.5, 1.2, 20, "brick"), ob("box", 10, 0, 0, 0.5, 1.2, 20, "brick"),
      ob("pyramid", -4, 0, -4, 3, 3, 3, "concrete"), ob("box", 4, 0, -5, 3, 3, 3, "planks", 0.4),
      ob("cylinder", 0, 0, -2, 1.2, 4, 1.2, "stripes"), ob("sphere", 5, 0, 2, 1.8, 1.8, 1.8, "checker"),
      ob("box", -5, 0, 3, 2, 1, 2, "brick"), ob("box", -5, 1, 3, 1.2, 1, 1.2, "planks"),
      ob("cylinder", 7, 0, -7, 0.6, 2.5, 0.6, "concrete"), ob("pyramid", 2, 0, 5, 1.5, 2, 1.5, "brick", 0.3),
    ],
  }),
  "Block city": () => {
    const r = rng(7);
    const objects: PgObject[] = [];
    for (let i = -2; i <= 2; i++) {
      for (let j = -2; j <= 2; j++) {
        if (i === 0 && j === 2) continue; // a street to start in
        const h = 3 + Math.floor(r() * 9);
        objects.push(ob("box", i * 7, 0, j * 7, 4 + r() * 1.5, h, 4 + r() * 1.5, MATERIALS[Math.floor(r() * 6)]));
        if (r() < 0.35) objects.push(ob("pyramid", i * 7, h, j * 7, 3, 2, 3, "concrete"));
      }
    }
    return { name: "Block city", spawn: [0, 20, 0], objects };
  },
  "Obstacle course": () => ({
    name: "Obstacle course",
    spawn: [0, 10, 0],
    objects: [
      ...[0, 1, 2, 3, 4, 5].map((k) => ob("box", -6 + k * 2.4, 0, 2 - k * 2.2, 2, 0.5 + k * 0.5, 2, k % 2 ? "planks" : "checker")),
      ob("cylinder", 6, 0, -10, 1.2, 3.2, 1.2, "stripes"), ob("box", 9, 0, -8, 2, 2.5, 2, "brick"),
      ob("pyramid", -8, 0, -6, 3, 2.5, 3, "concrete"), ob("sphere", 3, 0, 4, 1.5, 1.5, 1.5, "grass"),
      ob("box", 0, 0, -14, 14, 4, 0.5, "brick"),
    ],
  }),
};

export function shuffled(seed: number): PgScene {
  const r = rng(seed);
  const objects: PgObject[] = [];
  for (let k = 0; k < 24; k++) {
    const shape = SHAPES[Math.floor(r() * 4)];
    const s = 0.8 + r() * 2.5;
    const x = (r() - 0.5) * 30, z = (r() - 0.5) * 30;
    if (Math.hypot(x, z - 12) < 3) continue; // keep the spawn clear
    objects.push(ob(shape, x, 0, z, s, shape === "box" ? s * (0.5 + r() * 1.5) : s, s, MATERIALS[Math.floor(r() * 6)], r() * Math.PI));
  }
  return { name: `Shuffle ${seed}`, spawn: [0, 12, 0], objects };
}

// --- textures (procedural, seeded: plain colours give depth models nothing to work with) ------------------

const textures = new Map<Material, THREE.CanvasTexture>();
function texture(m: Material): THREE.CanvasTexture {
  if (textures.has(m)) return textures.get(m)!;
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d")!;
  const r = rng(MATERIALS.indexOf(m) + 1);
  const noise = (n: number, a: number) => {
    for (let i = 0; i < n; i++) {
      g.fillStyle = `rgba(${r() < 0.5 ? "0,0,0" : "255,255,255"},${r() * a})`;
      g.fillRect(r() * 256, r() * 256, 1 + r() * 3, 1 + r() * 3);
    }
  };
  if (m === "brick") {
    g.fillStyle = "#cfc6b8"; g.fillRect(0, 0, 256, 256);
    for (let row = 0; row < 8; row++) for (let col = -1; col < 4; col++) {
      g.fillStyle = `hsl(${8 + r() * 12},${45 + r() * 20}%,${32 + r() * 14}%)`;
      g.fillRect(col * 64 + (row % 2) * 32 + 3, row * 32 + 3, 58, 26);
    }
    noise(2500, 0.25);
  } else if (m === "planks") {
    for (let k = 0; k < 8; k++) { g.fillStyle = `hsl(${28 + r() * 10},${45 + r() * 15}%,${36 + r() * 16}%)`; g.fillRect(0, k * 32, 256, 31); }
    for (let k = 0; k < 400; k++) { g.strokeStyle = `rgba(60,30,10,${r() * 0.35})`; const y = r() * 256; g.beginPath(); g.moveTo(0, y); g.lineTo(256, y + (r() - 0.5) * 8); g.stroke(); }
  } else if (m === "checker") {
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { g.fillStyle = (i + j) % 2 ? "#f2f2ee" : "#2f6fb0"; g.fillRect(i * 32, j * 32, 32, 32); }
    noise(1500, 0.2);
  } else if (m === "grass") {
    g.fillStyle = "#4f8f3a"; g.fillRect(0, 0, 256, 256);
    for (let k = 0; k < 6000; k++) { g.fillStyle = `hsl(${90 + r() * 30},${40 + r() * 30}%,${25 + r() * 30}%)`; g.fillRect(r() * 256, r() * 256, 1 + r() * 2, 2 + r() * 4); }
  } else if (m === "concrete") {
    g.fillStyle = "#9a9a96"; g.fillRect(0, 0, 256, 256);
    noise(9000, 0.35);
    g.strokeStyle = "rgba(40,40,40,0.5)"; g.lineWidth = 2; g.strokeRect(1, 1, 254, 254);
  } else {
    for (let k = -8; k < 16; k++) { g.fillStyle = ["#e9c46a", "#e76f51", "#264653", "#2a9d8f"][(k + 8) % 4]; g.beginPath(); g.moveTo(k * 32, 0); g.lineTo(k * 32 + 32, 0); g.lineTo(k * 32 - 224, 256); g.lineTo(k * 32 - 256, 256); g.fill(); }
    noise(1500, 0.2);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  textures.set(m, t);
  return t;
}

// Unit geometries, scaled per object. The pyramid is a 4-sided cone turned 45°: a 1 x 1 square base
// at y = -0.5 and the apex at +0.5, matching backend/playground.py.
const GEOM: Record<Shape, THREE.BufferGeometry> = {
  box: new THREE.BoxGeometry(1, 1, 1),
  pyramid: new THREE.ConeGeometry(Math.SQRT1_2, 1, 4, 1).rotateY(Math.PI / 4),
  sphere: new THREE.SphereGeometry(0.5, 48, 24),
  cylinder: new THREE.CylinderGeometry(0.5, 0.5, 1, 40),
};

function SceneObject({ o }: { o: PgObject }) {
  const material = useMemo(() => {
    const t = texture(o.material).clone();
    t.needsUpdate = true;
    t.repeat.set(Math.max(1, Math.round(Math.max(o.size[0], o.size[2]) / 1.5)), Math.max(1, Math.round(o.size[1] / 1.5)));
    return new THREE.MeshStandardMaterial({ map: t, roughness: 0.9 });
  }, [o.material, o.size[0], o.size[1], o.size[2]]);
  useEffect(() => () => { material.map?.dispose(); material.dispose(); }, [material]);
  return (
    <mesh geometry={GEOM[o.shape]} material={material} position={o.center} rotation={[0, o.rot, 0]} scale={o.size}
      castShadow receiveShadow userData={{ id: o.id }} />
  );
}

function Ground() {
  const material = useMemo(() => {
    const t = texture("concrete").clone();
    t.needsUpdate = true;
    t.repeat.set(GROUND_HALF, GROUND_HALF); // 2 m tiles
    return new THREE.MeshStandardMaterial({ map: t, roughness: 1 });
  }, []);
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow material={material} userData={{ ground: true }}>
      <planeGeometry args={[GROUND_HALF * 2, GROUND_HALF * 2]} />
    </mesh>
  );
}

// --- collision -------------------------------------------------------------------------------------------

type Aabb = { x0: number; x1: number; y0: number; y1: number; z0: number; z1: number };
function aabb(o: PgObject): Aabb {
  const [w, h, d] = o.size;
  const c = Math.abs(Math.cos(o.rot)), s = Math.abs(Math.sin(o.rot));
  const ex = c * w / 2 + s * d / 2, ez = s * w / 2 + c * d / 2;
  return { x0: o.center[0] - ex, x1: o.center[0] + ex, y0: o.center[1] - h / 2, y1: o.center[1] + h / 2, z0: o.center[2] - ez, z1: o.center[2] + ez };
}

const typingIn = (t: EventTarget | null) =>
  t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement ||
  (t instanceof HTMLInputElement && !["range", "checkbox", "radio", "button"].includes(t.type));

// --- the game loop ----------------------------------------------------------------------------------------

type Shared = { moving: boolean; phase: number; recording: boolean };

type GameProps = {
  scene: PgScene; onScene: (s: PgScene) => void;
  tool: Tool; onTool: (t: Tool) => void;
  nuisances: Nuisances; shared: Shared;
  autopilot: { kind: AutopilotKind; n: number } | null; onAutopilotDone: () => void;
  onClip: (video: Blob, truth: Truth) => void; onSnapshot: (png: Blob, truth: Truth) => void;
  onRecording: (on: boolean, seconds: number) => void; enabled: boolean;
};

function Game(p: GameProps) {
  const { camera, gl, scene: three } = useThree() as { camera: THREE.PerspectiveCamera; gl: THREE.WebGLRenderer; scene: THREE.Scene };
  const props = useRef(p);
  props.current = p;
  const st = useMemo(() => ({
    keys: new Set<string>(), pos: new THREE.Vector3(), vy: 0, yaw: 0, pitch: 0, ground: true,
    rec: null as null | { r: MediaRecorder; chunks: Blob[]; t0: number; frames: Truth["frames"] },
    ap: null as null | { kind: AutopilotKind; t: number; T: number; c: THREE.Vector3; R: number; yaw0: number },
  }), []);
  const boxes = useMemo(() => p.scene.objects.map(aabb), [p.scene]);

  // Spawn whenever the scene changes.
  useEffect(() => {
    const [x, z, yaw] = p.scene.spawn;
    st.pos.set(x, 0, z); st.vy = 0; st.yaw = yaw; st.pitch = 0;
  }, [p.scene.name, st]);

  const truthOf = (frames: Truth["frames"]): Truth => ({
    version: 1, scene: props.current.scene.name, width: gl.domElement.width, height: gl.domElement.height, fov_deg: camera.fov,
    frames, objects: props.current.scene.objects.map((o) => ({ shape: o.shape, center: o.center, size: o.size, rot: o.rot })),
    ground: { y: 0, half: GROUND_HALF }, nuisances: props.current.nuisances,
  });
  const pose = () => ({ p: camera.position.toArray(), q: camera.quaternion.toArray() as number[] });

  const startRecording = () => {
    if (st.rec) return;
    const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
    const r = new MediaRecorder(gl.domElement.captureStream(30), { mimeType: type, videoBitsPerSecond: 10_000_000 });
    const rec = { r, chunks: [] as Blob[], t0: performance.now(), frames: [] as Truth["frames"] };
    r.ondataavailable = (e) => e.data.size && rec.chunks.push(e.data);
    r.onstart = () => { rec.t0 = performance.now(); rec.frames.length = 0; };
    r.onstop = () => props.current.onClip(new Blob(rec.chunks, { type: "video/webm" }), truthOf(rec.frames));
    r.start(250);
    st.rec = rec;
    props.current.shared.recording = true;
  };
  const stopRecording = () => {
    if (!st.rec) return;
    st.rec.r.stop();
    st.rec = null;
    props.current.shared.recording = false;
    props.current.onRecording(false, 0);
  };
  const snapshot = () => {
    const truth = truthOf([{ t: 0, ...pose() }]);
    gl.domElement.toBlob((b) => b && props.current.onSnapshot(b, truth), "image/png");
  };

  // The object (or ground point) under the crosshair.
  const aim = () => {
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(0, 0), camera);
    return ray.intersectObjects(three.children, true).find((h) => h.object.userData.id || h.object.userData.ground);
  };

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const q = props.current;
      if (!q.enabled || typingIn(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
      st.keys.add(e.code);
      if (e.code === "Space") e.preventDefault();
      if (e.repeat) return;
      if (e.code === "KeyR" && !st.ap) (st.rec ? stopRecording() : startRecording());
      if (e.code === "KeyP") snapshot();
      if (/^Digit[1-4]$/.test(e.code)) q.onTool({ ...q.tool, shape: SHAPES[+e.code.slice(5) - 1] });
      if (e.code === "BracketLeft") q.onTool({ ...q.tool, size: Math.max(0.5, q.tool.size - 0.5) });
      if (e.code === "BracketRight") q.onTool({ ...q.tool, size: Math.min(12, q.tool.size + 0.5) });
      if (e.code === "KeyT") q.onTool({ ...q.tool, material: MATERIALS[(MATERIALS.indexOf(q.tool.material) + 1) % MATERIALS.length] });
      if (e.code === "KeyY") q.onTool({ ...q.tool, rot: (q.tool.rot + Math.PI / 4) % (2 * Math.PI) });
      if (e.code === "KeyF") {
        const hit = aim();
        if (!hit || hit.distance > 40) return;
        const s = q.tool.size;
        const n = hit.face ? hit.face.normal.clone().transformDirection(hit.object.matrixWorld) : new THREE.Vector3(0, 1, 0);
        const at = hit.point.clone();
        const snap = (v: number) => Math.round(v * 2) / 2;
        // On top of what's hit, or beside it when aiming at a wall.
        const bottom = n.y > 0.5 ? at.y : Math.max(0, at.y - s / 2);
        const x = n.y > 0.5 ? snap(at.x) : at.x + n.x * s / 2, z = n.y > 0.5 ? snap(at.z) : at.z + n.z * s / 2;
        q.onScene({ ...q.scene, objects: [...q.scene.objects, ob(q.tool.shape, x, bottom, z, s, s, s, q.tool.material, q.tool.rot)] });
      }
      if (e.code === "KeyX") {
        const id = aim()?.object.userData.id;
        if (id) q.onScene({ ...q.scene, objects: q.scene.objects.filter((o) => o.id !== id) });
      }
    };
    const up = (e: KeyboardEvent) => st.keys.delete(e.code);
    const blur = () => st.keys.clear();
    const look = (e: MouseEvent) => {
      if (document.pointerLockElement !== gl.domElement || st.ap) return;
      const k = 0.0022 * (props.current.nuisances.fast ? 3 : 1);
      st.yaw -= e.movementX * k;
      st.pitch = Math.max(-1.5, Math.min(1.5, st.pitch - e.movementY * k));
    };
    const lock = () => props.current.enabled && gl.domElement.requestPointerLock();
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    window.addEventListener("mousemove", look);
    gl.domElement.addEventListener("click", lock);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
      window.removeEventListener("mousemove", look);
      gl.domElement.removeEventListener("click", lock);
      if (st.rec) st.rec.r.stop();
    };
  }, [gl, st]);

  // Autopilot: a scripted, repeatable path that records itself. Only a new button press starts it,
  // never a remount (which would start a stray recording).
  const startedRun = useRef(p.autopilot?.n ?? 0);
  useEffect(() => {
    if (!p.autopilot || p.autopilot.n === startedRun.current) return;
    startedRun.current = p.autopilot.n;
    const objs = props.current.scene.objects;
    const c = new THREE.Vector3();
    objs.forEach((o) => c.add(new THREE.Vector3(o.center[0], 0, o.center[2])));
    if (objs.length) c.divideScalar(objs.length);
    // Circle just outside the free-standing objects; long walls don't count (they'd push it far out).
    const standing = objs.filter((o) => Math.max(o.size[0], o.size[2]) < 6);
    const R = Math.min(20, Math.max(5, ...standing.map((o) => Math.hypot(o.center[0] - c.x, o.center[2] - c.z) + Math.max(o.size[0], o.size[2]) / 2)) + 2.5);
    const kind = p.autopilot.kind;
    const T = kind === "loop" ? Math.min(30, Math.max(12, (2 * Math.PI * R) / 2.5)) : kind === "turn" ? 9 : 10;
    // Every run starts from the scene's spawn point, so the same scene gives the same clip.
    const [sx, sz, syaw] = props.current.scene.spawn;
    st.pos.set(sx, 0, sz);
    st.vy = 0;
    st.yaw = syaw;
    st.pitch = 0;
    st.ap = { kind, t: 0, T, c, R, yaw0: syaw };
    startRecording();
  }, [p.autopilot?.n]);

  useFrame((_, dtRaw) => {
    const dt = Math.min(dtRaw, 0.05);
    const q = props.current;
    let fwd = 0, side = 0;
    if (st.ap) {
      const a = st.ap;
      a.t += dt;
      if (a.kind === "loop") { // walk a circle around the scene, facing its middle
        const phi = (a.t / a.T) * 2 * Math.PI;
        st.pos.set(a.c.x + a.R * Math.cos(phi), 0, a.c.z + a.R * Math.sin(phi));
        st.yaw = Math.PI / 2 - phi;
        st.pitch = -0.12;
      } else if (a.kind === "turn") {
        st.yaw = a.yaw0 + a.t * (40 * Math.PI / 180);
        st.pitch = 0;
      } else { // zig-zag forward with quick turns
        st.yaw = a.yaw0 + 0.9 * Math.sin((2 * Math.PI * a.t) / 1.4);
        fwd = 1;
      }
      if (a.t >= a.T) {
        st.ap = null;
        stopRecording();
        q.onAutopilotDone();
      }
    } else if (q.enabled) {
      fwd = (st.keys.has("KeyW") ? 1 : 0) - (st.keys.has("KeyS") ? 1 : 0);
      side = (st.keys.has("KeyD") ? 1 : 0) - (st.keys.has("KeyA") ? 1 : 0);
      if (st.keys.has("Space") && st.ground) { st.vy = JUMP; st.ground = false; }
    }
    // Walk (with the sliding collision done one axis at a time), then fall.
    const speed = st.keys.has("ShiftLeft") || st.keys.has("ShiftRight") ? SPRINT : WALK;
    const len = Math.hypot(fwd, side) || 1;
    const dx = ((-Math.sin(st.yaw) * fwd + Math.cos(st.yaw) * side) / len) * speed * dt;
    const dz = ((-Math.cos(st.yaw) * fwd - Math.sin(st.yaw) * side) / len) * speed * dt;
    const blocked = (x: number, z: number, y: number) =>
      boxes.some((b) => x + RADIUS > b.x0 && x - RADIUS < b.x1 && z + RADIUS > b.z0 && z - RADIUS < b.z1 && y + HEIGHT > b.y0 && y + STEP < b.y1);
    if (!(st.ap && st.ap.kind === "loop")) {
      if (!blocked(st.pos.x + dx, st.pos.z, st.pos.y)) st.pos.x += dx;
      if (!blocked(st.pos.x, st.pos.z + dz, st.pos.y)) st.pos.z += dz;
      st.pos.x = Math.max(-GROUND_HALF + 1, Math.min(GROUND_HALF - 1, st.pos.x));
      st.pos.z = Math.max(-GROUND_HALF + 1, Math.min(GROUND_HALF - 1, st.pos.z));
      // Standing height here: the ground or the highest top under the player within a step.
      let floor = 0;
      for (const b of boxes) {
        if (st.pos.x + RADIUS > b.x0 && st.pos.x - RADIUS < b.x1 && st.pos.z + RADIUS > b.z0 && st.pos.z - RADIUS < b.z1 && b.y1 <= st.pos.y + STEP) floor = Math.max(floor, b.y1);
      }
      st.vy -= GRAVITY * dt;
      st.pos.y += st.vy * dt;
      if (st.pos.y <= floor) { st.pos.y = floor; st.vy = 0; st.ground = true; } else st.ground = false;
    }
    const moving = Math.abs(dx) + Math.abs(dz) > 1e-4 && st.ground;
    q.shared.moving = moving;
    if (moving) q.shared.phase += dt * (speed === SPRINT ? 14 : 9);
    const bob = q.nuisances.bob && moving ? Math.sin(q.shared.phase * 2) * 0.05 : 0;
    camera.position.set(st.pos.x, st.pos.y + EYE + bob, st.pos.z);
    camera.rotation.set(st.pitch, st.yaw, 0, "YXZ");
    camera.updateMatrixWorld();
    if (st.rec) {
      st.rec.frames.push({ t: (performance.now() - st.rec.t0) / 1000, ...pose() });
      q.onRecording(true, (performance.now() - st.rec.t0) / 1000);
    }
  });
  return null;
}

/** HUD panels and arms fixed to the camera, drawn into the canvas so they end up in recordings. */
function Nuisance({ nuisances, shared }: { nuisances: Nuisances; shared: Shared }) {
  const { camera, size } = useThree();
  const group = useRef<THREE.Group>(null);
  const fill = useRef<THREE.Mesh>(null);
  const left = useRef<THREE.Mesh>(null);
  const right = useRef<THREE.Mesh>(null);
  const hudMat = (color: string) => new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.92 });
  const mats = useMemo(() => ({ panel: hudMat("#1b1f27"), fill: hudMat("#3fd16a"), map: hudMat("#3a6ea5"), slot: hudMat("#d9d9d9"), arm: new THREE.MeshStandardMaterial({ color: "#f2c49b" }) }), []);
  useFrame(({ clock }) => {
    if (!group.current) return;
    group.current.position.copy(camera.position);
    group.current.quaternion.copy(camera.quaternion);
    if (fill.current) fill.current.scale.x = 0.25 + 0.75 * (0.5 + 0.5 * Math.sin(clock.elapsedTime * 0.8)); // a changing health bar
    const swing = shared.moving ? Math.sin(shared.phase) * 0.6 : 0;
    if (left.current) left.current.rotation.x = -0.3 + swing;
    if (right.current) right.current.rotation.x = -0.3 - swing;
  });
  const d = 0.3; // HUD distance in front of the eye
  const hh = d * Math.tan(((FOV / 2) * Math.PI) / 180), hw = hh * (size.width / size.height);
  return (
    <group ref={group}>
      {nuisances.hud && (
        <group renderOrder={10}>
          <mesh position={[-hw + 0.075, hh - 0.022, -d]} material={mats.panel} renderOrder={10}><planeGeometry args={[0.13, 0.02]} /></mesh>
          <mesh ref={fill} position={[-hw + 0.075, hh - 0.022, -d + 0.001]} material={mats.fill} renderOrder={11}><planeGeometry args={[0.12, 0.012]} /></mesh>
          <mesh position={[hw - 0.045, hh - 0.045, -d]} material={mats.map} renderOrder={10}><planeGeometry args={[0.07, 0.07]} /></mesh>
          {[-2, -1, 0, 1, 2].map((k) => (
            <mesh key={k} position={[k * 0.034, -hh + 0.02, -d]} material={mats.slot} renderOrder={10}><planeGeometry args={[0.028, 0.028]} /></mesh>
          ))}
        </group>
      )}
      {nuisances.arms && (
        <>
          <mesh ref={left} position={[-0.3, -0.32, -0.42]} material={mats.arm}><boxGeometry args={[0.13, 0.13, 0.62]} /></mesh>
          <mesh ref={right} position={[0.3, -0.32, -0.42]} material={mats.arm}><boxGeometry args={[0.13, 0.13, 0.62]} /></mesh>
        </>
      )}
    </group>
  );
}

export type PlaygroundProps = Omit<GameProps, "shared" | "onRecording" | "enabled">;

export default function Playground(props: PlaygroundProps) {
  const shared = useMemo<Shared>(() => ({ moving: false, phase: 0, recording: false }), []);
  const [rec, setRec] = useState<{ on: boolean; s: number }>({ on: false, s: 0 });
  const [locked, setLocked] = useState(false);
  useEffect(() => {
    const change = () => setLocked(!!document.pointerLockElement);
    document.addEventListener("pointerlockchange", change);
    return () => document.removeEventListener("pointerlockchange", change);
  }, []);
  const lastShown = useRef(0);
  const onRecording = (on: boolean, s: number) => {
    if (!on || Math.floor(s * 4) !== lastShown.current) { lastShown.current = Math.floor(s * 4); setRec({ on, s }); }
  };
  return (
    <div className="playground">
      <div className="pg-stage">
        <Canvas shadows dpr={1} gl={{ preserveDrawingBuffer: true, antialias: true }} camera={{ fov: FOV, near: 0.05, far: 400 }}>
          <color attach="background" args={["#a9cdee"]} />
          <hemisphereLight args={["#dbeeff", "#6b6250", 0.9]} />
          <directionalLight position={[18, 30, 12]} intensity={1.6} castShadow shadow-mapSize={[2048, 2048]}
            shadow-camera-left={-40} shadow-camera-right={40} shadow-camera-top={40} shadow-camera-bottom={-40} />
          <Ground />
          {props.scene.objects.map((o) => <SceneObject key={o.id} o={o} />)}
          <Game {...props} shared={shared} onRecording={onRecording} enabled />
          <Nuisance nuisances={props.nuisances} shared={shared} />
        </Canvas>
        <div className="pg-crosshair" aria-hidden="true" />
        {rec.on && <div className="pg-rec" role="status">● REC {rec.s.toFixed(1)} s · press R to stop</div>}
        {!locked && !rec.on && (
          <div className="pg-hint">Click to look around · <kbd>WASD</kbd> walk · <kbd>Space</kbd> jump · <kbd>R</kbd> record · <kbd>P</kbd> snapshot · <kbd>Esc</kbd> release the mouse</div>
        )}
      </div>
    </div>
  );
}
