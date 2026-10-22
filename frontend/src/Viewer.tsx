import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { BlockPreview, Cloud, RunMeta, Tear } from "./api";
import { colorize, type ColorMode } from "./colors";

type Props = {
  cloud: Cloud;
  meta: RunMeta;
  mode: ColorMode;
  field?: Float32Array | null; // per-point curl/residual, same order as the cloud
  tears?: Tear[];
  pointSize: number;
  snapKey: number;
  sync?: OrbitSync;
  nav: NavSettings;
  blocks?: BlockPreview | null; // show the Minecraft build instead of the points
  frame?: number; // scenes: the frame whose camera Snap goes to
};

/** The camera Snap goes to: a scene frame's camera, or the input camera (identity) for images. */
const cameraOf = (meta: RunMeta, frame = 0) => meta.cameras?.[frame] ?? { c2w: null, K: meta.intrinsics };

/**
 * Shares one orbit between viewers. Poses are exchanged in units of each cloud's median depth, so
 * models with different scales (metric vs affine-invariant) stay visually aligned, and the snapped
 * pose (camera at origin, target at depth 1) is the snapped pose for every panel.
 */
export type Pose = { position: THREE.Vector3; target: THREE.Vector3; quaternion: THREE.Quaternion };
export class OrbitSync {
  private listeners = new Set<(p: Pose, src: object) => void>();
  subscribe(fn: (p: Pose, src: object) => void) {
    this.listeners.add(fn);
    return () => void this.listeners.delete(fn);
  }
  publish(p: Pose, src: object) {
    this.listeners.forEach((fn) => fn(p, src));
  }
}

type Controls = THREE.EventDispatcher<{ change: object }> & { target: THREE.Vector3; update(): void };

function SyncControls({ sync, scale }: { sync: OrbitSync; scale: number }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.Camera; controls: Controls | null };
  useEffect(() => {
    if (!controls) return;
    let applying = false;
    const onChange = () => {
      if (applying) return;
      sync.publish(
        {
          position: camera.position.clone().divideScalar(scale),
          target: controls.target.clone().divideScalar(scale),
          quaternion: camera.quaternion.clone(),
        },
        controls,
      );
    };
    const unsub = sync.subscribe((p, src) => {
      if (src === controls) return;
      applying = true;
      camera.position.copy(p.position).multiplyScalar(scale);
      controls.target.copy(p.target).multiplyScalar(scale);
      camera.quaternion.copy(p.quaternion);
      controls.update();
      applying = false;
    });
    controls.addEventListener("change", onChange);
    return () => {
      unsub();
      controls.removeEventListener("change", onChange);
    };
  }, [sync, scale, camera, controls]);
  return null;
}

/** Depth range of the cloud, used for clip planes and the orbit target. Scenes have points behind
 * frame 0's camera too, so distances are |z| and the far plane covers the whole cloud. */
function zStats(cloud: Cloud) {
  const z = new Float32Array(cloud.n);
  let far = 0;
  for (let i = 0; i < cloud.n; i++) {
    z[i] = Math.abs(cloud.xyz[3 * i + 2]);
    far = Math.max(far, Math.abs(cloud.xyz[3 * i]), Math.abs(cloud.xyz[3 * i + 1]), z[i]);
  }
  z.sort();
  const median = z[z.length >> 1] || 1;
  return { min: Math.max(z[0] ?? 1, median * 1e-3), median, max: far || 1 };
}

function Points({ cloud, mode, field, pointSize }: Pick<Props, "cloud" | "mode" | "field" | "pointSize">) {
  // OpenCV camera frame -> three.js camera frame: flip Y and Z.
  const positions = useMemo(() => {
    const p = new Float32Array(cloud.xyz.length);
    for (let i = 0; i < cloud.n; i++) {
      p[3 * i] = cloud.xyz[3 * i];
      p[3 * i + 1] = -cloud.xyz[3 * i + 1];
      p[3 * i + 2] = -cloud.xyz[3 * i + 2];
    }
    return p;
  }, [cloud]);
  // Built imperatively: R3F does not re-upload a <bufferAttribute> when only its args change.
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colorize(cloud, mode, field), 3));
    return g;
  }, [positions, cloud, mode, field]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points frustumCulled={false} geometry={geometry}>
      <pointsMaterial vertexColors size={pointSize} sizeAttenuation={false} />
    </points>
  );
}

// Minecraft's fixed face shading, so the build reads like it does in game: top, sides, bottom.
// BoxGeometry face order is +x, -x, +y, -y, +z, -z; x is east/west and z north/south in both frames.
const FACE_SHADE = [0.6, 0.6, 1.0, 0.5, 0.8, 0.8];

/** The Minecraft build as real cubes (one instanced mesh), in the cloud's camera frame. */
function Blocks({ blocks }: { blocks: BlockPreview }) {
  const mesh = useMemo(() => {
    const g = new THREE.BoxGeometry(1, 1, 1);
    const shade = new Float32Array(g.attributes.position.count * 3);
    for (let v = 0; v < g.attributes.position.count; v++) shade.fill(FACE_SHADE[Math.floor(v / 4)], 3 * v, 3 * v + 3);
    g.setAttribute("color", new THREE.BufferAttribute(shade, 3));
    const m = new THREE.InstancedMesh(g, new THREE.MeshBasicMaterial({ vertexColors: true }), blocks.n);
    const o = new THREE.Object3D();
    const c = new THREE.Color();
    const s = blocks.size;
    // Minecraft axes -> cloud (OpenCV) -> three.js (flip y, z). Cubes stay axis-aligned in Minecraft.
    const [a, o3] = [blocks.axes, blocks.origin];
    const toThree = new THREE.Matrix4().set(
      a[0], a[1], a[2], o3[0],
      -a[3], -a[4], -a[5], -o3[1],
      -a[6], -a[7], -a[8], -o3[2],
      0, 0, 0, 1,
    );
    o.quaternion.setFromRotationMatrix(toThree);
    for (let i = 0; i < blocks.n; i++) {
      const x = blocks.lo[0] + (blocks.ijk[3 * i] + 0.5) * s;
      const y = blocks.lo[1] + (blocks.ijk[3 * i + 1] + 0.5) * s;
      const z = blocks.lo[2] + (blocks.ijk[3 * i + 2] + 0.5) * s;
      o.position.set(x, y, z).applyMatrix4(toThree);
      o.scale.setScalar(s);
      o.updateMatrix();
      m.setMatrixAt(i, o.matrix);
      m.setColorAt(i, c.setRGB(blocks.rgb[3 * i] / 255, blocks.rgb[3 * i + 1] / 255, blocks.rgb[3 * i + 2] / 255));
    }
    m.frustumCulled = false;
    return m;
  }, [blocks]);
  useEffect(() => () => {
    mesh.geometry.dispose();
    (mesh.material as THREE.Material).dispose();
    mesh.dispose();
  }, [mesh]);
  return <primitive object={mesh} />;
}

/** Tear locations as large markers drawn over the cloud (no depth test, so they're never hidden). */
function TearMarkers({ tears }: { tears: Tear[] }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const p = new Float32Array(tears.flatMap((t) => [t.xyz[0], -t.xyz[1], -t.xyz[2]]));
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    return g;
  }, [tears]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points geometry={geometry} renderOrder={1} frustumCulled={false}>
      <pointsMaterial color="#ff2bd6" size={9} sizeAttenuation={false} depthTest={false} transparent />
    </points>
  );
}

/**
 * Projection built straight from the run's intrinsics, so that from the origin the cloud lands on
 * exactly the pixels it came from. The image is fit inside the canvas (letterboxed) at scale s:
 * canvas_px = (image_px - size/2) * s + canvas/2. Re-applied every frame because R3F resets the
 * projection on resize.
 */
function IntrinsicsProjection({ meta, frame, near, far }: { meta: RunMeta; frame?: number; near: number; far: number }) {
  const m = useMemo(() => new THREE.Matrix4(), []);
  useFrame(({ camera, size }) => {
    const [[fx, , cx], [, fy, cy]] = cameraOf(meta, frame).K;
    const { width: W, height: H } = meta;
    const cw = size.width, ch = size.height;
    const s = Math.min(cw / W, ch / H);
    m.set(
      (2 * s * fx) / cw, 0, (-2 * s * (cx - W / 2)) / cw, 0,
      0, (2 * s * fy) / ch, (2 * s * (cy - H / 2)) / ch, 0,
      0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near),
      0, 0, -1, 0,
    );
    camera.projectionMatrix.copy(m);
    camera.projectionMatrixInverse.copy(m).invert();
  });
  return null;
}

/** Navigation sensitivities. `move` is in cloud-median-depths per second, so it feels the same for
 * metric and relative models; the rest are OrbitControls' own speed factors (1 = default). */
export type NavSettings = { move: number; rotate: number; zoom: number; pan: number };

// Camera-local directions: three.js cameras look down -Z with +Y up.
const MOVE_KEYS: Record<string, [number, number, number]> = {
  KeyW: [0, 0, -1],
  KeyS: [0, 0, 1],
  KeyA: [-1, 0, 0],
  KeyD: [1, 0, 0],
  KeyE: [0, 1, 0],
  KeyQ: [0, -1, 0],
};
const SPRINT = 3;

// In a comparison grid only one viewer may drive from the keyboard (the one under the mouse); the
// others follow through OrbitSync. Otherwise every panel would move and then echo each other.
let keyOwner: object | null = null;

const typingIn = (t: EventTarget | null) =>
  document.querySelector("dialog[open]") !== null || // e.g. the image lightbox
  t instanceof HTMLTextAreaElement ||
  (t instanceof HTMLInputElement && !["range", "checkbox", "radio", "button", "file"].includes(t.type));

/** WASD fly: moves the camera and the orbit target together, so mouse orbiting keeps working. */
function WasdControls({ speed, scale }: { speed: number; scale: number }) {
  const { camera, controls, gl } = useThree() as unknown as {
    camera: THREE.Camera;
    controls: Controls | null;
    gl: THREE.WebGLRenderer;
  };
  const me = useMemo(() => ({}), []);
  const held = useMemo(() => new Set<string>(), []);
  const v = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    const el = gl.domElement;
    const claim = () => void (keyOwner = me);
    keyOwner ??= me;
    const down = (e: KeyboardEvent) => {
      if (typingIn(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.code in MOVE_KEYS || e.key === "Shift") held.add(e.code in MOVE_KEYS ? e.code : "Shift");
    };
    const up = (e: KeyboardEvent) => void held.delete(e.code in MOVE_KEYS ? e.code : e.key === "Shift" ? "Shift" : "");
    const clear = () => held.clear();
    el.addEventListener("pointerenter", claim);
    el.addEventListener("pointerdown", claim);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", clear); // keys released while the window was unfocused
    return () => {
      el.removeEventListener("pointerenter", claim);
      el.removeEventListener("pointerdown", claim);
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", clear);
      if (keyOwner === me) keyOwner = null;
    };
  }, [gl, me, held]);

  useFrame((_, dt) => {
    keyOwner ??= me; // the previous owner's panel was closed
    if (!controls || keyOwner !== me || held.size === 0) return;
    v.set(0, 0, 0);
    held.forEach((k) => {
      const d = MOVE_KEYS[k];
      if (d) v.set(v.x + d[0], v.y + d[1], v.z + d[2]);
    });
    if (v.lengthSq() === 0) return;
    const step = speed * scale * Math.min(dt, 0.1) * (held.has("Shift") ? SPRINT : 1);
    v.normalize().multiplyScalar(step).applyQuaternion(camera.quaternion);
    camera.position.add(v);
    controls.target.add(v);
    controls.update(); // fires "change", which also moves synced panels
  });
  return null;
}

function Snap({ target, snapKey, meta, frame }: { target: number; snapKey: number; meta: RunMeta; frame?: number }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.Camera; controls: Controls | null };
  const c2w = cameraOf(meta, frame).c2w;
  useEffect(() => {
    if (!controls) return;
    camera.up.set(0, 1, 0);
    if (!c2w) {
      camera.position.set(0, 0, 0);
      controls.target.set(0, 0, -target);
      controls.update(); // looks from the origin straight down -Z (= +Z in OpenCV): identity rotation
    } else {
      // OpenCV camera-to-world -> three.js: F R F with F = diag(1, -1, -1) on both sides.
      const r = (i: number, j: number) => (i === 0 ? 1 : -1) * (j === 0 ? 1 : -1) * c2w[i][j];
      const R = new THREE.Matrix4().set(r(0, 0), r(0, 1), r(0, 2), 0, r(1, 0), r(1, 1), r(1, 2), 0, r(2, 0), r(2, 1), r(2, 2), 0, 0, 0, 0, 1);
      camera.position.set(c2w[0][3], -c2w[1][3], -c2w[2][3]);
      controls.target.copy(new THREE.Vector3(0, 0, -target).applyMatrix4(R).add(camera.position));
      controls.update(); // aims at the target but drops the camera's roll...
      camera.quaternion.setFromRotationMatrix(R); // ...so restore the exact orientation
    }
    controls.dispatchEvent({ type: "change" }); // update() alone doesn't fire when nothing moved
  }, [camera, controls, target, snapKey, c2w]);
  return null;
}

export default function Viewer({ cloud, meta, mode, field, tears, pointSize, snapKey, sync, nav, blocks, frame }: Props) {
  const z = useMemo(() => zStats(cloud), [cloud]);
  return (
    // linear + flat: colors pass through untouched, so the snapped view reproduces the input pixels.
    <Canvas className="viewer" linear flat gl={{ antialias: false, preserveDrawingBuffer: true }} dpr={1}>
      <color attach="background" args={["#111317"]} />
      <IntrinsicsProjection meta={meta} frame={frame} near={z.min * 0.01} far={z.max * 100} />
      {blocks ? <Blocks blocks={blocks} /> : <Points cloud={cloud} mode={mode} field={field} pointSize={pointSize} />}
      {tears && tears.length > 0 && <TearMarkers tears={tears} />}
      <OrbitControls makeDefault enableDamping={false} rotateSpeed={nav.rotate} zoomSpeed={nav.zoom} panSpeed={nav.pan} />
      <WasdControls speed={nav.move} scale={z.median} />
      {sync && <SyncControls sync={sync} scale={z.median} />}
      <Snap target={z.median} snapKey={snapKey} meta={meta} frame={frame} />
    </Canvas>
  );
}
