import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { Cloud, RunMeta, Tear } from "./api";
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
};

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

/** Depth range of the cloud, used for clip planes and the orbit target. */
function zStats(cloud: Cloud) {
  const z = new Float32Array(cloud.n);
  for (let i = 0; i < cloud.n; i++) z[i] = cloud.xyz[3 * i + 2];
  z.sort();
  return { min: z[0] ?? 1, median: z[z.length >> 1] ?? 1, max: z[z.length - 1] ?? 1 };
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
function IntrinsicsProjection({ meta, near, far }: { meta: RunMeta; near: number; far: number }) {
  const m = useMemo(() => new THREE.Matrix4(), []);
  useFrame(({ camera, size }) => {
    const [[fx, , cx], [, fy, cy]] = meta.intrinsics;
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

function Snap({ target, snapKey }: { target: number; snapKey: number }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.Camera; controls: Controls | null };
  useEffect(() => {
    if (!controls) return;
    camera.position.set(0, 0, 0);
    camera.up.set(0, 1, 0);
    controls.target.set(0, 0, -target);
    controls.update(); // looks from the origin straight down -Z (= +Z in OpenCV): identity rotation
    controls.dispatchEvent({ type: "change" }); // update() alone doesn't fire when nothing moved
  }, [camera, controls, target, snapKey]);
  return null;
}

export default function Viewer({ cloud, meta, mode, field, tears, pointSize, snapKey, sync }: Props) {
  const z = useMemo(() => zStats(cloud), [cloud]);
  return (
    // linear + flat: colors pass through untouched, so the snapped view reproduces the input pixels.
    <Canvas className="viewer" linear flat gl={{ antialias: false, preserveDrawingBuffer: true }} dpr={1}>
      <color attach="background" args={["#111317"]} />
      <IntrinsicsProjection meta={meta} near={z.min * 0.01} far={z.max * 100} />
      <Points cloud={cloud} mode={mode} field={field} pointSize={pointSize} />
      {tears && tears.length > 0 && <TearMarkers tears={tears} />}
      <OrbitControls makeDefault enableDamping={false} />
      {sync && <SyncControls sync={sync} scale={z.median} />}
      <Snap target={z.median} snapKey={snapKey} />
    </Canvas>
  );
}
