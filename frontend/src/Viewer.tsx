import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { Cloud, RunMeta } from "./api";
import { colorize, type ColorMode } from "./colors";

type Props = { cloud: Cloud; meta: RunMeta; mode: ColorMode; pointSize: number; snapKey: number };

/** Depth range of the cloud, used for clip planes and the orbit target. */
function zStats(cloud: Cloud) {
  const z = new Float32Array(cloud.n);
  for (let i = 0; i < cloud.n; i++) z[i] = cloud.xyz[3 * i + 2];
  z.sort();
  return { min: z[0] ?? 1, median: z[z.length >> 1] ?? 1, max: z[z.length - 1] ?? 1 };
}

function Points({ cloud, mode, pointSize }: Pick<Props, "cloud" | "mode" | "pointSize">) {
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
    g.setAttribute("color", new THREE.BufferAttribute(colorize(cloud, mode), 3));
    return g;
  }, [positions, cloud, mode]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points frustumCulled={false} geometry={geometry}>
      <pointsMaterial vertexColors size={pointSize} sizeAttenuation={false} />
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
  const { camera, controls } = useThree() as { camera: THREE.Camera; controls: { target: THREE.Vector3; update(): void } | null };
  useEffect(() => {
    if (!controls) return;
    camera.position.set(0, 0, 0);
    camera.up.set(0, 1, 0);
    controls.target.set(0, 0, -target);
    controls.update(); // looks from the origin straight down -Z (= +Z in OpenCV): identity rotation
  }, [camera, controls, target, snapKey]);
  return null;
}

export default function Viewer({ cloud, meta, mode, pointSize, snapKey }: Props) {
  const z = useMemo(() => zStats(cloud), [cloud]);
  return (
    // linear + flat: colors pass through untouched, so the snapped view reproduces the input pixels.
    <Canvas className="viewer" linear flat gl={{ antialias: false, preserveDrawingBuffer: true }} dpr={1}>
      <color attach="background" args={["#111317"]} />
      <IntrinsicsProjection meta={meta} near={z.min * 0.01} far={z.max * 100} />
      <Points cloud={cloud} mode={mode} pointSize={pointSize} />
      <OrbitControls makeDefault enableDamping={false} />
      <Snap target={z.median} snapKey={snapKey} />
    </Canvas>
  );
}
