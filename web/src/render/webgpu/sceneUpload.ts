import type { CloudLayer, FlowLayer, IsosurfaceLayer, KeyframeBlend, KeyframeFrame } from "../../types/models.js";
import { MAX_DENS_LAYERS, gpu, DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2, type RgbTriplet } from "./gpuState.js";
import { normalizeRgbStops, writeLayerColors } from "./uniforms.js";
import { uploadFlowLayers, resetFlowGpuLayers } from "./flowAdvect.js";

const MAX_FLOW_LAYERS = 4;

export interface SceneUploadPayload {
  cloudLayers?: CloudLayer[];
  isosurfaceLayers?: IsosurfaceLayer[];
  flowLayers?: FlowLayer[];
  M: number;
  half?: number;
}

export interface SceneUploadResult {
  M: number;
  bakeMs: number;
  epoch: number;
}

export interface KeyframeBlendPatch {
  id?: string;
  i0: number;
  i1: number;
  t: number;
}

function ensureVolumeBuf(floatCount: number): void {
  const { device } = gpu;
  if (!device) return;
  const aligned = Math.max(256, Math.ceil((floatCount * 4) / 256) * 256);
  if (gpu.volumeBuf && gpu.volumeCapacity >= aligned) return;
  const old = gpu.volumeBuf;
  gpu.volumeBuf = device.createBuffer({
    size: aligned,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  gpu.volumeCapacity = aligned;
  if (old) {
    void device.queue.onSubmittedWorkDone().then(() => {
      try { old.destroy(); } catch { /* device lost */ }
    });
  }
}

function stopsFromLayer(d: Pick<CloudLayer, "color" | "color2" | "colors"> | undefined): RgbTriplet[] {
  if (Array.isArray(d?.colors) && d.colors.length) {
    return d.colors.map((c) => [c[0], c[1], c[2]]);
  }
  return normalizeRgbStops(
    null,
    (d?.color as RgbTriplet) || DEFAULT_DENS_RGB,
    (d?.color2 as RgbTriplet) || DEFAULT_DENS_RGB2,
  );
}

export function uploadSceneColors(layerStopsList: RgbTriplet[][] | null | undefined): void {
  gpu.densGradStops = (layerStopsList || []).slice(0, MAX_DENS_LAYERS).map((stops) =>
    normalizeRgbStops(
      Array.isArray(stops) && stops.length && Array.isArray(stops[0]) ? stops : null,
      DEFAULT_DENS_RGB,
      DEFAULT_DENS_RGB2,
    ),
  );
  writeLayerColors(gpu.device, gpu.colorBuf, gpu.densGradStops);
}

export function uploadSceneVolumes(scene: SceneUploadPayload | null): SceneUploadResult | null {
  if (!gpu.device || !scene) return null;
  const t0 = performance.now();
  const M = Math.max(2, scene.M | 0);
  const volN = M * M * M;
  gpu.sceneM = M;

  const cons = scene.isosurfaceLayers || [];
  const flow = (scene.flowLayers || []).slice(0, MAX_FLOW_LAYERS);
  const scalarDens = (scene.cloudLayers || []).slice(0, Math.max(0, MAX_DENS_LAYERS - flow.length));
  const flowAsDens: CloudLayer[] = flow.map((f) => ({
    id: f.id,
    dens: f.dye,
    color: f.color,
    color2: f.color2,
    colors: f.colors,
  }));
  const dens = [...scalarDens, ...flowAsDens].slice(0, MAX_DENS_LAYERS);
  gpu.densLayerCount = dens.length;
  gpu.densGradStops = dens.map((d) => stopsFromLayer(d));
  const half = scene.half ?? gpu.flowHalf ?? 2.5;

  const consStride = 4;
  let consFloats = 0;
  for (const c of cons) {
    const K = Array.isArray(c.keyframes) && c.keyframes.length > 0
      ? c.keyframes.length
      : 1;
    consFloats += K * consStride * volN;
  }
  const totalFloats = consFloats + dens.length * volN;
  gpu.scenePacked = totalFloats > 0 ? new Float32Array(Math.max(volN, totalFloats)) : null;
  let off = 0;
  const putVol = (src: Float32Array | undefined) => {
    if (!gpu.scenePacked) return;
    if (src && src.length) {
      gpu.scenePacked.set(src.length >= volN ? src.subarray(0, volN) : src, off);
    }
    off += volN;
  };
  gpu.sceneConstraints = cons.map((c) => {
    const base = off;
    const frames = Array.isArray(c.keyframes) && c.keyframes.length > 0
      ? c.keyframes
      : null;
    if (frames) {
      for (const fr of frames) {
        putVol(fr.dens);
        putVol(fr.gx);
        putVol(fr.gy);
        putVol(fr.gz);
      }
    } else {
      putVol(c.dens);
      putVol(c.gx);
      putVol(c.gy);
      putVol(c.gz);
    }
    const blend: KeyframeBlend = c.blend || { i0: 0, i1: 0, t: 0 };
    const K = frames ? frames.length : 1;
    const stops = stopsFromLayer(c);
    return {
      id: c.id || null,
      color: stops[0],
      color2: stops[stops.length - 1],
      colors: stops,
      isoLevel: Number.isFinite(c.isoLevel) ? c.isoLevel : 0,
      base,
      frameStride: consStride * volN,
      K,
      i0: blend.i0 | 0,
      i1: blend.i1 | 0,
      t: Number.isFinite(blend.t) ? blend.t : 0,
    };
  });
  gpu.densBase = off;
  gpu.densPacked = dens.length > 0;
  const flowDensOffsets: number[] = [];
  if (gpu.densPacked && gpu.scenePacked) {
    for (let i = 0; i < dens.length; i++) {
      if (i >= scalarDens.length) flowDensOffsets.push(off);
      putVol(dens[i]!.dens);
    }
  }

  if (flow.length) {
    uploadFlowLayers(flow, M, half, flowDensOffsets);
  } else {
    resetFlowGpuLayers();
  }

  gpu.sceneEpoch++;
  ensureVolumeBuf(Math.max(volN, gpu.scenePacked ? gpu.scenePacked.length : volN));
  if (gpu.scenePacked && gpu.volumeBuf) gpu.device.queue.writeBuffer(gpu.volumeBuf, 0, gpu.scenePacked);
  writeLayerColors(gpu.device, gpu.colorBuf, gpu.densGradStops);

  gpu.profileBakeMs = gpu.profileBakeMs * 0.5 + (performance.now() - t0) * 0.5;
  gpu.profileGridM = M;
  const anyKf = gpu.sceneConstraints.some((c) => c.K > 1);
  gpu.profileMethod = anyKf ? "gpu-kf-scene" : "cpu-idct-scene";
  return { M, bakeMs: performance.now() - t0, epoch: gpu.sceneEpoch };
}

export function setConstraintKeyframeBlends(blends: KeyframeBlendPatch[] | null | undefined): void {
  if (!blends?.length || !gpu.sceneConstraints.length) return;
  const byId = new Map<string, KeyframeBlendPatch>();
  for (const b of blends) {
    if (b?.id != null) byId.set(b.id, b);
  }
  for (let i = 0; i < gpu.sceneConstraints.length; i++) {
    const c = gpu.sceneConstraints[i];
    const b = (c.id != null && byId.get(c.id)) || blends[i];
    if (!b) continue;
    c.i0 = Math.max(0, Math.min((c.K || 1) - 1, b.i0 | 0));
    c.i1 = Math.max(0, Math.min((c.K || 1) - 1, b.i1 | 0));
    c.t = Number.isFinite(b.t) ? b.t : 0;
  }
}

export function patchConstraintKeyframeFrame(
  layerId: string,
  frameIndex: number,
  frame: Partial<KeyframeFrame> | null | undefined,
): boolean {
  if (!gpu.device || !gpu.volumeBuf || !gpu.scenePacked || !frame) return false;
  const c = gpu.sceneConstraints.find((x) => x.id === layerId);
  if (!c || !(c.K > 1)) return false;
  const k = frameIndex | 0;
  if (k < 0 || k >= c.K) return false;
  const volN = gpu.sceneM * gpu.sceneM * gpu.sceneM;
  const stride = c.frameStride || 4 * volN;
  const base = c.base + k * stride;
  if (base + stride > gpu.scenePacked.length) return false;

  const put = (src: Float32Array | undefined, slot: number) => {
    const off = base + slot * volN;
    if (src && src.length) {
      gpu.scenePacked!.set(src.length >= volN ? src.subarray(0, volN) : src, off);
    } else {
      gpu.scenePacked!.fill(0, off, off + volN);
    }
  };
  put(frame.dens, 0);
  put(frame.gx, 1);
  put(frame.gy, 2);
  put(frame.gz, 3);

  const byteOffset = base * 4;
  const view = gpu.scenePacked.subarray(base, base + stride);
  gpu.device.queue.writeBuffer(gpu.volumeBuf, byteOffset, view);
  return true;
}

export function hasUploadedVolume(): boolean {
  return (
    gpu.sceneM > 0 &&
    (gpu.densLayerCount > 0 || gpu.sceneConstraints.length > 0 || (gpu.flowGridM ?? 0) > 0)
  );
}

export { ensureVolumeBuf };
