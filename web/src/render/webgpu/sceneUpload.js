import { MAX_DENS_LAYERS, gpu, DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2 } from "./gpuState.js";
import { normalizeRgbStops, writeLayerColors } from "./uniforms.js";

function ensureVolumeBuf(floatCount) {
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
      try { old.destroy(); } catch (_) {}
    });
  }
}

/** @param {{ color?: number[], color2?: number[], colors?: number[][] }} d */
function stopsFromLayer(d) {
  if (Array.isArray(d?.colors) && d.colors.length) {
    return d.colors.map((c) => [c[0], c[1], c[2]]);
  }
  return normalizeRgbStops(
    null,
    d?.color || DEFAULT_DENS_RGB,
    d?.color2 || DEFAULT_DENS_RGB2,
  );
}

/** @param {number[][][]} layerStopsList per-layer list of [r,g,b] stops */
export function uploadSceneColors(layerStopsList) {
  gpu.densGradStops = (layerStopsList || []).slice(0, MAX_DENS_LAYERS).map((stops) =>
    normalizeRgbStops(
      Array.isArray(stops) && stops.length && Array.isArray(stops[0]) ? stops : null,
      DEFAULT_DENS_RGB,
      DEFAULT_DENS_RGB2,
    ),
  );
  writeLayerColors(gpu.device, gpu.colorBuf, gpu.densGradStops);
}

export function uploadSceneVolumes(scene) {
  if (!gpu.device || !scene) return null;
  const t0 = performance.now();
  const M = Math.max(2, scene.M | 0);
  const volN = M * M * M;
  gpu.sceneM = M;

  const cons = scene.constraints || [];
  const dens = (scene.densLayers || []).slice(0, MAX_DENS_LAYERS);
  gpu.densLayerCount = dens.length;
  gpu.densGradStops = dens.map((d) => stopsFromLayer(d));

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
  const putVol = (src) => {
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
    const blend = c.blend || { i0: 0, i1: 0, t: 0 };
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
  if (gpu.densPacked && gpu.scenePacked) {
    for (let i = 0; i < dens.length; i++) {
      putVol(dens[i].dens);
    }
  }

  gpu.sceneEpoch++;
  ensureVolumeBuf(Math.max(volN, gpu.scenePacked ? gpu.scenePacked.length : volN));
  if (gpu.scenePacked) gpu.device.queue.writeBuffer(gpu.volumeBuf, 0, gpu.scenePacked);
  writeLayerColors(gpu.device, gpu.colorBuf, gpu.densGradStops);

  gpu.profileBakeMs = gpu.profileBakeMs * 0.5 + (performance.now() - t0) * 0.5;
  gpu.profileGridM = M;
  const anyKf = gpu.sceneConstraints.some((c) => c.K > 1);
  gpu.profileMethod = anyKf ? "gpu-kf-scene" : "cpu-idct-scene";
  return { M, bakeMs: performance.now() - t0, epoch: gpu.sceneEpoch };
}

/** @param {{ id?: string, i0: number, i1: number, t: number }[]} blends */
export function setConstraintKeyframeBlends(blends) {
  if (!blends?.length || !gpu.sceneConstraints.length) return;
  const byId = new Map();
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

/**
 * @param {string} layerId
 * @param {number} frameIndex
 * @param {{ dens?: Float32Array, gx?: Float32Array, gy?: Float32Array, gz?: Float32Array }} frame
 * @returns {boolean}
 */
export function patchConstraintKeyframeFrame(layerId, frameIndex, frame) {
  if (!gpu.device || !gpu.volumeBuf || !gpu.scenePacked || !frame) return false;
  const c = gpu.sceneConstraints.find((x) => x.id === layerId);
  if (!c || !(c.K > 1)) return false;
  const k = frameIndex | 0;
  if (k < 0 || k >= c.K) return false;
  const volN = gpu.sceneM * gpu.sceneM * gpu.sceneM;
  const stride = c.frameStride || 4 * volN;
  const base = c.base + k * stride;
  if (base + stride > gpu.scenePacked.length) return false;

  const put = (src, slot) => {
    const off = base + slot * volN;
    if (src && src.length) {
      gpu.scenePacked.set(src.length >= volN ? src.subarray(0, volN) : src, off);
    } else {
      gpu.scenePacked.fill(0, off, off + volN);
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

export function hasUploadedVolume() {
  return gpu.sceneM > 0 && (gpu.densLayerCount > 0 || gpu.sceneConstraints.length > 0);
}

export { ensureVolumeBuf };
