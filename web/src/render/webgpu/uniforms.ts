import { MAX_GRAD_STOPS } from "../../model/expressions.js";
import {
  MAX_DENS_LAYERS,
  DEFAULT_DENS_RGB,
  DEFAULT_DENS_RGB2,
  DEFAULT_ISO_RGB,
  DEFAULT_ISO_RGB2,
  gpu,
  type RgbTriplet,
} from "./gpuState.js";

export function normalizeRgbStops(
  gradRgbs: RgbTriplet[] | null | undefined,
  absorb: RgbTriplet,
  emit: RgbTriplet,
): RgbTriplet[] {
  let stops: RgbTriplet[] = Array.isArray(gradRgbs) && gradRgbs.length
    ? gradRgbs.map((c) => [c[0], c[1], c[2]])
    : [absorb, emit];
  if (stops.length < 1) stops = [DEFAULT_ISO_RGB, DEFAULT_ISO_RGB2];
  if (stops.length === 1) stops = [stops[0], stops[0]];
  if (stops.length > MAX_GRAD_STOPS) stops = stops.slice(0, MAX_GRAD_STOPS);
  return stops;
}

export function packDrawParamsIso(
  fbW: number,
  fbH: number,
  gridM: number,
  steps: number,
  half: number,
  scale: number,
  isoLevel: number,
  volBase: number,
  ro: number[],
  M: Float64Array | Float32Array | number[],
  absorb: RgbTriplet,
  emit: RgbTriplet,
  volBaseB: number = volBase,
  blendT: number = 0,
  gradRgbs: RgbTriplet[] | null = null,
  debugTint: boolean = false,
): ArrayBuffer {
  const buf = new ArrayBuffer(256);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0; u32[2] = gridM | 0; u32[3] = steps | 0;
  f32[4] = half; f32[5] = scale; f32[6] = isoLevel; f32[7] = debugTint ? 1 : 0;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[11] = volBase;
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  f32[24] = volBaseB;
  f32[25] = blendT;
  const stops = normalizeRgbStops(gradRgbs, absorb, emit);
  f32[26] = stops.length;
  f32[27] = 0;
  for (let i = 0; i < MAX_GRAD_STOPS; i++) {
    const c = stops[Math.min(i, stops.length - 1)];
    const o = 28 + i * 4;
    f32[o] = c[0]; f32[o + 1] = c[1]; f32[o + 2] = c[2]; f32[o + 3] = 1;
  }
  return buf;
}

export function packDrawParamsBeer(
  fbW: number,
  fbH: number,
  gridM: number,
  steps: number,
  half: number,
  scale: number,
  densBaseOff: number,
  layerCount: number,
  ro: number[],
  M: Float64Array | Float32Array | number[],
  flowLayerStart: number = -1,
): ArrayBuffer {
  const buf = new ArrayBuffer(512);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0; u32[2] = gridM | 0; u32[3] = steps | 0;
  f32[4] = half; f32[5] = scale; f32[6] = densBaseOff; u32[7] = layerCount | 0;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  f32[24] = flowLayerStart;
  const volN = gridM * gridM * gridM;
  for (let L = 0; L < MAX_DENS_LAYERS; L++) {
    const d = gpu.densLayers[L];
    const o = 28 + L * 4;
    if (d && d.K > 0) {
      f32[o] = d.base;
      f32[o + 1] = d.frameStride || volN;
      f32[o + 2] = d.i0 | 0;
      f32[o + 3] = d.i1 | 0;
      f32[60 + L] = Number.isFinite(d.t) ? d.t : 0;
    } else if (L < layerCount) {
      f32[o] = densBaseOff + L * volN;
      f32[o + 1] = volN;
      f32[o + 2] = 0;
      f32[o + 3] = 0;
      f32[60 + L] = 0;
    }
  }
  return buf;
}

export function packSsaoParams(
  fbW: number,
  fbH: number,
  half: number,
  radius: number,
  strength: number,
  bias: number,
  ro: number[],
  M: Float64Array | Float32Array | number[],
): ArrayBuffer {
  const buf = new ArrayBuffer(128);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0;
  f32[4] = half; f32[5] = radius; f32[6] = strength; f32[7] = bias;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  return buf;
}

export function writeLayerColors(
  device: GPUDevice | null,
  colorBuf: GPUBuffer | null,
  layerStopsList: RgbTriplet[][] | null | undefined,
): void {
  if (!device || !colorBuf) return;
  const data = new Float32Array(MAX_DENS_LAYERS * MAX_GRAD_STOPS * 4);
  for (let L = 0; L < MAX_DENS_LAYERS; L++) {
    const raw = layerStopsList?.[L];
    const stops = normalizeRgbStops(
      Array.isArray(raw) && raw.length && Array.isArray(raw[0]) ? raw : null,
      DEFAULT_DENS_RGB,
      DEFAULT_DENS_RGB2,
    );
    for (let i = 0; i < MAX_GRAD_STOPS; i++) {
      const c = stops[Math.min(i, stops.length - 1)];
      const o = (L * MAX_GRAD_STOPS + i) * 4;
      data[o] = c[0];
      data[o + 1] = c[1];
      data[o + 2] = c[2];
      data[o + 3] = i === 0 ? stops.length : 0;
    }
  }
  device.queue.writeBuffer(colorBuf, 0, data);
}
