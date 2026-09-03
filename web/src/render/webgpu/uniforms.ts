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

const ISO_DRAW_PARAM_BYTES = 256;
const isoDrawParamScratch = new ArrayBuffer(ISO_DRAW_PARAM_BYTES);
const isoDrawParamU32 = new Uint32Array(isoDrawParamScratch);
const isoDrawParamF32 = new Float32Array(isoDrawParamScratch);

const beerDrawParamScratch = new ArrayBuffer(512);
const beerDrawParamU32 = new Uint32Array(beerDrawParamScratch);
const beerDrawParamF32 = new Float32Array(beerDrawParamScratch);

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
  layerIndex: number = 0,
): ArrayBuffer {
  const u32 = isoDrawParamU32;
  const f32 = isoDrawParamF32;
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
  f32[27] = Math.max(0, layerIndex | 0);
  for (let i = 0; i < MAX_GRAD_STOPS; i++) {
    const c = stops[Math.min(i, stops.length - 1)];
    const o = 28 + i * 4;
    f32[o] = c[0]; f32[o + 1] = c[1]; f32[o + 2] = c[2]; f32[o + 3] = 1;
  }
  return isoDrawParamScratch;
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
  /** isoRefineDebug tint: 0 = off, 1 = cheap, 2 = mid, 3 = final beer tier. */
  debugTier: number = 0,
  /** Should this pass's fsRefine also claim box-silhouette pixels? False for
   *  the final pass when a mid tier already claimed them this frame. */
  nearEdgeActive: number = 1,
  /** NDC-space dilation radius for beerNearBoxEdge: 0 for a single center
   *  sample (final pass — matches blit.wgsl's compose-res evaluation
   *  exactly); half this pass's own NDC pixel width for the mid pass, so it
   *  samples its own footprint instead of just its (coarser) pixel center —
   *  see beer.wgsl's beerNearBoxEdge doc comment. */
  dilateNdc: number = 0,
): ArrayBuffer {
  const u32 = beerDrawParamU32;
  const f32 = beerDrawParamF32;
  f32.fill(0);
  u32[0] = fbW | 0; u32[1] = fbH | 0; u32[2] = gridM | 0; u32[3] = steps | 0;
  f32[4] = half; f32[5] = scale; f32[6] = densBaseOff; u32[7] = layerCount | 0;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  f32[24] = flowLayerStart;
  f32[25] = debugTier;
  f32[26] = nearEdgeActive;
  f32[27] = dilateNdc;
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
  return beerDrawParamScratch;
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
