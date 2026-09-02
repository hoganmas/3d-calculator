import type { PerspectiveCamera } from "three";
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "../camera.js";
import {
  gpu,
  DEFAULT_ISO_RGB,
  DEFAULT_ISO_RGB2,
} from "./gpuState.js";
import {
  packDrawParamsIso,
  packDrawParamsBeer,
} from "./uniforms.js";
import { packGridParams, syncClipGpuWorldGrid, uploadAxisLabelBillboards } from "./gridOverlay.js";
import { state } from "../../app/state.js";
import { clampIsoStepsForTier, coarseIsoSteps } from "../../app/deviceTier.js";
import { ensureMarchTargets, ensureIsoCoarseTargets, ensureIsoMidTargets, ensureVolumeTargets, resizeClipGpuCanvas } from "./marchCanvas.js";
import {
  isoFineFramebufferSize,
  isoMidFramebufferSize,
  isoRefineEnabled,
} from "./isoRefine.js";
import { isIsoRefineDebugEnabled } from "../../app/isoRefineDebug.js";
import {
  beginGpuFrame,
  endGpuFrame,
  gpuWriteBuffer,
  sampleGpuPresent,
  submitEnc,
  withStampWrites,
} from "./gpuSubmit.js";
import { hasFlowGpuLayers } from "./flowGpu.js";
import {
  drawFlowParticlesPass,
  tickFlowParticles,
} from "./flowParticles.js";
import {
  acquireMarchGpuHandles,
  acquireMarchTargets,
  type MarchGpuHandles,
  type MarchTargets,
  type RenderClipFrameGpuParams,
} from "./marchTypes.js";

/** NDC from fragPos must use the color attachment's pixel size. */
function setPassViewport(pass: GPURenderPassEncoder, w: number, h: number): void {
  pass.setViewport(0, 0, Math.max(1, w), Math.max(1, h), 0, 1);
}

const texViewCache = new WeakMap<GPUTexture, GPUTextureView>();
function texView(tex: GPUTexture): GPUTextureView {
  let view = texViewCache.get(tex);
  if (!view) {
    view = tex.createView();
    texViewCache.set(tex, view);
  }
  return view;
}

function isoParamBuf(device: GPUDevice, ci: number): GPUBuffer {
  const bufs = gpu.isoDrawParamBufs;
  if (bufs.length === 0) {
    if (gpu.drawParamBuf) bufs.push(gpu.drawParamBuf);
    if (gpu.drawParamBufRefine && gpu.drawParamBufRefine !== gpu.drawParamBuf) {
      bufs.push(gpu.drawParamBufRefine);
    }
  }
  while (bufs.length <= ci) {
    bufs.push(device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }));
  }
  return bufs[ci]!;
}

let isoBgPipeline: GPURenderPipeline | null = null;
let isoBgVolume: GPUBuffer | null = null;
const isoBgs: GPUBindGroup[] = [];

let refineBgPipeline: GPURenderPipeline | null = null;
let refineBgVolume: GPUBuffer | null = null;
const refineBgByOccl: { occl: GPUTexture; bgs: GPUBindGroup[] }[] = [];

function isoBindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  volumeBuf: GPUBuffer,
  paramBuf: GPUBuffer,
  ci: number,
): GPUBindGroup {
  if (isoBgPipeline !== pipeline || isoBgVolume !== volumeBuf) {
    isoBgs.length = 0;
    isoBgPipeline = pipeline;
    isoBgVolume = volumeBuf;
  }
  let bg = isoBgs[ci];
  if (!bg) {
    bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: volumeBuf } },
      ],
    });
    isoBgs[ci] = bg;
  }
  return bg;
}

function isoRefineBindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  volumeBuf: GPUBuffer,
  paramBuf: GPUBuffer,
  srcOccl: GPUTexture,
  ci: number,
): GPUBindGroup {
  if (refineBgPipeline !== pipeline || refineBgVolume !== volumeBuf) {
    refineBgByOccl.length = 0;
    refineBgPipeline = pipeline;
    refineBgVolume = volumeBuf;
  }
  let slot = refineBgByOccl.find((s) => s.occl === srcOccl);
  if (!slot) {
    slot = { occl: srcOccl, bgs: [] };
    refineBgByOccl.push(slot);
    if (refineBgByOccl.length > 2) refineBgByOccl.shift();
  }
  let bg = slot.bgs[ci];
  if (!bg) {
    bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: volumeBuf } },
        { binding: 2, resource: texView(srcOccl) },
      ],
    });
    slot.bgs[ci] = bg;
  }
  return bg;
}

function writeIsoConstraintParams(
  device: GPUDevice,
  buf: GPUBuffer,
  destW: number,
  destH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
  debugTint: boolean,
  ci: number,
): void {
  const c = gpu.sceneConstraints[ci];
  const stride = c.frameStride || 0;
  const base0 = c.base + (c.i0 | 0) * stride;
  const base1 = c.base + (c.i1 | 0) * stride;
  const blendT = Number.isFinite(c.t) ? c.t : 0;
  const c0 = c.color || DEFAULT_ISO_RGB;
  const c1 = c.color2 || DEFAULT_ISO_RGB2;
  const stops = c.colors || [c0, c1];
  gpuWriteBuffer(
    device,
    buf,
    packDrawParamsIso(
      destW, destH, Mgrid, steps, half, scale, c.isoLevel, base0, ro, dirMatrix,
      c0, c1,
      base1, blendT, stops,
      debugTint,
      ci + 1,
    ),
  );
}

function buildRaySetup(
  params: RenderClipFrameGpuParams,
): {
  handles: MarchGpuHandles;
  targets: MarchTargets;
  ro: [number, number, number];
  dirMatrix: ReturnType<typeof offsetDirMatrix>;
  half: number;
  marchW: number;
  marchH: number;
  volW: number;
  volH: number;
  outW: number;
  outH: number;
  composeW: number;
  composeH: number;
  refine: boolean;
  midRefine: boolean;
} | null {
  if (gpu.densLayerCount < 1 && gpu.sceneConstraints.length < 1 && !hasFlowGpuLayers()) return null;

  const handles = acquireMarchGpuHandles();
  if (!handles) return null;

  const { camera, half, fbW, fbH, ndcOffsetX = 0, ndcOffsetY = 0, displayW = 0, displayH = 0 } = params;
  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const dirMatrix = offsetDirMatrix(ndcToDirMatrix(camera, sx, sy), ndcOffsetX, ndcOffsetY);
  const ro: [number, number, number] = [o.x, o.y, o.z];
  const h = half ?? 2;

  const marchW = Math.max(1, fbW | 0);
  const marchH = Math.max(1, fbH | 0);
  const volW = Math.max(1, (params.volFbW ?? 0) || marchW);
  const volH = Math.max(1, (params.volFbH ?? 0) || marchH);
  const outW = Math.max(1, (displayW | 0) || marchW);
  const outH = Math.max(1, (displayH | 0) || marchH);
  const fineDown = Math.min(16, Math.max(1, (params.isoFineDownscale ?? 1) | 0));
  const refine = gpu.sceneConstraints.length > 0 && isoRefineEnabled(marchW, marchH, outW, outH, fineDown);
  const { fw: composeW, fh: composeH } = refine
    ? isoFineFramebufferSize(marchW, marchH, outW, outH, fineDown)
    : { fw: marchW, fh: marchH };
  const midSize = refine ? isoMidFramebufferSize(marchW, marchH, outW, outH, fineDown) : null;

  resizeClipGpuCanvas(outW, outH);
  if (refine) ensureIsoCoarseTargets(marchW, marchH);
  if (midSize) ensureIsoMidTargets(midSize.mw, midSize.mh);
  ensureMarchTargets(composeW, composeH);
  if (gpu.densLayerCount > 0) ensureVolumeTargets(volW, volH);
  else ensureVolumeTargets(1, 1);
  syncClipGpuWorldGrid(h);

  const targets = acquireMarchTargets();
  if (!targets) return null;

  return {
    handles, targets, ro, dirMatrix, half: h,
    marchW, marchH, volW, volH, outW, outH, composeW, composeH, refine,
    midRefine: !!midSize,
  };
}

function clearIsoGBuffer(
  device: GPUDevice,
  color: GPUTexture,
  occl: GPUTexture,
  normal: GPUTexture,
  depth: GPUTexture,
  stampBegin: boolean,
): void {
  const desc: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: color.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
      {
        view: occl.createView(),
        clearValue: { r: 1, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
      {
        view: normal.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depth.createView(),
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  };
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass(stampBegin ? withStampWrites(desc, "begin") : desc);
  pass.end();
  submitEnc(device, enc);
}

function clearMarchTargets(
  device: GPUDevice,
  targets: MarchTargets,
): void {
  clearIsoGBuffer(device, targets.sceneColorTex, targets.occlIsoTex, targets.normalTex, targets.depthTex, true);
}

function drawIsoConstraints(
  handles: MarchGpuHandles,
  sceneTex: GPUTexture,
  occlTex: GPUTexture,
  normalTex: GPUTexture,
  depthTex: GPUTexture,
  fbW: number,
  fbH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
  stampBegin = false,
): void {
  const n = gpu.sceneConstraints.length;
  if (n < 1) return;
  const { device, isoPipeline, volumeBuf } = handles;
  const sceneView = texView(sceneTex);
  const occlIsoView = texView(occlTex);
  const depthView = texView(depthTex);
  const normalView = texView(normalTex);

  for (let ci = 0; ci < n; ci++) {
    writeIsoConstraintParams(
      device, isoParamBuf(device, ci),
      sceneTex.width, sceneTex.height,
      Mgrid, steps, half, scale, ro, dirMatrix, false, ci,
    );
  }

  const desc: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view: sceneView,
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        storeOp: "store",
      },
      {
        view: occlIsoView,
        loadOp: "clear",
        clearValue: { r: 1, g: 0, b: 0, a: 1 },
        storeOp: "store",
      },
      {
        view: normalView,
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  };
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass(stampBegin ? withStampWrites(desc, "begin") : desc);
  pass.setPipeline(isoPipeline);
  setPassViewport(pass, sceneTex.width, sceneTex.height);
  for (let ci = 0; ci < n; ci++) {
    pass.setBindGroup(0, isoBindGroup(device, isoPipeline, volumeBuf, isoParamBuf(device, ci), ci));
    pass.draw(3);
  }
  pass.end();
  submitEnc(device, enc);
}

interface IsoGBuffer {
  color: GPUTexture;
  occl: GPUTexture;
  normal: GPUTexture;
  depth: GPUTexture;
}

function acquireIsoMidGBuffer(): IsoGBuffer | null {
  const color = gpu.isoMidColorTex;
  const occl = gpu.isoMidOcclTex;
  const normal = gpu.isoMidNormalTex;
  const depth = gpu.isoMidDepthTex;
  if (!color || !occl || !normal || !depth) return null;
  return { color, occl, normal, depth };
}

let upBgPipeline: GPURenderPipeline | null = null;
let upBgSrcColor: GPUTexture | null = null;
let upBgSrcOccl: GPUTexture | null = null;
let upBgSrcNormal: GPUTexture | null = null;
let upBg: GPUBindGroup | null = null;
const upsampleParamScratch = new Uint32Array(4);

function upsampleBindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  paramBuf: GPUBuffer,
  src: { color: GPUTexture; occl: GPUTexture; normal: GPUTexture },
): GPUBindGroup {
  if (
    upBg &&
    upBgPipeline === pipeline &&
    upBgSrcColor === src.color &&
    upBgSrcOccl === src.occl &&
    upBgSrcNormal === src.normal
  ) {
    return upBg;
  }
  upBgPipeline = pipeline;
  upBgSrcColor = src.color;
  upBgSrcOccl = src.occl;
  upBgSrcNormal = src.normal;
  upBg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramBuf } },
      { binding: 1, resource: texView(src.color) },
      { binding: 2, resource: texView(src.occl) },
      { binding: 3, resource: texView(src.normal) },
    ],
  });
  return upBg;
}

function upsampleIso(
  handles: MarchGpuHandles,
  src: { color: GPUTexture; occl: GPUTexture; normal: GPUTexture },
  dest: IsoGBuffer,
): boolean {
  const { device, isoUpsamplePipeline, isoUpsampleParamBuf } = handles;
  const fw = dest.color.width;
  const fh = dest.color.height;
  const debug = isIsoRefineDebugEnabled() ? 1 : 0;
  upsampleParamScratch[0] = fw | 0;
  upsampleParamScratch[1] = fh | 0;
  upsampleParamScratch[2] = debug;
  upsampleParamScratch[3] = 0;
  gpuWriteBuffer(device, isoUpsampleParamBuf, upsampleParamScratch);
  const bg = upsampleBindGroup(device, isoUpsamplePipeline, isoUpsampleParamBuf, src);
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      { view: texView(dest.color), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" },
      { view: texView(dest.occl), loadOp: "clear", clearValue: { r: 1, g: 0, b: 0, a: 1 }, storeOp: "store" },
      { view: texView(dest.normal), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" },
    ],
    depthStencilAttachment: {
      view: texView(dest.depth),
      depthClearValue: 1,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });
  pass.setPipeline(isoUpsamplePipeline);
  pass.setBindGroup(0, bg);
  setPassViewport(pass, fw, fh);
  pass.draw(3);
  pass.end();
  submitEnc(device, enc);
  return true;
}

function drawIsoRefine(
  handles: MarchGpuHandles,
  dest: IsoGBuffer,
  srcOccl: GPUTexture,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): void {
  const n = gpu.sceneConstraints.length;
  if (n < 1) return;
  const { device, isoRefinePipeline, volumeBuf } = handles;
  const sceneView = texView(dest.color);
  const occlIsoView = texView(dest.occl);
  const depthView = texView(dest.depth);
  const normalView = texView(dest.normal);

  for (let ci = 0; ci < n; ci++) {
    writeIsoConstraintParams(
      device, isoParamBuf(device, ci),
      dest.color.width, dest.color.height,
      Mgrid, steps, half, scale, ro, dirMatrix, isIsoRefineDebugEnabled(), ci,
    );
  }

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      { view: sceneView, loadOp: "load", storeOp: "store" },
      { view: occlIsoView, loadOp: "load", storeOp: "store" },
      { view: normalView, loadOp: "load", storeOp: "store" },
    ],
    depthStencilAttachment: {
      view: depthView,
      depthLoadOp: "load",
      depthStoreOp: "store",
    },
  });
  pass.setPipeline(isoRefinePipeline);
  setPassViewport(pass, dest.color.width, dest.color.height);
  for (let ci = 0; ci < n; ci++) {
    pass.setBindGroup(
      0,
      isoRefineBindGroup(
        device, isoRefinePipeline, volumeBuf, isoParamBuf(device, ci), srcOccl, ci,
      ),
    );
    pass.draw(3);
  }
  pass.end();
  submitEnc(device, enc);
}

/** Coarse occupancy, optional 4× mid remarch, then slider-sized compose. */
function runIsoRefineLadder(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  marchW: number,
  marchH: number,
  midRefine: boolean,
  Mgrid: number,
  occSteps: number,
  isoSteps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): "mid" | "two" | false {
  const coarseColor = gpu.isoCoarseColorTex;
  const coarseOccl = gpu.isoCoarseOcclTex;
  const coarseNormal = gpu.isoCoarseNormalTex;
  const coarseDepth = gpu.isoCoarseDepthTex;
  if (!coarseColor || !coarseOccl || !coarseNormal || !coarseDepth) return false;

  const coarseSrc = { color: coarseColor, occl: coarseOccl, normal: coarseNormal };
  const fine: IsoGBuffer = {
    color: targets.sceneColorTex,
    occl: targets.occlIsoTex,
    normal: targets.normalTex,
    depth: targets.depthTex,
  };

  drawIsoConstraints(
    handles, coarseColor, coarseOccl, coarseNormal, coarseDepth,
    marchW, marchH, Mgrid, occSteps, half, scale, ro, dirMatrix,
    true,
  );

  const mid = midRefine ? acquireIsoMidGBuffer() : null;
  if (mid && upsampleIso(handles, coarseSrc, mid)) {
    drawIsoRefine(handles, mid, coarseOccl, Mgrid, isoSteps, half, scale, ro, dirMatrix);
    if (upsampleIso(handles, { color: mid.color, occl: mid.occl, normal: mid.normal }, fine)) {
      drawIsoRefine(handles, fine, mid.occl, Mgrid, isoSteps, half, scale, ro, dirMatrix);
      return "mid";
    }
  }

  if (upsampleIso(handles, coarseSrc, fine)) {
    drawIsoRefine(handles, fine, coarseOccl, Mgrid, isoSteps, half, scale, ro, dirMatrix);
    return "two";
  }
  return false;
}

function drawBeerPass(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  volColorView: GPUTextureView,
  volW: number,
  volH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): void {
  const { device, beerPipeline, drawParamBufBeer, volumeBuf, colorBuf } = handles;
  const occlIsoView = targets.occlIsoTex.createView();
  const occlSurfView = targets.occlSurfTex.createView();

  gpuWriteBuffer(
    device,
    drawParamBufBeer,
    packDrawParamsBeer(
      targets.volColorTex.width,
      targets.volColorTex.height,
      Mgrid,
      steps,
      half,
      scale,
      gpu.densBase,
      gpu.densLayerCount,
      ro,
      dirMatrix,
      gpu.flowLayerStart,
    ),
  );
  const bg = device.createBindGroup({
    layout: beerPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: drawParamBufBeer } },
      { binding: 1, resource: { buffer: volumeBuf } },
      { binding: 2, resource: occlIsoView },
      { binding: 3, resource: { buffer: colorBuf } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view: volColorView,
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        storeOp: "store",
      },
      { view: occlSurfView, loadOp: "clear", clearValue: { r: 1, g: 0, b: 0, a: 1 }, storeOp: "store" },
    ],
  });
  pass.setPipeline(beerPipeline);
  pass.setBindGroup(0, bg);
  setPassViewport(pass, targets.volColorTex.width, targets.volColorTex.height);
  pass.draw(3);
  pass.end();
  submitEnc(device, enc);
}

function compositeVolumeOntoScene(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  sceneView: GPUTextureView,
  occTex: GPUTexture,
): void {
  if (!gpu.blitPipeline || !gpu.blitSampler) return;
  const { device } = handles;
  const bg = device.createBindGroup({
    layout: gpu.blitPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: targets.volColorTex.createView() },
      { binding: 1, resource: gpu.blitSampler },
      { binding: 2, resource: texView(targets.occlIsoTex) },
      { binding: 3, resource: texView(occTex) },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: sceneView,
      loadOp: "load",
      storeOp: "store",
    }],
  });
  pass.setPipeline(gpu.blitPipeline);
  pass.setBindGroup(0, bg);
  setPassViewport(pass, targets.sceneColorTex.width, targets.sceneColorTex.height);
  pass.draw(3);
  pass.end();
  submitEnc(device, enc);
}

/** Iso-res beer with tExit clip, only on occupancy-refine tiles. */
function drawBeerRefine(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  sceneView: GPUTextureView,
  occTex: GPUTexture,
  destW: number,
  destH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): void {
  const { device, beerRefinePipeline, drawParamBufBeer, volumeBuf, colorBuf } = handles;
  gpuWriteBuffer(
    device,
    drawParamBufBeer,
    packDrawParamsBeer(
      destW, destH, Mgrid, steps, half, scale,
      gpu.densBase, gpu.densLayerCount, ro, dirMatrix, gpu.flowLayerStart,
    ),
  );
  const bg = device.createBindGroup({
    layout: beerRefinePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: drawParamBufBeer } },
      { binding: 1, resource: { buffer: volumeBuf } },
      { binding: 2, resource: texView(targets.occlIsoTex) },
      { binding: 3, resource: { buffer: colorBuf } },
      { binding: 4, resource: texView(occTex) },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: sceneView,
      loadOp: "load",
      storeOp: "store",
    }],
  });
  pass.setPipeline(beerRefinePipeline);
  pass.setBindGroup(0, bg);
  setPassViewport(pass, destW, destH);
  pass.draw(3);
  pass.end();
  submitEnc(device, enc);
}

function drawFxaaPass(
  handles: MarchGpuHandles,
  sceneView: GPUTextureView,
  swapTex: GPUTexture,
  stampEnd: boolean,
): void {
  const { device, fxaaPipeline, fxaaParamBuf, fxaaSampler } = handles;
  const destW = Math.max(1, swapTex.width);
  const destH = Math.max(1, swapTex.height);
  const inv = new Float32Array([1 / destW, 1 / destH, 0, 0]);
  gpuWriteBuffer(device, fxaaParamBuf, inv);
  const bg = device.createBindGroup({
    layout: fxaaPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: fxaaParamBuf } },
      { binding: 1, resource: sceneView },
      { binding: 2, resource: fxaaSampler },
    ],
  });
  const enc = device.createCommandEncoder();
  const passDesc: GPURenderPassDescriptor = {
    colorAttachments: [{
      view: swapTex.createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  };
  const pass = enc.beginRenderPass(stampEnd ? withStampWrites(passDesc, "end") : passDesc);
  pass.setPipeline(fxaaPipeline);
  pass.setBindGroup(0, bg);
  setPassViewport(pass, destW, destH);
  pass.draw(3);
  pass.end();
  submitEnc(device, enc);
}

function drawGridOverlay(
  handles: MarchGpuHandles,
  occlForGrid: GPUTextureView,
  camera: PerspectiveCamera,
  swapView: GPUTextureView,
  half: number,
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
  ro: [number, number, number],
  outW: number,
  outH: number,
): void {
  if (!gpu.gridVertexBuf || gpu.gridVertexCount <= 0) return;

  const { device, gridPipeline, gridParamBuf } = handles;

  camera.updateMatrixWorld(true);
  const viewProj = new Float32Array(16);
  const e = camera.projectionMatrix.elements;
  const v = camera.matrixWorldInverse.elements;
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      viewProj[c * 4 + r] =
        e[0 * 4 + r] * v[c * 4 + 0] +
        e[1 * 4 + r] * v[c * 4 + 1] +
        e[2 * 4 + r] * v[c * 4 + 2] +
        e[3 * 4 + r] * v[c * 4 + 3];
    }
  }
  gpuWriteBuffer(
    device,
    gridParamBuf,
    packGridParams(viewProj, ro, half, dirMatrix, outW, outH),
  );
  const labelVertCount = gpu.labelPipeline
    ? uploadAxisLabelBillboards(camera, half)
    : 0;
  const bg = device.createBindGroup({
    layout: gridPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gridParamBuf } },
      { binding: 1, resource: occlForGrid },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: swapView, loadOp: "load", storeOp: "store" }],
  });
  pass.setPipeline(gridPipeline);
  pass.setBindGroup(0, bg);
  pass.setVertexBuffer(0, gpu.gridVertexBuf);
  setPassViewport(pass, outW, outH);
  pass.draw(gpu.gridVertexCount);

  if (gpu.labelPipeline && gpu.labelVertexBuf && gpu.labelAtlasTex && gpu.labelAtlasSamp && labelVertCount > 0) {
    const labelPipeline = gpu.labelPipeline;
    const labelBg = device.createBindGroup({
      layout: labelPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gridParamBuf } },
        { binding: 1, resource: gpu.labelAtlasTex.createView() },
        { binding: 2, resource: gpu.labelAtlasSamp },
        { binding: 3, resource: occlForGrid },
      ],
    });
    pass.setPipeline(labelPipeline);
    pass.setBindGroup(0, labelBg);
    pass.setVertexBuffer(0, gpu.labelVertexBuf);
    pass.draw(labelVertCount);
  }

  pass.end();
  submitEnc(device, enc);
}

export function renderClipFrameGpu(params: RenderClipFrameGpuParams): boolean {
  const setup = buildRaySetup(params);
  if (!setup) return false;

  const {
    handles, targets, ro, dirMatrix, half, marchW, marchH, volW, volH,
    composeW, composeH, outW, outH,
  } = setup;
  let { refine, midRefine } = setup;
  const { device, ctx, volumeBuf, colorBuf } = handles;
  const { camera, scale, steps } = params;
  const isoSteps = clampIsoStepsForTier(params.isoSteps ?? steps, state.deviceTier);
  const occSteps = coarseIsoSteps(isoSteps, state.deviceTier);
  const volumeSteps = Math.min(96, Math.max(8, steps | 0));
  const Mgrid = gpu.sceneM;
  const sameRes = volW === composeW && volH === composeH;

  if (
    refine &&
    (!gpu.isoCoarseColorTex || !gpu.isoCoarseOcclTex ||
      !gpu.isoCoarseNormalTex || !gpu.isoCoarseDepthTex)
  ) {
    refine = false;
    midRefine = false;
  }
  if (
    midRefine &&
    (!gpu.isoMidColorTex || !gpu.isoMidOcclTex ||
      !gpu.isoMidNormalTex || !gpu.isoMidDepthTex)
  ) {
    midRefine = false;
  }

  gpu.profileMarchFbW = composeW;
  gpu.profileMarchFbH = composeH;
  gpu.profileGridM = Mgrid;

  beginGpuFrame();

  if (gpu.scenePacked && gpu.volumeBuf && gpu.volumeUploadEpoch !== gpu.sceneEpoch) {
    gpuWriteBuffer(device, volumeBuf, gpu.scenePacked);
    gpu.volumeUploadEpoch = gpu.sceneEpoch;
  }

  const swapTex = ctx.getCurrentTexture();

  let refinePath: "mid" | "two" | false = false;
  if (refine) {
    refinePath = runIsoRefineLadder(
      handles, targets, marchW, marchH, midRefine,
      Mgrid, occSteps, isoSteps, half, scale, ro, dirMatrix,
    );
    if (!refinePath) {
      refine = false;
      drawIsoConstraints(
        handles,
        targets.sceneColorTex,
        targets.occlIsoTex,
        targets.normalTex,
        targets.depthTex,
        composeW, composeH, Mgrid, isoSteps, half, scale, ro, dirMatrix,
      );
    }
  } else if (gpu.sceneConstraints.length > 0) {
    drawIsoConstraints(
      handles,
      targets.sceneColorTex,
      targets.occlIsoTex,
      targets.normalTex,
      targets.depthTex,
      composeW, composeH, Mgrid, isoSteps, half, scale, ro, dirMatrix,
      true,
    );
  } else {
    clearMarchTargets(device, targets);
  }

  const sceneView = targets.sceneColorTex.createView();
  const presentW = targets.sceneColorTex.width;
  const presentH = targets.sceneColorTex.height;

  const ranBeer = gpu.densLayerCount > 0 && gpu.densPacked;
  if (ranBeer) {
    if (sameRes) {
      const occlIsoView = targets.occlIsoTex.createView();
      const occlSurfView = targets.occlSurfTex.createView();
      gpuWriteBuffer(
        device,
        handles.drawParamBufBeer,
        packDrawParamsBeer(
          presentW, presentH, Mgrid, volumeSteps, half, scale,
          gpu.densBase, gpu.densLayerCount, ro, dirMatrix, gpu.flowLayerStart,
        ),
      );
      const bg = device.createBindGroup({
        layout: handles.beerPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: handles.drawParamBufBeer } },
          { binding: 1, resource: { buffer: volumeBuf } },
          { binding: 2, resource: occlIsoView },
          { binding: 3, resource: { buffer: colorBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: sceneView, loadOp: "load", storeOp: "store" },
          { view: occlSurfView, loadOp: "clear", clearValue: { r: 1, g: 0, b: 0, a: 1 }, storeOp: "store" },
        ],
      });
      pass.setPipeline(handles.beerPipeline);
      pass.setBindGroup(0, bg);
      setPassViewport(pass, presentW, presentH);
      pass.draw(3);
      pass.end();
      submitEnc(device, enc);
    } else {
      const occTex = (midRefine && gpu.isoMidOcclTex) || gpu.isoCoarseOcclTex || targets.occlIsoTex;
      drawBeerPass(
        handles, targets, targets.volColorTex.createView(),
        volW, volH, Mgrid, volumeSteps, half, scale, ro, dirMatrix,
      );
      compositeVolumeOntoScene(handles, targets, sceneView, occTex);
      if (gpu.sceneConstraints.length > 0) {
        drawBeerRefine(
          handles, targets, sceneView, occTex,
          presentW, presentH, Mgrid, volumeSteps, half, scale, ro, dirMatrix,
        );
      }
    }
  }

  const ranFlow = hasFlowGpuLayers();
  if (ranFlow) {
    const viewDir: [number, number, number] = [
      -dirMatrix[2],
      -dirMatrix[5],
      -dirMatrix[8],
    ];
    tickFlowParticles(ro, viewDir);
    drawFlowParticlesPass(
      camera,
      sceneView,
      targets.occlIsoTex.createView(),
      ro,
      dirMatrix,
      half,
      presentW,
      presentH,
    );
  }

  const occlForGrid = ranBeer
    ? targets.occlSurfTex.createView()
    : targets.occlIsoTex.createView();
  const ranGrid = !!state.showGridAxes;
  // Present iso at display res first, then stroke the grid on the swapchain.
  // Occlusion still samples the compose-sized depth; grid.wgsl maps display
  // clip → occl texels with dims/fbW. Drawing lines into the 2× compose
  // buffer made axes look like fat pixel art after FXAA upsample.
  drawFxaaPass(handles, sceneView, swapTex, true);
  if (ranGrid) {
    drawGridOverlay(
      handles, occlForGrid, camera, swapTex.createView(), half, dirMatrix, ro, outW, outH,
    );
  }

  const method: string[] = [
    refinePath === "mid" ? "gpu-iso-refine-mid" : refine ? "gpu-iso-refine" : "gpu-iso",
  ];
  if (ranBeer) method.push(sameRes ? "beer" : "beer(volFB)+occl+iso-tiles");
  if (ranFlow) method.push("flow");
  method.push("fxaa");
  if (ranGrid) method.push("grid");
  gpu.profileMethod = method.join("+");

  const submitWallAt = performance.now();
  endGpuFrame(device);
  sampleGpuPresent(submitWallAt);
  return true;
}
