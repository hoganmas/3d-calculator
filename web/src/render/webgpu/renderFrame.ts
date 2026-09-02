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
  packSsaoParams,
} from "./uniforms.js";
import { packGridParams, syncClipGpuWorldGrid, uploadAxisLabelBillboards } from "./gridOverlay.js";
import { state } from "../../app/state.js";
import { ensureMarchTargets, ensureIsoCoarseTargets, ensureVolumeTargets, resizeClipGpuCanvas } from "./marchCanvas.js";
import {
  isoFineFramebufferSize,
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
  const refine = gpu.sceneConstraints.length > 0 && isoRefineEnabled(marchW, marchH, outW, outH);
  const { fw: composeW, fh: composeH } = refine
    ? isoFineFramebufferSize(marchW, marchH, outW, outH)
    : { fw: marchW, fh: marchH };

  resizeClipGpuCanvas(outW, outH);
  if (refine) ensureIsoCoarseTargets(marchW, marchH);
  ensureMarchTargets(composeW, composeH);
  ensureVolumeTargets(volW, volH);
  syncClipGpuWorldGrid(h);

  const targets = acquireMarchTargets();
  if (!targets) return null;

  return {
    handles, targets, ro, dirMatrix, half: h,
    marchW, marchH, volW, volH, outW, outH, composeW, composeH, refine,
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
): void {
  const { device, isoPipeline, drawParamBuf, volumeBuf } = handles;
  const sceneView = sceneTex.createView();
  const occlIsoView = occlTex.createView();
  const depthView = depthTex.createView();
  const normalView = normalTex.createView();

  for (let ci = 0; ci < gpu.sceneConstraints.length; ci++) {
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
      drawParamBuf,
      packDrawParamsIso(
        sceneTex.width, sceneTex.height, Mgrid, steps, half, scale, c.isoLevel, base0, ro, dirMatrix,
        c0, c1,
        base1, blendT, stops,
        false,
        ci + 1,
      ),
    );
    const bg = device.createBindGroup({
      layout: isoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: drawParamBuf } },
        { binding: 1, resource: { buffer: volumeBuf } },
      ],
    });
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
    pass.setPipeline(isoPipeline);
    pass.setBindGroup(0, bg);
    setPassViewport(pass, sceneTex.width, sceneTex.height);
    pass.draw(3);
    pass.end();
    submitEnc(device, enc);
  }
}

function upsampleIsoToFine(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  composeW: number,
  composeH: number,
): boolean {
  const color = gpu.isoCoarseColorTex;
  const occl = gpu.isoCoarseOcclTex;
  const normal = gpu.isoCoarseNormalTex;
  if (!color || !occl || !normal) return false;
  const { device, isoUpsamplePipeline, isoUpsampleParamBuf } = handles;
  const fw = targets.sceneColorTex.width;
  const fh = targets.sceneColorTex.height;
  const debug = isIsoRefineDebugEnabled() ? 1 : 0;
  const params = new Uint32Array([fw | 0, fh | 0, debug, 0]);
  gpuWriteBuffer(device, isoUpsampleParamBuf, params);
  const bg = device.createBindGroup({
    layout: isoUpsamplePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: isoUpsampleParamBuf } },
      { binding: 1, resource: color.createView() },
      { binding: 2, resource: occl.createView() },
      { binding: 3, resource: normal.createView() },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      { view: targets.sceneColorTex.createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" },
      { view: targets.occlIsoTex.createView(), loadOp: "clear", clearValue: { r: 1, g: 0, b: 0, a: 1 }, storeOp: "store" },
      { view: targets.normalTex.createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" },
    ],
    depthStencilAttachment: {
      view: targets.depthTex.createView(),
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
  targets: MarchTargets,
  composeW: number,
  composeH: number,
  Mgrid: number,
  steps: number,
  half: number,
  scale: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): void {
  const coarseOccl = gpu.isoCoarseOcclTex;
  if (!coarseOccl) return;
  const { device, isoRefinePipeline, drawParamBuf, volumeBuf } = handles;
  const sceneView = targets.sceneColorTex.createView();
  const occlIsoView = targets.occlIsoTex.createView();
  const depthView = targets.depthTex.createView();
  const normalView = targets.normalTex.createView();
  const coarseOcclView = coarseOccl.createView();

  for (let ci = 0; ci < gpu.sceneConstraints.length; ci++) {
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
      drawParamBuf,
      packDrawParamsIso(
        targets.sceneColorTex.width, targets.sceneColorTex.height,
        Mgrid, steps, half, scale, c.isoLevel, base0, ro, dirMatrix,
        c0, c1,
        base1, blendT, stops,
        isIsoRefineDebugEnabled(),
        ci + 1,
      ),
    );
    const bg = device.createBindGroup({
      layout: isoRefinePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: drawParamBuf } },
        { binding: 1, resource: { buffer: volumeBuf } },
        { binding: 2, resource: coarseOcclView },
      ],
    });
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
    pass.setBindGroup(0, bg);
    setPassViewport(pass, targets.sceneColorTex.width, targets.sceneColorTex.height);
    pass.draw(3);
    pass.end();
    submitEnc(device, enc);
  }
}

function drawSsaoPass(
  handles: MarchGpuHandles,
  targets: MarchTargets,
  sceneView: GPUTextureView,
  marchW: number,
  marchH: number,
  half: number,
  ro: [number, number, number],
  dirMatrix: ReturnType<typeof offsetDirMatrix>,
): GPUTextureView {
  const { device, ssaoPipeline, ssaoParamBuf } = handles;
  const aoView = targets.sceneColorAoTex.createView();
  const occlIsoView = targets.occlIsoTex.createView();
  const normalView = targets.normalTex.createView();

  gpuWriteBuffer(
    device,
    ssaoParamBuf,
    packSsaoParams(
      targets.sceneColorAoTex.width, targets.sceneColorAoTex.height, half,
      Math.max(0.2, half * 0.18),
      0.85,
      0.03,
      ro, dirMatrix,
    ),
  );
  const bg = device.createBindGroup({
    layout: ssaoPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ssaoParamBuf } },
      { binding: 1, resource: sceneView },
      { binding: 2, resource: occlIsoView },
      { binding: 3, resource: normalView },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: aoView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(ssaoPipeline);
  pass.setBindGroup(0, bg);
  setPassViewport(pass, targets.sceneColorAoTex.width, targets.sceneColorAoTex.height);
  pass.draw(3);
  pass.end();
  submitEnc(device, enc);
  return aoView;
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
): void {
  if (!gpu.blitPipeline || !gpu.blitSampler) return;
  const { device } = handles;
  const bg = device.createBindGroup({
    layout: gpu.blitPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: targets.volColorTex.createView() },
      { binding: 1, resource: gpu.blitSampler },
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
    composeW, composeH,
  } = setup;
  let { refine } = setup;
  const { device, ctx, volumeBuf, colorBuf } = handles;
  const { camera, scale, steps } = params;
  const isoSteps = Math.min(192, Math.max(16, (params.isoSteps ?? steps) | 0));
  const volumeSteps = Math.min(96, Math.max(8, steps | 0));
  const Mgrid = gpu.sceneM;
  const sameRes = volW === composeW && volH === composeH;

  if (
    refine &&
    (!gpu.isoCoarseColorTex || !gpu.isoCoarseOcclTex ||
      !gpu.isoCoarseNormalTex || !gpu.isoCoarseDepthTex)
  ) {
    refine = false;
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

  if (refine) {
    clearIsoGBuffer(
      device,
      gpu.isoCoarseColorTex!,
      gpu.isoCoarseOcclTex!,
      gpu.isoCoarseNormalTex!,
      gpu.isoCoarseDepthTex!,
      true,
    );
    drawIsoConstraints(
      handles,
      gpu.isoCoarseColorTex!,
      gpu.isoCoarseOcclTex!,
      gpu.isoCoarseNormalTex!,
      gpu.isoCoarseDepthTex!,
      marchW, marchH, Mgrid, isoSteps, half, scale, ro, dirMatrix,
    );
    if (upsampleIsoToFine(handles, targets, composeW, composeH)) {
      drawIsoRefine(
        handles, targets, composeW, composeH, Mgrid, isoSteps, half, scale, ro, dirMatrix,
      );
    } else {
      refine = false;
      clearMarchTargets(device, targets);
      drawIsoConstraints(
        handles,
        targets.sceneColorTex,
        targets.occlIsoTex,
        targets.normalTex,
        targets.depthTex,
        composeW, composeH, Mgrid, isoSteps, half, scale, ro, dirMatrix,
      );
    }
  } else {
    clearMarchTargets(device, targets);
    drawIsoConstraints(
      handles,
      targets.sceneColorTex,
      targets.occlIsoTex,
      targets.normalTex,
      targets.depthTex,
      composeW, composeH, Mgrid, isoSteps, half, scale, ro, dirMatrix,
    );
  }

  let sceneView = targets.sceneColorTex.createView();
  const sceneFbW = targets.sceneColorTex.width;
  const sceneFbH = targets.sceneColorTex.height;

  const ranSsao = gpu.sceneConstraints.length > 0;
  if (ranSsao) {
    sceneView = drawSsaoPass(handles, targets, sceneView, composeW, composeH, half, ro, dirMatrix);
  }
  const presentW = ranSsao ? targets.sceneColorAoTex.width : sceneFbW;
  const presentH = ranSsao ? targets.sceneColorAoTex.height : sceneFbH;

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
      drawBeerPass(
        handles, targets, targets.volColorTex.createView(),
        volW, volH, Mgrid, volumeSteps, half, scale, ro, dirMatrix,
      );
      compositeVolumeOntoScene(handles, targets, sceneView);
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
  // Rasterize the box onto the compose buffer with the iso so they share one
  // present path. Drawing the grid on the swap chain after FXAA let a UV/size
  // mismatch slide the surface off the wireframe.
  if (ranGrid) {
    drawGridOverlay(
      handles, occlForGrid, camera, sceneView, half, dirMatrix, ro, presentW, presentH,
    );
  }

  drawFxaaPass(handles, sceneView, swapTex, true);

  const method: string[] = [refine ? "gpu-iso-refine" : "gpu-iso"];
  if (ranSsao) method.push("ssao");
  if (ranBeer) method.push(sameRes ? "beer" : "beer(volFB)+blit");
  if (ranFlow) method.push("flow");
  method.push("fxaa");
  if (ranGrid) method.push("grid");
  gpu.profileMethod = method.join("+");

  const submitWallAt = performance.now();
  endGpuFrame(device);
  sampleGpuPresent(submitWallAt);
  return true;
}
